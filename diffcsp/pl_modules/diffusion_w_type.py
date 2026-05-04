import math, copy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from typing import Any, Dict

import hydra
import omegaconf
import pytorch_lightning as pl
from torch_scatter import scatter
from torch_scatter.composite import scatter_softmax
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from tqdm import tqdm

from diffcsp.common.utils import PROJECT_ROOT
from diffcsp.common.data_utils import (
    EPSILON, cart_to_frac_coords, mard, lengths_angles_to_volume, lattice_params_to_matrix_torch,
    frac_to_cart_coords, min_distance_sqr_pbc)

from diffcsp.pl_modules.diff_utils import (
    d_log_p_wrapped_normal,
    gaussian_log_prob,
    log_p_wrapped_normal,
)

MAX_ATOMIC_NUM=100


class BaseModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # populate self.hparams with args and kwargs automagically!
        self.save_hyperparameters()
        if hasattr(self.hparams, "model"):
            self._hparams = self.hparams.model


    def configure_optimizers(self):
        opt = hydra.utils.instantiate(
            self.hparams.optim.optimizer, params=self.parameters(), _convert_="partial"
        )
        if not self.hparams.optim.use_lr_scheduler:
            return [opt]
        scheduler = hydra.utils.instantiate(
            self.hparams.optim.lr_scheduler, optimizer=opt
        )
        return {"optimizer": opt, "lr_scheduler": scheduler, "monitor": "val_loss"}


### Model definition

class SinusoidalTimeEmbeddings(nn.Module):
    """ Attention is all you need. """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class CSPDiffusion(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.decoder = hydra.utils.instantiate(self.hparams.decoder, latent_dim = self.hparams.latent_dim + self.hparams.time_dim, pred_type = True, smooth = True)
        self.beta_scheduler = hydra.utils.instantiate(self.hparams.beta_scheduler)
        self.sigma_scheduler = hydra.utils.instantiate(self.hparams.sigma_scheduler)
        self.time_dim = self.hparams.time_dim
        self.time_embedding = SinusoidalTimeEmbeddings(self.time_dim)
        self.keep_lattice = self.hparams.cost_lattice < 1e-5
        self.keep_coords = self.hparams.cost_coord < 1e-5
        self.p_uncond = self.hparams.p_uncond
        ##self.guide_w = self.hparams.guide_w

    def _prepare_band_gap(self, band_gap, batch_size):
        if not torch.is_tensor(band_gap):
            band_gap = torch.tensor(band_gap, dtype=torch.float32, device=self.device)
        band_gap = band_gap.to(self.device).float().view(-1)
        if band_gap.numel() == 1 and batch_size > 1:
            band_gap = band_gap.repeat(batch_size)
        return band_gap

    def _clone_state(self, state):
        return {
            'num_atoms': state['num_atoms'].clone(),
            'atom_types': state['atom_types'].clone(),
            'frac_coords_raw': state['frac_coords_raw'].clone(),
            'frac_coords': state['frac_coords'].clone(),
            'lattices': state['lattices'].clone(),
        }

    def _state_for_traj(self, state):
        return {
            'num_atoms': state['num_atoms'],
            'atom_types': state['atom_types'],
            'frac_coords': state['frac_coords'],
            'lattices': state['lattices'],
        }

    def _model_dtype(self):
        return next(self.parameters()).dtype

    def _init_sample_state(self, batch):
        batch_size = batch.num_graphs
        dtype = self._model_dtype()
        l_T = torch.randn([batch_size, 3, 3], device=self.device, dtype=dtype)
        x_T = torch.rand([batch.num_nodes, 3], device=self.device, dtype=dtype)
        t_T = torch.randn([batch.num_nodes, MAX_ATOMIC_NUM], device=self.device, dtype=dtype)

        if self.keep_coords:
            x_T = batch.frac_coords.to(device=self.device, dtype=dtype)

        if self.keep_lattice:
            l_T = lattice_params_to_matrix_torch(batch.lengths, batch.angles).to(dtype=dtype)

        return {
            'num_atoms': batch.num_atoms.clone(),
            'atom_types': t_T,
            'frac_coords_raw': x_T,
            'frac_coords': x_T % 1.,
            'lattices': l_T,
        }

    def _guided_prediction(self, time_emb, atom_types, frac_coords, lattices, batch, band_gap, guide_w, sigma_norm):
        batch_size = batch.num_graphs
        dtype = self._model_dtype()
        property_on = torch.ones(batch_size, device=self.device, dtype=dtype)
        property_off = torch.zeros(batch_size, device=self.device, dtype=dtype)
        time_emb = time_emb.to(dtype=dtype)
        atom_types = atom_types.to(dtype=dtype)
        frac_coords = frac_coords.to(dtype=dtype)
        lattices = lattices.to(dtype=dtype)
        band_gap = band_gap.to(dtype=dtype)
        sigma_norm = sigma_norm.to(dtype=dtype)

        pred_l1, pred_x1, pred_t1 = self.decoder(
            time_emb, atom_types, frac_coords, lattices, batch.num_atoms, batch.batch, band_gap, property_on
        )
        pred_x1 = pred_x1 * torch.sqrt(sigma_norm)

        pred_l2, pred_x2, pred_t2 = self.decoder(
            time_emb, atom_types, frac_coords, lattices, batch.num_atoms, batch.batch, band_gap, property_off
        )
        pred_x2 = pred_x2 * torch.sqrt(sigma_norm)

        pred_x = (1 + guide_w) * pred_x1 - guide_w * pred_x2
        pred_l = (1 + guide_w) * pred_l1 - guide_w * pred_l2
        pred_t = (1 + guide_w) * pred_t1 - guide_w * pred_t2
        return pred_l, pred_x, pred_t

    def _predict_clean_state(self, batch, state, t, band_gap, guide_w):
        if t == 0:
            return {
                'num_atoms': state['num_atoms'].clone(),
                'atom_types': state['atom_types'].argmax(dim=-1) + 1,
                'frac_coords': state['frac_coords'].clone(),
                'lattices': state['lattices'].clone(),
            }

        times = torch.full((batch.num_graphs,), t, device=self.device)
        time_emb = self.time_embedding(times)
        alpha_bar = self.beta_scheduler.alphas_cumprod[t]
        sigma_x = self.sigma_scheduler.sigmas[t]
        sigma_norm = self.sigma_scheduler.sigmas_norm[t]
        c0_bar = torch.sqrt(alpha_bar)
        c1_bar = torch.sqrt(1. - alpha_bar)

        pred_l, pred_x, pred_t = self._guided_prediction(
            time_emb,
            state['atom_types'],
            state['frac_coords'],
            state['lattices'],
            batch,
            band_gap,
            guide_w,
            sigma_norm,
        )

        frac_coords = (state['frac_coords'] - (sigma_x ** 2) * pred_x) % 1.
        lattices = (state['lattices'] - c1_bar * pred_l) / c0_bar
        atom_type_logits = (state['atom_types'] - c1_bar * pred_t) / c0_bar

        return {
            'num_atoms': state['num_atoms'].clone(),
            'atom_types': atom_type_logits.argmax(dim=-1) + 1,
            'frac_coords': frac_coords,
            'lattices': lattices,
        }

    def _sample_step(self, batch, state, t, band_gap, guide_w, step_lr=1e-5, capture_buffer=False):
        batch_size = batch.num_graphs
        times = torch.full((batch_size,), t, device=self.device)
        time_emb = self.time_embedding(times)

        alphas = self.beta_scheduler.alphas[t]
        alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]
        sigmas = self.beta_scheduler.sigmas[t]
        sigma_x = self.sigma_scheduler.sigmas[t]
        sigma_norm = self.sigma_scheduler.sigmas_norm[t]

        c0 = 1.0 / torch.sqrt(alphas)
        c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)

        x_t_raw = state['frac_coords_raw']
        x_t = state['frac_coords']
        l_t = state['lattices']
        t_t = state['atom_types']

        if self.keep_coords:
            x_t_raw = batch.frac_coords
            x_t = batch.frac_coords

        if self.keep_lattice:
            l_t = lattice_params_to_matrix_torch(batch.lengths, batch.angles)

        pred_l_corr, pred_x_corr, _ = self._guided_prediction(
            time_emb, t_t, x_t, l_t, batch, band_gap, guide_w, sigma_norm
        )

        rand_x_corr = torch.randn_like(x_t_raw) if t > 1 else torch.zeros_like(x_t_raw)
        step_size_corr = step_lr * (sigma_x / self.sigma_scheduler.sigma_begin) ** 2
        std_x_corr = torch.sqrt(2 * step_size_corr)
        mean_x_corr = x_t_raw - step_size_corr * pred_x_corr if not self.keep_coords else x_t_raw
        x_mid_raw = mean_x_corr + std_x_corr * rand_x_corr if not self.keep_coords else x_t_raw
        x_mid = x_mid_raw % 1.

        l_mid = l_t
        t_mid = t_t

        pred_l_pred, pred_x_pred, pred_t_pred = self._guided_prediction(
            time_emb, t_mid, x_mid, l_mid, batch, band_gap, guide_w, sigma_norm
        )

        rand_l_pred = torch.randn_like(l_mid) if t > 1 else torch.zeros_like(l_mid)
        rand_t_pred = torch.randn_like(t_mid) if t > 1 else torch.zeros_like(t_mid)
        rand_x_pred = torch.randn_like(x_mid_raw) if t > 1 else torch.zeros_like(x_mid_raw)

        adjacent_sigma_x = self.sigma_scheduler.sigmas[t - 1]
        step_size_pred = sigma_x ** 2 - adjacent_sigma_x ** 2
        std_x_pred = torch.sqrt(
            (adjacent_sigma_x ** 2 * (sigma_x ** 2 - adjacent_sigma_x ** 2)) / (sigma_x ** 2)
        )

        mean_x_pred = x_mid_raw - step_size_pred * pred_x_pred if not self.keep_coords else x_mid_raw
        x_next_raw = mean_x_pred + std_x_pred * rand_x_pred if not self.keep_coords else x_mid_raw

        mean_l_pred = c0 * (l_mid - c1 * pred_l_pred) if not self.keep_lattice else l_mid
        l_next = mean_l_pred + sigmas * rand_l_pred if not self.keep_lattice else l_mid

        mean_t_pred = c0 * (t_mid - c1 * pred_t_pred)
        t_next = mean_t_pred + sigmas * rand_t_pred

        next_state = {
            'num_atoms': state['num_atoms'].clone(),
            'atom_types': t_next,
            'frac_coords_raw': x_next_raw,
            'frac_coords': x_next_raw % 1.,
            'lattices': l_next,
        }

        if not capture_buffer:
            return next_state, None

        logp_corr = (
            log_p_wrapped_normal(x_mid - mean_x_corr, std_x_corr).sum()
            if not self.keep_coords
            else torch.zeros([], device=self.device)
        )
        logp_pred_x = (
            log_p_wrapped_normal(next_state['frac_coords'] - mean_x_pred, std_x_pred).sum()
            if not self.keep_coords
            else torch.zeros([], device=self.device)
        )
        logp_pred_l = (
            gaussian_log_prob(l_next, mean_l_pred, sigmas).sum()
            if not self.keep_lattice
            else torch.zeros([], device=self.device)
        )
        logp_pred_t = gaussian_log_prob(t_next, mean_t_pred, sigmas).sum()

        step_buffer = {
            'time': int(t),
            'state_atom_types': t_t.detach().clone(),
            'state_frac_coords': x_t.detach().clone(),
            'state_frac_coords_raw': x_t_raw.detach().clone(),
            'state_lattices': l_t.detach().clone(),
            'mid_atom_types': t_mid.detach().clone(),
            'mid_frac_coords': x_mid.detach().clone(),
            'mid_frac_coords_raw': x_mid_raw.detach().clone(),
            'mid_lattices': l_mid.detach().clone(),
            'next_atom_types': t_next.detach().clone(),
            'next_frac_coords': next_state['frac_coords'].detach().clone(),
            'next_frac_coords_raw': x_next_raw.detach().clone(),
            'next_lattices': l_next.detach().clone(),
            'old_logp_corrector': logp_corr.detach().clone(),
            'old_logp_predictor': (logp_pred_x + logp_pred_l + logp_pred_t).detach().clone(),
        }
        return next_state, step_buffer

    def compute_step_log_probs_with_step_lr(self, batch, band_gap, guide_w, step_buffer, step_lr):
        t = int(step_buffer['time'])
        batch_size = batch.num_graphs
        times = torch.full((batch_size,), t, device=self.device)
        time_emb = self.time_embedding(times)

        alphas = self.beta_scheduler.alphas[t]
        alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]
        sigmas = self.beta_scheduler.sigmas[t]
        sigma_x = self.sigma_scheduler.sigmas[t]
        sigma_norm = self.sigma_scheduler.sigmas_norm[t]

        c0 = 1.0 / torch.sqrt(alphas)
        c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)

        pred_l_corr, pred_x_corr, _ = self._guided_prediction(
            time_emb,
            step_buffer['state_atom_types'],
            step_buffer['state_frac_coords'],
            step_buffer['state_lattices'],
            batch,
            band_gap,
            guide_w,
            sigma_norm,
        )

        step_size_corr = step_lr * (sigma_x / self.sigma_scheduler.sigma_begin) ** 2
        std_x_corr = torch.sqrt(2 * step_size_corr)
        mean_x_corr = (
            step_buffer['state_frac_coords_raw'] - step_size_corr * pred_x_corr
            if not self.keep_coords
            else step_buffer['state_frac_coords_raw']
        )

        pred_l_pred, pred_x_pred, pred_t_pred = self._guided_prediction(
            time_emb,
            step_buffer['mid_atom_types'],
            step_buffer['mid_frac_coords'],
            step_buffer['mid_lattices'],
            batch,
            band_gap,
            guide_w,
            sigma_norm,
        )

        adjacent_sigma_x = self.sigma_scheduler.sigmas[t - 1]
        step_size_pred = sigma_x ** 2 - adjacent_sigma_x ** 2
        std_x_pred = torch.sqrt(
            (adjacent_sigma_x ** 2 * (sigma_x ** 2 - adjacent_sigma_x ** 2)) / (sigma_x ** 2)
        )

        mean_x_pred = (
            step_buffer['mid_frac_coords_raw'] - step_size_pred * pred_x_pred
            if not self.keep_coords
            else step_buffer['mid_frac_coords_raw']
        )
        mean_l_pred = c0 * (step_buffer['mid_lattices'] - c1 * pred_l_pred) if not self.keep_lattice else step_buffer['mid_lattices']
        mean_t_pred = c0 * (step_buffer['mid_atom_types'] - c1 * pred_t_pred)

        logp_corr = (
            log_p_wrapped_normal(step_buffer['mid_frac_coords'] - mean_x_corr, std_x_corr).sum()
            if not self.keep_coords
            else torch.zeros([], device=self.device)
        )
        logp_pred_x = (
            log_p_wrapped_normal(step_buffer['next_frac_coords'] - mean_x_pred, std_x_pred).sum()
            if not self.keep_coords
            else torch.zeros([], device=self.device)
        )
        logp_pred_l = (
            gaussian_log_prob(step_buffer['next_lattices'], mean_l_pred, sigmas).sum()
            if not self.keep_lattice
            else torch.zeros([], device=self.device)
        )
        logp_pred_t = gaussian_log_prob(step_buffer['next_atom_types'], mean_t_pred, sigmas).sum()

        return {
            'corrector': logp_corr,
            'predictor': logp_pred_x + logp_pred_l + logp_pred_t,
            'total': logp_corr + logp_pred_x + logp_pred_l + logp_pred_t,
        }



    def forward(self, batch):

        batch_size = batch.num_graphs
        times = self.beta_scheduler.uniform_sample_t(batch_size, self.device)
        time_emb = self.time_embedding(times)

        alphas_cumprod = self.beta_scheduler.alphas_cumprod[times]
        beta = self.beta_scheduler.betas[times]

        c0 = torch.sqrt(alphas_cumprod)
        c1 = torch.sqrt(1. - alphas_cumprod)

        sigmas = self.sigma_scheduler.sigmas[times]
        sigmas_norm = self.sigma_scheduler.sigmas_norm[times]

        lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        frac_coords = batch.frac_coords

        rand_l, rand_x = torch.randn_like(lattices), torch.randn_like(frac_coords)

        input_lattice = c0[:, None, None] * lattices + c1[:, None, None] * rand_l
        sigmas_per_atom = sigmas.repeat_interleave(batch.num_atoms)[:, None]
        sigmas_norm_per_atom = sigmas_norm.repeat_interleave(batch.num_atoms)[:, None]
        input_frac_coords = (frac_coords + sigmas_per_atom * rand_x) % 1.

        gt_atom_types_onehot = F.one_hot(batch.atom_types - 1, num_classes=MAX_ATOMIC_NUM).float()

        rand_t = torch.randn_like(gt_atom_types_onehot)

        atom_type_probs = (c0.repeat_interleave(batch.num_atoms)[:, None] * gt_atom_types_onehot + c1.repeat_interleave(batch.num_atoms)[:, None] * rand_t)

        if self.keep_coords:
            input_frac_coords = frac_coords

        if self.keep_lattice:
            input_lattice = lattices


        # Need to apply property here, but before need to bernoulli sample
        property_indicator = torch.bernoulli(torch.ones(batch_size)*(1.-self.p_uncond))
        property_indicator = property_indicator.to(self.device)
        property_train = torch.squeeze(batch.y)

        pred_l, pred_x, pred_t = self.decoder(time_emb, atom_type_probs, input_frac_coords, input_lattice, batch.num_atoms, batch.batch, property_train, property_indicator)

        tar_x = d_log_p_wrapped_normal(sigmas_per_atom * rand_x, sigmas_per_atom) / torch.sqrt(sigmas_norm_per_atom)

        loss_lattice = F.mse_loss(pred_l, rand_l)
        loss_coord = F.mse_loss(pred_x, tar_x)
        loss_type = F.mse_loss(pred_t, rand_t)


        loss = (
            self.hparams.cost_lattice * loss_lattice +
            self.hparams.cost_coord * loss_coord + 
            self.hparams.cost_type * loss_type)

        return {
            'loss' : loss,
            'loss_lattice' : loss_lattice,
            'loss_coord' : loss_coord,
            'loss_type' : loss_type
        }

    @torch.no_grad()
    def sample(self, batch, band_gap, guide_w, diff_ratio = 1.0, step_lr = 1e-5):
        del diff_ratio  # kept for backwards compatibility with callers.
        band_gap = self._prepare_band_gap(band_gap, batch.num_graphs)
        state = self._init_sample_state(batch)

        traj = {self.beta_scheduler.timesteps: self._state_for_traj(state)}
        for t in tqdm(range(self.beta_scheduler.timesteps, 0, -1)):
            state, _ = self._sample_step(
                batch, state, t, band_gap, guide_w, step_lr=step_lr, capture_buffer=False
            )
            traj[t - 1] = self._state_for_traj(state)

        traj_stack = {
            'num_atoms' : batch.num_atoms,
            'atom_types' : torch.stack([traj[i]['atom_types'] for i in range(self.beta_scheduler.timesteps, -1, -1)]).argmax(dim=-1) + 1,
            'all_frac_coords' : torch.stack([traj[i]['frac_coords'] for i in range(self.beta_scheduler.timesteps, -1, -1)]),
            'all_lattices' : torch.stack([traj[i]['lattices'] for i in range(self.beta_scheduler.timesteps, -1, -1)])
        }

        return traj[0], traj_stack

    @torch.no_grad()
    def sample_rl(
        self,
        batch,
        band_gap,
        guide_w,
        num_branches=8,
        prefix_t=300,
        reward_interval=25,
        step_lr=1e-5,
    ):
        if batch.num_graphs != 1:
            raise ValueError("sample_rl currently supports one prompt graph per group.")

        band_gap = self._prepare_band_gap(band_gap, batch.num_graphs)
        state = self._init_sample_state(batch)

        for t in range(self.beta_scheduler.timesteps, prefix_t, -1):
            state, _ = self._sample_step(
                batch, state, t, band_gap, guide_w, step_lr=step_lr, capture_buffer=False
            )

        prefix_state = self._clone_state(state)
        reward_times = set(range(prefix_t, -1, -reward_interval))
        reward_times.add(prefix_t)
        reward_times.add(0)

        branches = []
        for _ in range(num_branches):
            branch_state = self._clone_state(prefix_state)
            branch_transitions = []
            reconstructions = {
                prefix_t: self._predict_clean_state(batch, branch_state, prefix_t, band_gap, guide_w)
            }

            for t in range(prefix_t, 0, -1):
                branch_state, step_buffer = self._sample_step(
                    batch, branch_state, t, band_gap, guide_w, step_lr=step_lr, capture_buffer=True
                )
                branch_transitions.append(step_buffer)
                if (t - 1) in reward_times:
                    reconstructions[t - 1] = self._predict_clean_state(
                        batch, branch_state, t - 1, band_gap, guide_w
                    )

            branches.append(
                {
                    'transitions': branch_transitions,
                    'reconstructions': reconstructions,
                    'final_state': self._predict_clean_state(batch, branch_state, 0, band_gap, guide_w),
                }
            )

        return {
            'prefix_t': prefix_t,
            'reward_times': sorted(reward_times, reverse=True),
            'prefix_latent_state': prefix_state,
            'prefix_clean_state': self._predict_clean_state(batch, prefix_state, prefix_t, band_gap, guide_w),
            'branches': branches,
            'band_gap': band_gap.detach().clone(),
            'guide_w': float(guide_w),
            'step_lr': float(step_lr),
        }



    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        loss_lattice = output_dict['loss_lattice']
        loss_coord = output_dict['loss_coord']
        loss_type = output_dict['loss_type']
        loss = output_dict['loss']


        self.log_dict(
            {'train_loss': loss,
            'lattice_loss': loss_lattice,
            'coord_loss': loss_coord,
            'type_loss': loss_type},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        if loss.isnan():
            return None

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        log_dict, loss = self.compute_stats(output_dict, prefix='val')

        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        log_dict, loss = self.compute_stats(output_dict, prefix='test')

        self.log_dict(
            log_dict,
        )
        return loss

    def compute_stats(self, output_dict, prefix):

        loss_lattice = output_dict['loss_lattice']
        loss_coord = output_dict['loss_coord']
        loss_type = output_dict['loss_type']
        loss = output_dict['loss']

        log_dict = {
            f'{prefix}_loss': loss,
            f'{prefix}_lattice_loss': loss_lattice,
            f'{prefix}_coord_loss': loss_coord,
            f'{prefix}_type_loss': loss_type,
        }

        return log_dict, loss

    
