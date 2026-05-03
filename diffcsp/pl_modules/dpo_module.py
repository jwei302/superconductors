"""Diffusion-DPO LightningModule for DiffCSP.

Three-term DPO logit per pair (winner, loser) with shared (t, m, noise):

    Δ_φ(x) = w_L · ‖ε_L − ε^L_φ‖² + w_F · ‖score_F − s^F_φ‖² + w_A · ‖ε_A − ε^A_φ‖²
    z      = −β·T·[(Δ_θ(w) − Δ_ref(w)) − (Δ_θ(l) − Δ_ref(l))]
    L      = −E[ log σ(z) ]

π_ref's three-component MSEs are pre-cached on disk (see scripts/dpo/build_preference_pairs.py)
so DPO gradient steps only forward π_θ.

Adapter-only training: only `adapters` and `property_embedding` parameters
in the policy's CSPNet decoder receive gradients (see _freeze_non_adapters).
"""

from __future__ import annotations

import math
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter

from diffcsp.common.data_utils import lattice_params_to_matrix_torch
from diffcsp.pl_modules.diff_utils import d_log_p_wrapped_normal
from diffcsp.pl_modules.diffusion_w_type import MAX_ATOMIC_NUM


class DPODiffusion(pl.LightningModule):
    def __init__(
        self,
        policy: nn.Module,
        beta: float = 200.0,
        total_timesteps: int = 1000,
        cost_lattice: float = 1.0,
        cost_coord: float = 1.0,
        cost_type: float = 20.0,
        normalize_components: bool = False,
        train_only_adapters: bool = True,
        lr: float = 1e-4,
    ):
        super().__init__()
        # `policy` is a CSPDiffusion instance with weights already loaded (π_ref's weights).
        # We do not save it as a hyperparameter (it is a big nn.Module).
        self.save_hyperparameters(ignore=["policy"])
        self.policy = policy
        self.beta = beta
        self.T = total_timesteps
        self.cost_lattice = cost_lattice
        self.cost_coord = cost_coord
        self.cost_type = cost_type
        self.normalize_components = normalize_components
        self.lr = lr
        if train_only_adapters:
            self._freeze_non_adapters()

    def _freeze_non_adapters(self) -> None:
        n_train, n_total = 0, 0
        for n, p in self.policy.named_parameters():
            n_total += p.numel()
            if "adapters" in n or "property_embedding" in n:
                p.requires_grad_(True)
                n_train += p.numel()
            else:
                p.requires_grad_(False)
        if hasattr(self, "_print_freeze") and not self._print_freeze:
            return
        print(f"[DPO] adapter-only freeze: {n_train:,} / {n_total:,} params trainable "
              f"({100*n_train/n_total:.2f}%)")
        self._print_freeze = False

    def per_component_mse(
        self,
        struct: Any,                 # PyG batch
        t: int,
        rand_l: torch.Tensor,        # [B, 3, 3]
        rand_x: torch.Tensor,        # [N, 3]
        rand_t: torch.Tensor,        # [N, MAX_ATOMIC_NUM]
        m: torch.Tensor,             # [B] property_indicator (0 or 1)
        y: torch.Tensor,             # [B] scaled property
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the policy decoder under the cached (t, noise, m) and return per-graph MSEs.

        Returns (mse_lattice, mse_coord, mse_type), each shape [B].
        """
        device = rand_l.device
        B = struct.num_graphs

        bs = self.policy.beta_scheduler
        ss = self.policy.sigma_scheduler
        alphas_cumprod_t = bs.alphas_cumprod[t]
        c0 = torch.sqrt(alphas_cumprod_t)
        c1 = torch.sqrt(1.0 - alphas_cumprod_t)
        sigma_t = ss.sigmas[t]
        sigma_norm_t = ss.sigmas_norm[t]

        t_vec = torch.full((B,), t, device=device, dtype=torch.long)
        time_emb = self.policy.time_embedding(t_vec)

        lattices_gt = lattice_params_to_matrix_torch(struct.lengths, struct.angles)
        input_lattice = c0 * lattices_gt + c1 * rand_l
        input_frac = (struct.frac_coords + sigma_t * rand_x) % 1.0
        gt_atom_onehot = F.one_hot(struct.atom_types - 1, num_classes=MAX_ATOMIC_NUM).float()
        atom_type_probs = c0 * gt_atom_onehot + c1 * rand_t

        pred_l, pred_x, pred_a = self.policy.decoder(
            time_emb, atom_type_probs, input_frac, input_lattice,
            struct.num_atoms, struct.batch, y, m,
        )

        # Per-graph MSE for lattice (mean over the 3x3 entries).
        mse_l = ((pred_l - rand_l) ** 2).mean(dim=(1, 2))   # [B]

        # Coord (wrapped-normal score-matching target).
        # Note: sigma_t is a scalar; expand for the d_log_p call.
        sigmas_per_atom = sigma_t.expand(rand_x.size(0))[:, None]
        sigmas_norm_per_atom = sigma_norm_t.expand(rand_x.size(0))[:, None]
        target_x = d_log_p_wrapped_normal(sigmas_per_atom * rand_x, sigmas_per_atom) / torch.sqrt(sigmas_norm_per_atom)
        per_atom_mse_f = ((pred_x - target_x) ** 2).mean(dim=1)   # [N]
        mse_f = scatter(per_atom_mse_f, struct.batch, dim=0, dim_size=B, reduce="mean")   # [B]

        # Type ε-MSE (per-graph mean of per-atom MSE).
        per_atom_mse_a = ((pred_a - rand_t) ** 2).mean(dim=1)
        mse_a = scatter(per_atom_mse_a, struct.batch, dim=0, dim_size=B, reduce="mean")   # [B]

        return mse_l, mse_f, mse_a

    def dpo_step(self, batch: Any) -> dict[str, torch.Tensor]:
        """Compute the 3-term DPO loss for a batch of pairs.

        Expected batch fields: winner (PyG batch), loser (PyG batch),
            t (int), m (Tensor [B]), y (Tensor [B]),
            rand_l_w/rand_x_w/rand_t_w (winner's noise),
            rand_l_l/rand_x_l/rand_t_l (loser's noise),
            ref_loss_w/ref_loss_l (Tensor [B, 3] per-component π_ref MSEs).
        """
        mse_l_w, mse_f_w, mse_a_w = self.per_component_mse(
            batch.winner, batch.t, batch.rand_l_w, batch.rand_x_w, batch.rand_t_w, batch.m, batch.y
        )
        mse_l_l, mse_f_l, mse_a_l = self.per_component_mse(
            batch.loser,  batch.t, batch.rand_l_l, batch.rand_x_l, batch.rand_t_l, batch.m, batch.y
        )

        if self.normalize_components:
            # Normalize each component by its π_ref baseline magnitude (averaged across pairs).
            ref_l_mag = 0.5 * (batch.ref_loss_w[:, 0].mean() + batch.ref_loss_l[:, 0].mean())
            ref_f_mag = 0.5 * (batch.ref_loss_w[:, 1].mean() + batch.ref_loss_l[:, 1].mean())
            ref_a_mag = 0.5 * (batch.ref_loss_w[:, 2].mean() + batch.ref_loss_l[:, 2].mean())
            w_L = self.cost_lattice / (ref_l_mag.detach() + 1e-8)
            w_F = self.cost_coord   / (ref_f_mag.detach() + 1e-8)
            w_A = self.cost_type    / (ref_a_mag.detach() + 1e-8)
        else:
            w_L = self.cost_lattice
            w_F = self.cost_coord
            w_A = self.cost_type

        delta_theta_w = w_L * mse_l_w + w_F * mse_f_w + w_A * mse_a_w   # [B]
        delta_theta_l = w_L * mse_l_l + w_F * mse_f_l + w_A * mse_a_l
        delta_ref_w = (w_L * batch.ref_loss_w[:, 0] + w_F * batch.ref_loss_w[:, 1] + w_A * batch.ref_loss_w[:, 2])
        delta_ref_l = (w_L * batch.ref_loss_l[:, 0] + w_F * batch.ref_loss_l[:, 1] + w_A * batch.ref_loss_l[:, 2])

        z = -self.beta * self.T * ((delta_theta_w - delta_ref_w) - (delta_theta_l - delta_ref_l))
        loss_per_pair = -F.logsigmoid(z)
        loss = loss_per_pair.mean()

        comp_l_logit = -self.beta * self.T * w_L * ((mse_l_w - batch.ref_loss_w[:, 0]) - (mse_l_l - batch.ref_loss_l[:, 0]))
        comp_f_logit = -self.beta * self.T * w_F * ((mse_f_w - batch.ref_loss_w[:, 1]) - (mse_f_l - batch.ref_loss_l[:, 1]))
        comp_a_logit = -self.beta * self.T * w_A * ((mse_a_w - batch.ref_loss_w[:, 2]) - (mse_a_l - batch.ref_loss_l[:, 2]))

        return {
            "loss": loss,
            "z": z,
            "sigma_z": torch.sigmoid(z),
            "comp_l_logit": comp_l_logit,
            "comp_f_logit": comp_f_logit,
            "comp_a_logit": comp_a_logit,
            "delta_theta_w": delta_theta_w,
            "delta_theta_l": delta_theta_l,
        }

    def training_step(self, batch, batch_idx):
        out = self.dpo_step(batch)
        self._log_components(out, prefix="train")
        return out["loss"]

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            out = self.dpo_step(batch)
        self._log_components(out, prefix="val")
        return out["loss"]

    def _log_components(self, out: dict[str, torch.Tensor], prefix: str) -> None:
        cl = out["comp_l_logit"].abs().mean()
        cf = out["comp_f_logit"].abs().mean()
        ca = out["comp_a_logit"].abs().mean()
        total = cl + cf + ca + 1e-8
        self.log_dict({
            f"{prefix}/loss": out["loss"],
            f"{prefix}/sigma_z_mean": out["sigma_z"].mean(),
            f"{prefix}/share_lattice": cl / total,
            f"{prefix}/share_coord":   cf / total,
            f"{prefix}/share_type":    ca / total,
            f"{prefix}/delta_theta_w_mean": out["delta_theta_w"].mean(),
            f"{prefix}/delta_theta_l_mean": out["delta_theta_l"].mean(),
        }, prog_bar=False, on_step=True, on_epoch=False)

    def configure_optimizers(self):
        params = [p for p in self.policy.parameters() if p.requires_grad]
        return torch.optim.Adam(params, lr=self.lr)
