from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import sys
import time
from typing import Iterable

import numpy as np
import torch
from torch import nn


DEFAULT_INVALID_REWARD = -10.0
PROXY_FEATURE_NAMES = [
    "tcad",
    "lambda",
    "wlog",
    "w2",
    "tc",
    "band_gap",
    "formation_energy",
    "num_atoms",
    "num_species",
    "volume_per_atom",
    "density",
    "min_distance",
]


def _require(module_name: str, extra_path: Path | None = None):
    if extra_path is not None and str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))
    try:
        return __import__(module_name, fromlist=["_dummy"])
    except Exception as exc:
        raise RuntimeError(
            f"Required dependency '{module_name}' is not available for RL rewards."
        ) from exc


def _softplus_scalar(x: float) -> float:
    return float(np.log1p(np.exp(x)))


def _safe_float(value: float, default: float = 0.0) -> float:
    try:
        value = float(value)
    except Exception:
        return default
    return value if np.isfinite(value) else default


def _nanmean_or_default(values, default: float = float("nan")) -> float:
    values = np.asarray(values, dtype=float)
    if not np.any(np.isfinite(values)):
        return default
    return float(np.nanmean(values))


def _flatten_state(clean_state: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_atoms = int(clean_state["num_atoms"].view(-1)[0].item())
    frac_coords = clean_state["frac_coords"][:num_atoms].detach().cpu().numpy()
    atom_types = clean_state["atom_types"][:num_atoms].detach().cpu().numpy()
    lattices = clean_state["lattices"][0].detach().cpu().numpy()
    return frac_coords, atom_types, lattices


def build_pymatgen_structure(clean_state: dict):
    pymatgen_core = _require("pymatgen.core")
    frac_coords, atom_types, lattice = _flatten_state(clean_state)
    Structure = pymatgen_core.Structure
    return Structure(lattice=lattice, species=atom_types.tolist(), coords=frac_coords, coords_are_cartesian=False)


def state_validity(clean_state: dict, min_distance: float = 0.5, min_volume: float = 0.1) -> tuple[bool, str]:
    try:
        structure = build_pymatgen_structure(clean_state)
    except Exception as exc:
        return False, f"structure_build_failed:{type(exc).__name__}"

    if not np.isfinite(structure.volume) or structure.volume <= min_volume:
        return False, "invalid_volume"

    dist_mat = structure.distance_matrix
    dist_mat = dist_mat + np.diag(np.ones(dist_mat.shape[0]) * (min_distance + 10.0))
    if float(dist_mat.min()) < min_distance:
        return False, "atom_overlap"
    return True, "ok"


def summarize_validity(clean_states: Iterable[dict]) -> dict:
    valid_flags = []
    invalid_reasons = {}
    for clean_state in clean_states:
        valid, reason = state_validity(clean_state)
        valid_flags.append(valid)
        if not valid:
            invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1

    valid_flags = np.asarray(valid_flags, dtype=bool)
    invalid_rate = 1.0 - float(valid_flags.mean()) if len(valid_flags) > 0 else float("nan")
    return {
        "valid_flags": valid_flags,
        "invalid_rate": invalid_rate,
        "invalid_reasons": invalid_reasons,
    }


def estimate_duplicate_rate(
    clean_states: Iterable[dict],
    stol: float = 0.5,
    angle_tol: float = 10.0,
    ltol: float = 0.3,
) -> float:
    matcher_mod = _require("pymatgen.analysis.structure_matcher")
    matcher = matcher_mod.StructureMatcher(stol=stol, angle_tol=angle_tol, ltol=ltol)

    duplicate_count = 0
    valid_count = 0
    buckets = {}

    for clean_state in clean_states:
        valid, _ = state_validity(clean_state)
        if not valid:
            continue
        structure = build_pymatgen_structure(clean_state)
        valid_count += 1
        key = (structure.composition.reduced_formula, len(structure))
        seen = buckets.setdefault(key, [])
        if any(matcher.fit(structure, existing) for existing in seen):
            duplicate_count += 1
        else:
            seen.append(structure)

    return duplicate_count / max(valid_count, 1)


def ase_atoms_from_state(clean_state: dict):
    adaptors = _require("pymatgen.io.ase")
    structure = build_pymatgen_structure(clean_state)
    return adaptors.AseAtomsAdaptor.get_atoms(structure)


def build_proxy_feature_vector(clean_state: dict, bee_output: dict, meg_output: dict) -> np.ndarray | None:
    valid, _ = state_validity(clean_state)
    if not valid or not bee_output.get("valid", False) or not meg_output.get("valid", False):
        return None

    structure = build_pymatgen_structure(clean_state)
    dist_mat = structure.distance_matrix
    dist_mat = dist_mat + np.diag(np.ones(dist_mat.shape[0]) * 1000.0)
    min_distance = float(dist_mat.min())
    num_atoms = float(len(structure))
    num_species = float(len(structure.composition))
    volume_per_atom = float(structure.volume / max(len(structure), 1))
    density = float(structure.density)

    return np.asarray(
        [
            _safe_float(bee_output.get("tcad")),
            _safe_float(bee_output.get("lambda")),
            _safe_float(bee_output.get("wlog")),
            _safe_float(bee_output.get("w2")),
            _safe_float(bee_output.get("tc")),
            _safe_float(meg_output.get("band_gap")),
            _safe_float(meg_output.get("formation_energy")),
            num_atoms,
            num_species,
            volume_per_atom,
            density,
            min_distance,
        ],
        dtype=np.float32,
    )


class ProxyMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, depth: int = 2) -> None:
        super().__init__()
        layers = []
        current_dim = input_dim
        for _ in range(max(depth, 1)):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.SiLU())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class TorchProxyRewardModel:
    def __init__(self, checkpoint_path: str | Path, reward_weight: float | None = None) -> None:
        checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
        metadata = checkpoint["metadata"]
        self.name = metadata["name"]
        self.task_type = metadata["task_type"]
        self.direction = metadata["direction"]
        self.feature_names = metadata["feature_names"]
        self.feature_mean = torch.tensor(metadata["feature_mean"], dtype=torch.float32)
        self.feature_std = torch.tensor(metadata["feature_std"], dtype=torch.float32)
        self.label_mean = float(metadata.get("label_mean", 0.0))
        self.label_std = float(metadata.get("label_std", 1.0))
        self.reward_weight = float(
            reward_weight if reward_weight is not None else metadata.get("reward_weight", 1.0)
        )

        self.model = ProxyMLP(
            input_dim=len(self.feature_names),
            hidden_dim=int(metadata["hidden_dim"]),
            depth=int(metadata["depth"]),
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    def _reward_from_prediction(self, prediction: float) -> float:
        if self.task_type == "binary":
            probability = 1.0 / (1.0 + math.exp(-prediction))
            base_reward = 2.0 * probability - 1.0
            if self.direction == "minimize":
                base_reward = -base_reward
            return self.reward_weight * base_reward

        normalized = (prediction - self.label_mean) / max(self.label_std, 1e-6)
        if self.direction == "minimize":
            normalized = -normalized
        return self.reward_weight * normalized

    def score_states(
        self,
        clean_states: Iterable[dict],
        bee_outputs: list[dict],
        meg_outputs: list[dict],
    ) -> list[dict]:
        outputs = []
        for clean_state, bee_output, meg_output in zip(clean_states, bee_outputs, meg_outputs):
            features = build_proxy_feature_vector(clean_state, bee_output, meg_output)
            if features is None:
                outputs.append(
                    {
                        "valid": False,
                        "prediction": float("nan"),
                        "reward": 0.0,
                    }
                )
                continue

            feature_tensor = torch.tensor(features, dtype=torch.float32)
            feature_tensor = (feature_tensor - self.feature_mean) / self.feature_std.clamp(min=1e-6)
            with torch.no_grad():
                prediction = float(self.model(feature_tensor.unsqueeze(0)).item())
            outputs.append(
                {
                    "valid": True,
                    "prediction": prediction,
                    "reward": self._reward_from_prediction(prediction),
                }
            )
        return outputs


class BEECSORewardModel:
    def __init__(
        self,
        repo_root: str | Path,
        checkpoints_dir: str | Path | None = None,
        ensemble_size: int = 4,
        start_index: int = 0,
        device: str = "cpu",
        r_max: float = 4.0,
    ) -> None:
        repo_root = Path(repo_root)
        workflow_root = repo_root / "external" / "BEE-NET" / "workflow"
        tg = _require("torch_geometric", workflow_root)
        torch_mod = _require("torch", workflow_root)
        ase_atom = _require("ase", workflow_root)
        neighbor = _require("ase.neighborlist", workflow_root)
        model_mod = _require("utils.model", workflow_root)

        self.torch = torch_mod
        self.tg = tg
        self.Atom = ase_atom.Atom
        self.neighbor_list = neighbor.neighbor_list
        self.PeriodicNetwork = model_mod.PeriodicNetwork
        self.device = torch.device(device)
        self.default_dtype = torch.float64
        self.r_max = r_max

        checkpoints_dir = Path(checkpoints_dir or repo_root / "checkpoints" / "bee_net" / "CSO")
        checkpoint_paths = sorted(checkpoints_dir.glob("model_CSO_derived_EMD_0_*.pt1.pt"))
        if not checkpoint_paths:
            raise FileNotFoundError(f"No BEE-Net CSO checkpoints found in {checkpoints_dir}.")

        selected = checkpoint_paths[start_index : start_index + ensemble_size]
        if not selected:
            raise ValueError("No BEE-Net checkpoints selected. Adjust ensemble_size/start_index.")

        self.type_encoding = {}
        specie_am = []
        for z in range(1, 119):
            specie = self.Atom(z)
            self.type_encoding[specie.symbol] = z - 1
            specie_am.append(specie.mass)
        self.type_onehot = torch.eye(len(self.type_encoding), dtype=self.default_dtype)
        self.am_onehot = torch.diag(torch.tensor(specie_am, dtype=self.default_dtype))
        self.freq_final = np.arange(0.25, 101, 2)
        self.mu_star = 0.1

        init_dict = dict(
            in_dim=118,
            em_dim=64,
            irreps_in="64x0e",
            irreps_out=f"{len(self.freq_final)}x0e",
            irreps_node_attr="64x0e",
            layers=2,
            mul=32,
            lmax=3,
            max_radius=self.r_max,
            num_neighbors=17.133204568916714,
            reduce_output=True,
            p=0.0,
        )
        self.model = self.PeriodicNetwork(**init_dict)
        self.model.pool = True
        self.model.to(self.device)
        self.model.eval()
        self.state_dicts = [torch.load(path, map_location=self.device) for path in selected]

    def _build_data(self, atoms):
        symbols = list(atoms.symbols).copy()
        positions = torch.from_numpy(atoms.positions.copy()).to(self.default_dtype)
        lattice = torch.from_numpy(atoms.cell.array.copy()).to(self.default_dtype).unsqueeze(0)

        edge_src, edge_dst, edge_shift = self.neighbor_list(
            "ijS", a=atoms, cutoff=self.r_max, self_interaction=True
        )
        edge_batch = positions.new_zeros(positions.shape[0], dtype=torch.long)[
            torch.from_numpy(edge_src)
        ]
        edge_vec = (
            positions[torch.from_numpy(edge_dst)]
            - positions[torch.from_numpy(edge_src)]
            + torch.einsum(
                "ni,nij->nj",
                torch.tensor(edge_shift, dtype=self.default_dtype),
                lattice[edge_batch],
            )
        )

        x = self.am_onehot[[self.type_encoding[symbol] for symbol in symbols]]
        z = self.type_onehot[[self.type_encoding[symbol] for symbol in symbols]]

        return self.tg.data.Data(
            pos=positions,
            lattice=lattice,
            symbol=symbols,
            x=x,
            z=z,
            edge_index=torch.stack(
                [torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0
            ),
            edge_shift=torch.tensor(edge_shift, dtype=self.default_dtype),
            edge_vec=edge_vec,
            edge_len=edge_vec.norm(dim=1).cpu().numpy(),
            target=torch.zeros((1, len(self.freq_final)), dtype=self.default_dtype),
        )

    def _cal_lambda(self, alpha_f):
        value = 0.0
        for i in range(1, len(self.freq_final)):
            if self.freq_final[i] > 0:
                dw = self.freq_final[i] - self.freq_final[i - 1]
                value += (alpha_f[i] / self.freq_final[i]) * dw
        return 2.0 * value

    def _cal_w_log(self, alpha_f, lamb):
        if lamb <= 0:
            return float("nan")
        value = 0.0
        for i in range(1, len(self.freq_final)):
            if self.freq_final[i] > 0:
                dw = self.freq_final[i] - self.freq_final[i - 1]
                value += alpha_f[i] * np.log(self.freq_final[i]) * dw / self.freq_final[i]
        return np.exp(2.0 * value / lamb) / 0.08617

    def _cal_w_sq(self, alpha_f, lamb):
        if lamb <= 0:
            return float("nan")
        value = 0.0
        for i in range(1, len(self.freq_final)):
            if self.freq_final[i] > 0:
                dw = self.freq_final[i] - self.freq_final[i - 1]
                value += alpha_f[i] * self.freq_final[i] * dw
        return ((2.0 * value / lamb) ** 0.5) / 0.08617

    def _cal_tc(self, lamb, omega_log):
        denom = lamb - self.mu_star * (1.0 + 0.62 * lamb)
        if denom <= 0 or omega_log <= 0:
            return float("nan")
        frac = -1.04 * (1.0 + lamb) / denom
        return (omega_log / 1.2) * np.exp(frac)

    def _cal_tc_ad(self, lamb, wlog, w2, tc):
        if not np.isfinite(tc) or not np.isfinite(wlog) or not np.isfinite(w2) or wlog <= 0:
            return float("nan")
        f1 = (1 + (lamb / (2.46 * (1 + 3.8 * self.mu_star))) ** 1.5) ** (1 / 3)
        f2 = 1 + ((lamb**2) * ((w2 / wlog) - 1)) / (
            lamb**2 + (1.82 * (1 + 6.3 * self.mu_star) * (w2 / wlog)) ** 2
        )
        return f1 * f2 * tc

    def score_states(self, clean_states: Iterable[dict]) -> list[dict]:
        outputs = []
        for clean_state in clean_states:
            valid, reason = state_validity(clean_state)
            if not valid:
                outputs.append({"valid": False, "reason": reason, "tcad": float("nan")})
                continue

            atoms = ase_atoms_from_state(clean_state)
            data = self._build_data(atoms)
            loader = self.tg.loader.DataLoader([data], batch_size=1)
            predictions = []
            with torch.no_grad():
                for state_dict in self.state_dicts:
                    self.model.load_state_dict(state_dict)
                    for batch in loader:
                        batch = batch.to(self.device)
                        output = self.model(batch).detach().cpu().numpy()[0]
                        predictions.append(output)
            pred_avg = np.mean(np.asarray(predictions), axis=0)
            lamb = self._cal_lambda(pred_avg)
            wlog = self._cal_w_log(pred_avg, lamb)
            w2 = self._cal_w_sq(pred_avg, lamb)
            tc = self._cal_tc(lamb, wlog)
            tcad = self._cal_tc_ad(lamb, wlog, w2, tc)
            outputs.append(
                {
                    "valid": True,
                    "tcad": float(tcad),
                    "lambda": float(lamb),
                    "wlog": float(wlog),
                    "w2": float(w2),
                    "tc": float(tc),
                }
            )
        return outputs


class MEGNetRewardModel:
    def __init__(
        self,
        repo_root: str | Path,
        formation_model: str = "Eform_MP_2019",
        bandgap_model: str = "Bandgap_MP_2018",
        metal_threshold: float = 0.05,
    ) -> None:
        megnet_root = Path(repo_root) / "external" / "megnet"
        utils_models = _require("megnet.utils.models", megnet_root)
        self.load_model = utils_models.load_model
        self.formation_model = self.load_model(formation_model)
        self.bandgap_model = self.load_model(bandgap_model)
        self.metal_threshold = metal_threshold

    def score_states(self, clean_states: Iterable[dict]) -> list[dict]:
        outputs = []
        for clean_state in clean_states:
            valid, reason = state_validity(clean_state)
            if not valid:
                outputs.append(
                    {
                        "valid": False,
                        "reason": reason,
                        "formation_energy": float("nan"),
                        "band_gap": float("nan"),
                        "metallic": False,
                    }
                )
                continue

            structure = build_pymatgen_structure(clean_state)
            formation_energy = float(self.formation_model.predict_structure(structure).ravel()[0])
            band_gap = float(self.bandgap_model.predict_structure(structure).ravel()[0])
            outputs.append(
                {
                    "valid": True,
                    "formation_energy": formation_energy,
                    "band_gap": band_gap,
                    "metallic": band_gap <= self.metal_threshold,
                }
            )
        return outputs


class M3GNetRewardModel:
    def __init__(
        self,
        force_weight: float = 0.4,
        stress_weight: float = 0.2,
        deltae_weight: float = 0.2,
        rmsd_weight: float = 0.2,
        relax_steps: int = 5,
        relax_fmax: float = 0.05,
    ) -> None:
        matgl = _require("matgl")
        ext_ase = _require("matgl.ext.ase")
        self.potential = matgl.load_model("M3GNet-MP-2021.2.8-PES")
        self.PESCalculator = ext_ase.PESCalculator
        self.Relaxer = ext_ase.Relaxer
        self.relaxer = self.Relaxer(potential=self.potential)
        self.force_weight = force_weight
        self.stress_weight = stress_weight
        self.deltae_weight = deltae_weight
        self.rmsd_weight = rmsd_weight
        self.relax_steps = relax_steps
        self.relax_fmax = relax_fmax

    def score_states(self, clean_states: Iterable[dict]) -> list[dict]:
        outputs = []
        for clean_state in clean_states:
            valid, reason = state_validity(clean_state)
            if not valid:
                outputs.append({"valid": False, "reason": reason, "reward": DEFAULT_INVALID_REWARD})
                continue

            structure = build_pymatgen_structure(clean_state)
            atoms = ase_atoms_from_state(clean_state)
            atoms.calc = self.PESCalculator(self.potential)

            try:
                initial_energy = float(atoms.get_potential_energy())
                forces = atoms.get_forces()
                stress = atoms.get_stress(voigt=False)
                relax_result = self.relaxer.relax(
                    structure,
                    fmax=self.relax_fmax,
                    steps=self.relax_steps,
                )
                final_structure = relax_result["final_structure"]
                final_energy = float(relax_result["trajectory"].energies[-1])
                final_atoms = ase_atoms_from_state(
                    {
                        "num_atoms": clean_state["num_atoms"],
                        "atom_types": torch.tensor(final_structure.atomic_numbers),
                        "frac_coords": torch.tensor(final_structure.frac_coords, dtype=torch.float32),
                        "lattices": torch.tensor(final_structure.lattice.matrix, dtype=torch.float32).unsqueeze(0),
                    }
                )
                rmsd = float(np.sqrt(np.mean((atoms.positions - final_atoms.positions) ** 2)))
            except Exception as exc:
                outputs.append(
                    {
                        "valid": False,
                        "reason": f"m3g_failed:{type(exc).__name__}",
                        "reward": DEFAULT_INVALID_REWARD,
                    }
                )
                continue

            force_rms = float(np.sqrt(np.mean(np.square(forces))))
            stress_norm = float(np.linalg.norm(stress))
            delta_e = float(initial_energy - final_energy)
            reward = (
                -self.force_weight * force_rms
                -self.stress_weight * stress_norm
                -self.deltae_weight * delta_e
                -self.rmsd_weight * rmsd
            )
            outputs.append(
                {
                    "valid": True,
                    "reward": reward,
                    "force_rms": force_rms,
                    "stress_norm": stress_norm,
                    "delta_e_short": delta_e,
                    "rmsd_short": rmsd,
                }
            )
        return outputs


@dataclass
class RewardBreakdown:
    bee_returns: torch.Tensor
    meg_rewards: torch.Tensor
    m3g_rewards: torch.Tensor
    proxy_rewards: torch.Tensor
    ranking_scores: torch.Tensor
    bee_potentials: torch.Tensor
    bee_tcad: torch.Tensor
    meg_band_gap: torch.Tensor
    meg_formation_energy: torch.Tensor
    m3g_mask: torch.Tensor


class RewardManager:
    def __init__(self, repo_root: str | Path, reward_cfg, prop_scaler) -> None:
        self.repo_root = Path(repo_root)
        self.cfg = reward_cfg
        self.prop_scaler = prop_scaler
        self.sigma_tcad = float(np.asarray(prop_scaler.stds).reshape(-1)[0])

        self.bee = BEECSORewardModel(
            repo_root=self.repo_root,
            checkpoints_dir=reward_cfg.bee.checkpoints_dir,
            ensemble_size=reward_cfg.bee.ensemble_size,
            start_index=reward_cfg.bee.start_index,
            device=reward_cfg.bee.device,
        )
        self.meg = MEGNetRewardModel(
            repo_root=self.repo_root,
            formation_model=reward_cfg.megnet.formation_model,
            bandgap_model=reward_cfg.megnet.bandgap_model,
            metal_threshold=reward_cfg.megnet.metal_threshold,
        )
        self.m3g = None
        if reward_cfg.m3gnet.enabled:
            self.m3g = M3GNetRewardModel(
                force_weight=reward_cfg.m3gnet.force_weight,
                stress_weight=reward_cfg.m3gnet.stress_weight,
                deltae_weight=reward_cfg.m3gnet.deltae_weight,
                rmsd_weight=reward_cfg.m3gnet.rmsd_weight,
                relax_steps=reward_cfg.m3gnet.relax_steps,
                relax_fmax=reward_cfg.m3gnet.relax_fmax,
            )
        self.proxy_models = []
        proxy_cfg = reward_cfg.get("proxies") if hasattr(reward_cfg, "get") else getattr(reward_cfg, "proxies", None)
        if proxy_cfg is not None and proxy_cfg.get("enabled", False):
            for entry in proxy_cfg.entries:
                self.proxy_models.append(
                    TorchProxyRewardModel(
                        checkpoint_path=entry.checkpoint,
                        reward_weight=entry.get("weight"),
                    )
                )

    def _bee_potential(self, bee_output: dict, target_raw: float) -> float:
        if not bee_output.get("valid", False):
            return DEFAULT_INVALID_REWARD
        tcad = float(bee_output["tcad"])
        return (
            -abs(tcad - target_raw) / self.sigma_tcad
            + 0.25 * max(tcad - target_raw, 0.0) / self.sigma_tcad
        )

    def _meg_reward(self, meg_output: dict) -> float:
        if not meg_output.get("valid", False):
            return DEFAULT_INVALID_REWARD
        band_gap = float(meg_output["band_gap"])
        formation_energy = float(meg_output["formation_energy"])
        return 0.5 * float(band_gap <= self.cfg.megnet.metal_threshold) - 0.5 * _softplus_scalar(
            formation_energy / 0.05
        )

    def _score_proxy_models(
        self,
        clean_states: list[dict],
        bee_outputs: list[dict],
        meg_outputs: list[dict],
    ) -> tuple[np.ndarray, dict[str, list[dict]]]:
        total_rewards = np.zeros(len(clean_states), dtype=np.float32)
        proxy_outputs = {}
        for proxy_model in self.proxy_models:
            outputs = proxy_model.score_states(clean_states, bee_outputs, meg_outputs)
            proxy_outputs[proxy_model.name] = outputs
            total_rewards += np.asarray(
                [float(item.get("reward", 0.0)) for item in outputs],
                dtype=np.float32,
            )
        return total_rewards, proxy_outputs

    def score_group(self, rollout_group: dict, target_raw: float, phase: str) -> RewardBreakdown:
        reward_times = rollout_group["reward_times"]
        branches = rollout_group["branches"]
        num_branches = len(branches)
        prefix_t = int(rollout_group["prefix_t"])
        final_states = [branch["final_state"] for branch in branches]

        bee_potentials = np.zeros((num_branches, len(reward_times)), dtype=np.float32)
        bee_tcad = np.full((num_branches, len(reward_times)), np.nan, dtype=np.float32)
        meg_rewards = np.zeros(num_branches, dtype=np.float32)
        meg_band_gap = np.full(num_branches, np.nan, dtype=np.float32)
        meg_formation_energy = np.full(num_branches, np.nan, dtype=np.float32)
        terminal_bee_outputs = [None] * num_branches
        terminal_meg_outputs = [None] * num_branches

        for branch_idx, branch in enumerate(branches):
            checkpoint_states = [branch["reconstructions"][t] for t in reward_times]
            bee_outputs = self.bee.score_states(checkpoint_states)
            for time_idx, output in enumerate(bee_outputs):
                bee_potentials[branch_idx, time_idx] = self._bee_potential(output, target_raw)
                bee_tcad[branch_idx, time_idx] = float(output.get("tcad", np.nan))
            terminal_bee_outputs[branch_idx] = bee_outputs[-1]

            meg_output = self.meg.score_states([branch["final_state"]])[0]
            meg_rewards[branch_idx] = self._meg_reward(meg_output)
            meg_band_gap[branch_idx] = float(meg_output.get("band_gap", np.nan))
            meg_formation_energy[branch_idx] = float(meg_output.get("formation_energy", np.nan))
            terminal_meg_outputs[branch_idx] = meg_output

        ranking_scores = bee_potentials[:, -1] + meg_rewards
        m3g_rewards = np.zeros(num_branches, dtype=np.float32)
        m3g_mask = np.zeros(num_branches, dtype=bool)
        proxy_rewards = np.zeros(num_branches, dtype=np.float32)

        if phase != "A" and self.m3g is not None:
            top_k = max(1, math.ceil(num_branches * float(self.cfg.m3gnet.top_fraction)))
            top_indices = np.argsort(ranking_scores)[::-1][:top_k]
            m3g_states = [branches[idx]["final_state"] for idx in top_indices]
            m3g_outputs = self.m3g.score_states(m3g_states)
            for branch_idx, output in zip(top_indices, m3g_outputs):
                m3g_rewards[branch_idx] = float(output.get("reward", DEFAULT_INVALID_REWARD))
                m3g_mask[branch_idx] = output.get("valid", False)

        if phase == "C" and self.proxy_models:
            proxy_rewards, _ = self._score_proxy_models(
                final_states,
                terminal_bee_outputs,
                terminal_meg_outputs,
            )

        checkpoint_returns = bee_potentials[:, 1:] - bee_potentials[:, :-1]
        future_returns = np.flip(np.cumsum(np.flip(checkpoint_returns, axis=1), axis=1), axis=1)
        bee_returns = np.zeros((num_branches, prefix_t), dtype=np.float32)

        for interval_idx, current_time in enumerate(reward_times[:-1]):
            next_time = reward_times[interval_idx + 1]
            interval_value = future_returns[:, interval_idx]
            for t in range(current_time, next_time, -1):
                bee_returns[:, prefix_t - t] = interval_value

        return RewardBreakdown(
            bee_returns=torch.tensor(bee_returns, dtype=torch.float32),
            meg_rewards=torch.tensor(meg_rewards, dtype=torch.float32),
            m3g_rewards=torch.tensor(m3g_rewards, dtype=torch.float32),
            proxy_rewards=torch.tensor(proxy_rewards, dtype=torch.float32),
            ranking_scores=torch.tensor(ranking_scores, dtype=torch.float32),
            bee_potentials=torch.tensor(bee_potentials, dtype=torch.float32),
            bee_tcad=torch.tensor(bee_tcad, dtype=torch.float32),
            meg_band_gap=torch.tensor(meg_band_gap, dtype=torch.float32),
            meg_formation_energy=torch.tensor(meg_formation_energy, dtype=torch.float32),
            m3g_mask=torch.tensor(m3g_mask),
        )

    def evaluate_states(
        self,
        clean_states: list[dict],
        target_raws: list[float],
        run_m3g: bool = False,
        m3g_subset_size: int | None = None,
        include_proxies: bool = False,
    ) -> dict:
        validity = summarize_validity(clean_states)
        bee_outputs = self.bee.score_states(clean_states)
        meg_outputs = self.meg.score_states(clean_states)

        m3g_outputs = None
        if run_m3g and self.m3g is not None:
            subset_size = min(m3g_subset_size or len(clean_states), len(clean_states))
            m3g_outputs = self.m3g.score_states(clean_states[:subset_size])

        proxy_rewards = None
        proxy_outputs = None
        if include_proxies and self.proxy_models:
            proxy_rewards, proxy_outputs = self._score_proxy_models(
                clean_states,
                bee_outputs,
                meg_outputs,
            )

        duplicate_rate = estimate_duplicate_rate(clean_states)
        bee_errors = []
        bee_uplifts = []
        metallic = []
        negative_eform = []
        cheap_pass = []

        for target_raw, bee_output, meg_output, is_valid in zip(
            target_raws,
            bee_outputs,
            meg_outputs,
            validity["valid_flags"],
        ):
            tcad = float(bee_output.get("tcad", np.nan))
            bee_errors.append(abs(tcad - target_raw) if np.isfinite(tcad) else np.nan)
            bee_uplifts.append((tcad - target_raw) if np.isfinite(tcad) else np.nan)

            metallic_flag = bool(meg_output.get("metallic", False))
            negative_eform_flag = bool(
                np.isfinite(meg_output.get("formation_energy", np.nan))
                and float(meg_output["formation_energy"]) < 0.0
            )
            metallic.append(float(metallic_flag))
            negative_eform.append(float(negative_eform_flag))
            cheap_pass.append(float(is_valid and metallic_flag and negative_eform_flag))

        return {
            "mean_abs_tcad_error": _nanmean_or_default(bee_errors, default=1.0e6),
            "mean_tcad_uplift": _nanmean_or_default(bee_uplifts, default=-1.0e6),
            "metallic_pass_rate": _nanmean_or_default(metallic, default=0.0),
            "negative_formation_pass_rate": _nanmean_or_default(negative_eform, default=0.0),
            "cheap_filter_pass_rate": _nanmean_or_default(cheap_pass, default=0.0),
            "invalid_rate": float(validity["invalid_rate"]),
            "invalid_reasons": validity["invalid_reasons"],
            "duplicate_rate": float(duplicate_rate),
            "m3g_reward_mean": (
                _nanmean_or_default([item.get("reward", np.nan) for item in m3g_outputs], default=None)
                if m3g_outputs is not None
                else None
            ),
            "proxy_reward_mean": (
                _nanmean_or_default(proxy_rewards, default=None)
                if proxy_rewards is not None
                else None
            ),
            "bee_outputs": bee_outputs,
            "meg_outputs": meg_outputs,
            "m3g_outputs": m3g_outputs,
            "proxy_outputs": proxy_outputs,
        }

    def benchmark(self, clean_states: list[dict]) -> dict:
        started = time.perf_counter()
        bee_outputs = self.bee.score_states(clean_states)
        bee_time = time.perf_counter() - started

        started = time.perf_counter()
        meg_outputs = self.meg.score_states(clean_states)
        meg_time = time.perf_counter() - started

        m3g_outputs = None
        m3g_time = None
        if self.m3g is not None:
            started = time.perf_counter()
            m3g_outputs = self.m3g.score_states(clean_states)
            m3g_time = time.perf_counter() - started

        proxy_outputs = None
        proxy_time = None
        if self.proxy_models:
            meg_outputs = self.meg.score_states(clean_states)
            started = time.perf_counter()
            _, proxy_outputs = self._score_proxy_models(clean_states, bee_outputs, meg_outputs)
            proxy_time = time.perf_counter() - started

        return {
            "bee_time_s": bee_time,
            "meg_time_s": meg_time,
            "m3g_time_s": m3g_time,
            "proxy_time_s": proxy_time,
            "bee_outputs": bee_outputs,
            "meg_outputs": meg_outputs,
            "m3g_outputs": m3g_outputs,
            "proxy_outputs": proxy_outputs,
        }
