from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import copy

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch, Data


ATOM_COUNT_PRIORS = {
    "supccomb_12": [
        0.0,
        0.003132613992342499,
        0.01983988861816916,
        0.07518273581621998,
        0.5200139227288548,
        0.10111381830838845,
        0.1395753567699269,
        0.014792899408284023,
        0.09484859032370345,
        0.001740341106856944,
        0.004872955099199443,
        0.0,
        0.024886877828054297,
    ],
    "supercond_7": [
        0.0,
        0.0021713383339913147,
        0.01914725621792341,
        0.08626135017765496,
        0.5617844453217529,
        0.09198578760363206,
        0.13087248322147652,
        0.014607185155941572,
        0.0890248716936439,
        0.0005921831819976313,
        0.0035530990919857876,
    ],
}


@dataclass
class PromptSample:
    batch: Batch
    target_raw: float
    target_scaled: float


class TcCurriculumSampler:
    def __init__(
        self,
        train_csv: str | Path,
        prop_scaler,
        dataset_name: str = "supccomb_12",
        prop_key: str = "tcad",
        upper_fraction: float = 0.7,
        seed: int = 0,
    ) -> None:
        self.train_csv = Path(train_csv)
        self.dataset_name = dataset_name
        self.prop_key = prop_key
        self.prop_scaler = prop_scaler
        self.upper_fraction = upper_fraction
        self.rng = np.random.default_rng(seed)

        df = pd.read_csv(self.train_csv)
        if prop_key not in df.columns:
            raise KeyError(f"Expected property column '{prop_key}' in {self.train_csv}.")

        self.targets = df[prop_key].astype(float).to_numpy()
        self.upper_threshold = float(np.quantile(self.targets, 0.75))
        self.upper_targets = self.targets[self.targets >= self.upper_threshold]
        if len(self.upper_targets) == 0:
            self.upper_targets = self.targets

        if dataset_name not in ATOM_COUNT_PRIORS:
            raise KeyError(
                f"No atom-count prior registered for dataset '{dataset_name}'."
            )
        self.atom_count_prior = np.asarray(ATOM_COUNT_PRIORS[dataset_name], dtype=float)
        self.atom_count_prior = self.atom_count_prior / self.atom_count_prior.sum()

    def _sample_target_raw(self) -> float:
        if self.rng.random() < self.upper_fraction:
            return float(self.rng.choice(self.upper_targets))
        return float(self.rng.choice(self.targets))

    def _sample_num_atoms(self) -> int:
        return int(
            self.rng.choice(np.arange(len(self.atom_count_prior)), p=self.atom_count_prior)
        )

    def _make_prompt_batch(self, num_atoms: int) -> Batch:
        data = Data(
            num_atoms=torch.LongTensor([num_atoms]),
            num_nodes=int(num_atoms),
        )
        return Batch.from_data_list([data])

    def _scale_target(self, raw_target: float) -> float:
        scaled = self.prop_scaler.transform([[raw_target]])
        return float(np.asarray(scaled).reshape(-1)[0])

    def sample_prompt(self) -> PromptSample:
        raw_target = self._sample_target_raw()
        scaled_target = self._scale_target(raw_target)
        batch = self._make_prompt_batch(self._sample_num_atoms())
        return PromptSample(batch=batch, target_raw=raw_target, target_scaled=scaled_target)

    def build_validation_set(self, size: int, seed: int) -> list[PromptSample]:
        backup_rng = copy.deepcopy(self.rng)
        self.rng = np.random.default_rng(seed)
        samples = [self.sample_prompt() for _ in range(size)]
        self.rng = backup_rng
        return samples
