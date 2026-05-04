from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import hydra
import omegaconf
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import torch

from diffcsp.common.utils import PROJECT_ROOT
from diffcsp.rl.prompts import TcCurriculumSampler
from diffcsp.rl.rewards import RewardManager
from diffcsp.rl.trainer import GRPOTrainer


def _to_namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_namespace(v) for v in value]
    return value


@hydra.main(version_base=None, config_path=str(PROJECT_ROOT / "conf"), config_name="rl")
def main(cfg: omegaconf.DictConfig):
    from scripts.eval_utils import load_model

    hydra_dir = Path(HydraConfig.get().run.dir)
    hydra_dir.mkdir(parents=True, exist_ok=True)
    (hydra_dir / "rl_config.yaml").write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_path = Path(cfg.paths.policy_checkpoint).resolve()
    reference_path = Path(cfg.paths.reference_checkpoint or cfg.paths.policy_checkpoint).resolve()

    policy, _, _ = load_model(policy_path, load_data=False)
    reference_policy, _, _ = load_model(reference_path, load_data=False)
    policy.to(device)
    reference_policy.to(device)
    reference_policy.eval()
    for param in reference_policy.parameters():
        param.requires_grad_(False)

    optimizer = hydra.utils.instantiate(
        cfg.optim.optimizer, params=policy.parameters(), _convert_="partial"
    )

    prop_scaler = policy.scaler
    prompt_sampler = TcCurriculumSampler(
        train_csv=Path(cfg.data.root_path) / "train.csv",
        prop_scaler=prop_scaler,
        dataset_name=cfg.data.name_for_prior,
        prop_key=cfg.data.prop,
        upper_fraction=cfg.grpo.curriculum.upper_fraction,
        seed=cfg.grpo.seed,
    )
    validation_prompts = prompt_sampler.build_validation_set(
        size=cfg.grpo.eval.num_prompts,
        seed=cfg.grpo.eval.seed,
    )

    reward_manager = RewardManager(
        repo_root=PROJECT_ROOT,
        reward_cfg=cfg.grpo.rewards,
        prop_scaler=prop_scaler,
    )

    trainer = GRPOTrainer(
        policy=policy,
        reference_policy=reference_policy,
        optimizer=optimizer,
        reward_manager=reward_manager,
        prompt_sampler=prompt_sampler,
        validation_prompts=validation_prompts,
        cfg=_to_namespace(OmegaConf.to_container(cfg.grpo, resolve=True)),
        output_dir=hydra_dir,
        device=device,
    )
    trainer.train()


if __name__ == "__main__":
    main()
