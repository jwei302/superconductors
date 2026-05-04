from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import time

import numpy as np
import torch

from diffcsp.rl.prompts import PromptSample, TcCurriculumSampler
from diffcsp.rl.rewards import RewardBreakdown, RewardManager


@dataclass
class EvalSummary:
    guide_w: float
    mean_abs_tcad_error: float
    mean_tcad_uplift: float
    metallic_pass_rate: float
    negative_formation_pass_rate: float
    cheap_filter_pass_rate: float
    invalid_rate: float
    duplicate_rate: float
    m3g_reward_mean: float | None
    proxy_reward_mean: float | None
    composite_score: float


def determine_phase(update_idx: int, cfg) -> str:
    if update_idx < cfg.schedule.phase_a_updates:
        return "A"
    if update_idx < cfg.schedule.phase_b_updates:
        return "B"
    return "C"


def normalize_advantages(values: torch.Tensor, dim: int = 0, eps: float = 1e-6) -> torch.Tensor:
    mean = values.mean(dim=dim, keepdim=True)
    std = values.std(dim=dim, keepdim=True, unbiased=False)
    return (values - mean) / (std + eps)


def build_group_advantages(rewards: RewardBreakdown, cfg) -> torch.Tensor:
    bee_adv = normalize_advantages(rewards.bee_returns, dim=0, eps=cfg.loss.norm_eps)
    meg_adv = normalize_advantages(rewards.meg_rewards, dim=0, eps=cfg.loss.norm_eps).unsqueeze(1)

    if torch.count_nonzero(rewards.m3g_rewards).item() > 0:
        m3g_adv = normalize_advantages(rewards.m3g_rewards, dim=0, eps=cfg.loss.norm_eps).unsqueeze(1)
    else:
        m3g_adv = torch.zeros_like(meg_adv)

    if torch.count_nonzero(rewards.proxy_rewards).item() > 0:
        proxy_adv = normalize_advantages(rewards.proxy_rewards, dim=0, eps=cfg.loss.norm_eps).unsqueeze(1)
    else:
        proxy_adv = torch.zeros_like(meg_adv)

    advantages = (
        cfg.loss.bee_weight * bee_adv
        + cfg.loss.meg_weight * meg_adv
        + cfg.loss.m3g_weight * m3g_adv
        + cfg.loss.proxy_weight * proxy_adv
    )
    if cfg.loss.adv_clip_max is not None:
        advantages = advantages.clamp(min=-cfg.loss.adv_clip_max, max=cfg.loss.adv_clip_max)
    return advantages


class GRPOTrainer:
    def __init__(
        self,
        policy,
        reference_policy,
        optimizer,
        reward_manager: RewardManager,
        prompt_sampler: TcCurriculumSampler,
        validation_prompts: list[PromptSample],
        cfg,
        output_dir: str | Path,
        device: torch.device,
    ) -> None:
        self.policy = policy
        self.reference_policy = reference_policy
        self.optimizer = optimizer
        self.reward_manager = reward_manager
        self.prompt_sampler = prompt_sampler
        self.validation_prompts = validation_prompts
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.current_update = 0

    def _cfg_to_dict(self, value):
        if isinstance(value, list):
            return [self._cfg_to_dict(item) for item in value]
        if hasattr(value, "__dict__"):
            return {key: self._cfg_to_dict(val) for key, val in vars(value).items()}
        return value

    def _move_prompt(self, prompt: PromptSample) -> PromptSample:
        return PromptSample(
            batch=prompt.batch.to(self.device),
            target_raw=prompt.target_raw,
            target_scaled=prompt.target_scaled,
        )

    def _collect_group(self) -> tuple[PromptSample, dict, RewardBreakdown]:
        prompt = self._move_prompt(self.prompt_sampler.sample_prompt())
        scaled_target = torch.tensor([prompt.target_scaled], dtype=torch.float32, device=self.device)
        rollout_group = self.policy.sample_rl(
            prompt.batch,
            scaled_target,
            guide_w=self.cfg.sampling.guide_w,
            num_branches=self.cfg.sampling.num_branches,
            prefix_t=self.cfg.sampling.prefix_t,
            reward_interval=self.cfg.sampling.reward_interval,
            step_lr=self.cfg.sampling.step_lr,
        )
        phase = determine_phase(self.current_update, self.cfg)
        rewards = self.reward_manager.score_group(rollout_group, prompt.target_raw, phase=phase)
        return prompt, rollout_group, rewards

    def _group_loss(self, prompt: PromptSample, rollout_group: dict, rewards: RewardBreakdown) -> torch.Tensor:
        advantages = build_group_advantages(rewards, self.cfg)
        band_gap = rollout_group["band_gap"].to(self.device)

        total_loss = torch.zeros([], device=self.device)
        total_kl = torch.zeros([], device=self.device)
        total_steps = 0

        for branch_idx, branch in enumerate(rollout_group["branches"]):
            for step_idx, step_buffer in enumerate(branch["transitions"]):
                new_logs = self.policy.compute_step_log_probs_with_step_lr(
                    prompt.batch,
                    band_gap,
                    rollout_group["guide_w"],
                    step_buffer,
                    rollout_group["step_lr"],
                )
                with torch.no_grad():
                    ref_logs = self.reference_policy.compute_step_log_probs_with_step_lr(
                        prompt.batch,
                        band_gap,
                        rollout_group["guide_w"],
                        step_buffer,
                        rollout_group["step_lr"],
                    )

                old_total = step_buffer["old_logp_corrector"].to(self.device) + step_buffer["old_logp_predictor"].to(self.device)
                new_total = new_logs["total"]
                ratio = torch.exp(new_total - old_total)
                advantage = advantages[branch_idx, step_idx].to(self.device)
                unclipped = ratio * advantage
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.cfg.loss.clip_eps,
                    1.0 + self.cfg.loss.clip_eps,
                ) * advantage
                total_loss = total_loss - torch.minimum(unclipped, clipped)
                total_kl = total_kl + (new_total - ref_logs["total"])
                total_steps += 1

        total_loss = total_loss / max(total_steps, 1)
        total_kl = total_kl / max(total_steps, 1)
        return total_loss + self.cfg.loss.kl_beta * total_kl

    def _make_final_clean_state(self, sample_output: dict) -> dict:
        return {
            "num_atoms": sample_output["num_atoms"],
            "atom_types": sample_output["atom_types"].argmax(dim=-1) + 1,
            "frac_coords": sample_output["frac_coords"],
            "lattices": sample_output["lattices"],
        }

    def _sample_validation_states(self, policy, guide_w: float) -> list[dict]:
        clean_states = []
        with torch.no_grad():
            for prompt in self.validation_prompts:
                prompt = self._move_prompt(prompt)
                scaled_target = torch.tensor([prompt.target_scaled], dtype=torch.float32, device=self.device)
                final_state, _ = policy.sample(
                    prompt.batch,
                    scaled_target,
                    guide_w=guide_w,
                    step_lr=self.cfg.sampling.step_lr,
                )
                clean_states.append(self._make_final_clean_state(final_state))
        return clean_states

    def _evaluate_one_guide_w(self, policy, guide_w: float, phase: str) -> EvalSummary:
        clean_states = self._sample_validation_states(policy, guide_w)
        evaluation = self.reward_manager.evaluate_states(
            clean_states=clean_states,
            target_raws=[prompt.target_raw for prompt in self.validation_prompts],
            run_m3g=self.cfg.eval.run_m3g,
            m3g_subset_size=self.cfg.eval.m3g_subset_size,
            include_proxies=(phase == "C"),
        )

        mean_abs_error = evaluation["mean_abs_tcad_error"]
        mean_uplift = evaluation["mean_tcad_uplift"]
        metallic_rate = evaluation["metallic_pass_rate"]
        neg_eform_rate = evaluation["negative_formation_pass_rate"]
        cheap_pass_rate = evaluation["cheap_filter_pass_rate"]
        invalid_rate = evaluation["invalid_rate"]
        duplicate_rate = evaluation["duplicate_rate"]
        m3g_mean = evaluation["m3g_reward_mean"]
        proxy_reward_mean = evaluation["proxy_reward_mean"]

        composite = (
            cheap_pass_rate
            + 0.25 * mean_uplift / max(self.reward_manager.sigma_tcad, 1e-6)
            - mean_abs_error / max(self.reward_manager.sigma_tcad, 1e-6)
            - 0.25 * invalid_rate
            - 0.10 * duplicate_rate
        )
        if m3g_mean is not None and np.isfinite(m3g_mean):
            composite += 0.05 * m3g_mean
        if proxy_reward_mean is not None and np.isfinite(proxy_reward_mean):
            composite += 0.05 * proxy_reward_mean

        return EvalSummary(
            guide_w=guide_w,
            mean_abs_tcad_error=mean_abs_error,
            mean_tcad_uplift=mean_uplift,
            metallic_pass_rate=metallic_rate,
            negative_formation_pass_rate=neg_eform_rate,
            cheap_filter_pass_rate=cheap_pass_rate,
            invalid_rate=invalid_rate,
            duplicate_rate=duplicate_rate,
            m3g_reward_mean=m3g_mean,
            proxy_reward_mean=proxy_reward_mean,
            composite_score=float(composite),
        )

    def _select_summary(self, summaries: list[EvalSummary]) -> tuple[EvalSummary, EvalSummary, float]:
        summaries_sorted = sorted(summaries, key=lambda item: item.composite_score, reverse=True)
        best = summaries_sorted[0]
        deployment = next((item for item in summaries if abs(item.guide_w - 1.0) < 1e-6), None)
        if deployment is not None and best.composite_score > 0:
            relative_gap = (best.composite_score - deployment.composite_score) / abs(best.composite_score)
            deployment_choice = 1.0 if relative_gap <= self.cfg.eval.default_within_fraction else best.guide_w
        else:
            deployment_choice = best.guide_w
        deployment_summary = next(
            item for item in summaries if abs(item.guide_w - deployment_choice) < 1e-6
        )
        return best, deployment_summary, deployment_choice

    def evaluate(self, update_idx: int) -> dict:
        phase = determine_phase(update_idx - 1, self.cfg)
        summaries = [
            self._evaluate_one_guide_w(self.policy, guide_w, phase=phase)
            for guide_w in self.cfg.eval.guide_w_values
        ]
        best, deployment_summary, deployment_choice = self._select_summary(summaries)

        metrics = {
            "update": update_idx,
            "phase": phase,
            "policy": {
                "best_guide_w": best.guide_w,
                "deployment_guide_w": deployment_choice,
                "best_summary": asdict(best),
                "deployment_summary": asdict(deployment_summary),
                "summaries": [asdict(item) for item in summaries],
            },
        }

        if getattr(self.cfg.eval, "compare_reference", False):
            reference_guide_w_values = getattr(self.cfg.eval, "reference_guide_w_values", None)
            if reference_guide_w_values is None:
                reference_guide_w_values = self.cfg.eval.guide_w_values
            reference_summaries = [
                self._evaluate_one_guide_w(self.reference_policy, guide_w, phase=phase)
                for guide_w in reference_guide_w_values
            ]
            ref_best, ref_deployment_summary, ref_deployment_choice = self._select_summary(reference_summaries)
            metrics["reference"] = {
                "best_guide_w": ref_best.guide_w,
                "deployment_guide_w": ref_deployment_choice,
                "best_summary": asdict(ref_best),
                "deployment_summary": asdict(ref_deployment_summary),
                "summaries": [asdict(item) for item in reference_summaries],
            }
            metrics["comparison"] = {
                "cheap_filter_pass_ratio_vs_reference": deployment_summary.cheap_filter_pass_rate
                / max(ref_deployment_summary.cheap_filter_pass_rate, 1e-6),
                "mean_abs_tcad_error_delta_vs_reference": (
                    deployment_summary.mean_abs_tcad_error - ref_deployment_summary.mean_abs_tcad_error
                ),
                "mean_tcad_uplift_delta_vs_reference": (
                    deployment_summary.mean_tcad_uplift - ref_deployment_summary.mean_tcad_uplift
                ),
                "invalid_rate_delta_vs_reference": (
                    deployment_summary.invalid_rate - ref_deployment_summary.invalid_rate
                ),
                "duplicate_rate_delta_vs_reference": (
                    deployment_summary.duplicate_rate - ref_deployment_summary.duplicate_rate
                ),
                "m3g_reward_delta_vs_reference": (
                    None
                    if deployment_summary.m3g_reward_mean is None or ref_deployment_summary.m3g_reward_mean is None
                    else deployment_summary.m3g_reward_mean - ref_deployment_summary.m3g_reward_mean
                ),
            }

        with (self.output_dir / f"eval_update_{update_idx:06d}.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        return metrics

    def save_checkpoint(self, update_idx: int) -> None:
        checkpoint = {
            "state_dict": self.policy.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "update": update_idx,
            "config": self._cfg_to_dict(self.cfg),
        }
        torch.save(checkpoint, self.output_dir / f"rl_update_{update_idx:06d}.pt")

    def benchmark_rewards(self) -> dict:
        prompt = self._move_prompt(self.prompt_sampler.sample_prompt())
        scaled_target = torch.tensor([prompt.target_scaled], dtype=torch.float32, device=self.device)
        final_state, _ = self.policy.sample(
            prompt.batch,
            scaled_target,
            guide_w=self.cfg.sampling.guide_w,
            step_lr=self.cfg.sampling.step_lr,
        )
        bench = self.reward_manager.benchmark([self._make_final_clean_state(final_state)])
        serializable = {
            "bee_time_s": bench["bee_time_s"],
            "meg_time_s": bench["meg_time_s"],
            "m3g_time_s": bench["m3g_time_s"],
            "proxy_time_s": bench["proxy_time_s"],
        }
        with (self.output_dir / "reward_benchmark.json").open("w", encoding="utf-8") as handle:
            json.dump(serializable, handle, indent=2)
        return serializable

    def train(self) -> None:
        self.policy.train()
        self.reference_policy.eval()

        benchmark_done = False
        progress_path = self.output_dir / "train_progress.jsonl"
        for update_idx in range(self.cfg.schedule.max_updates):
            self.current_update = update_idx
            if not benchmark_done:
                self.benchmark_rewards()
                benchmark_done = True

            self.optimizer.zero_grad(set_to_none=True)
            update_loss = torch.zeros([], device=self.device)

            for _ in range(self.cfg.sampling.num_groups_per_update):
                prompt, rollout_group, rewards = self._collect_group()
                update_loss = update_loss + self._group_loss(prompt, rollout_group, rewards)

            update_loss = update_loss / max(self.cfg.sampling.num_groups_per_update, 1)
            if not torch.isfinite(update_loss):
                print(f"Skipping update {update_idx + 1}: non-finite GRPO loss {update_loss.item()}")
                with progress_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({
                        "time": time.time(),
                        "update": update_idx + 1,
                        "phase": determine_phase(update_idx, self.cfg),
                        "loss": float(update_loss.detach().cpu().item()),
                        "finite": False,
                        "skipped": True,
                    }) + "\n")
                continue
            update_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.loss.max_grad_norm)
            self.optimizer.step()

            checkpoint_saved = (update_idx + 1) % self.cfg.checkpoint.interval == 0
            eval_saved = (update_idx + 1) % self.cfg.eval.interval == 0
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "time": time.time(),
                    "update": update_idx + 1,
                    "phase": determine_phase(update_idx, self.cfg),
                    "loss": float(update_loss.detach().cpu().item()),
                    "finite": True,
                    "skipped": False,
                    "checkpoint_due": checkpoint_saved,
                    "eval_due": eval_saved,
                }) + "\n")

            if checkpoint_saved:
                self.save_checkpoint(update_idx + 1)

            if eval_saved:
                self.evaluate(update_idx + 1)
