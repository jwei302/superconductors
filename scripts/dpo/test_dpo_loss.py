"""End-to-end DPO loss smoke test (CPU-only, runs in ~1 min).

Three tests, each gates Day 2:
    1. log-2 identity:    when θ = θ_ref, dpo_step's loss = log 2 (boundary check).
    2. Antisymmetry:      ∂loss(w,l)/∂θ = −∂loss(l,w)/∂θ on the same noise.
    3. Smoke training:    50 grad steps on a synthetic pair set → loss decreases without NaN.

Loads the SFT'd checkpoint as the policy. Works on CPU; takes ~1 min on the login node.
"""

from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import torch
from torch_geometric.data import Batch, Data

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # project root + scripts/
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval_utils import load_model   # noqa: E402
from diffcsp.pl_modules.dpo_module import DPODiffusion   # noqa: E402
from diffcsp.pl_modules.diffusion_w_type import MAX_ATOMIC_NUM   # noqa: E402


def make_struct(n_atoms: int, max_z: int, generator: torch.Generator) -> Data:
    """One synthetic crystal."""
    return Data(
        num_atoms=torch.tensor([n_atoms]),
        num_nodes=n_atoms,
        atom_types=torch.randint(1, max_z + 1, (n_atoms,), generator=generator),
        frac_coords=torch.rand((n_atoms, 3), generator=generator),
        lengths=(torch.rand((1, 3), generator=generator) * 5 + 3.0),     # [3, 8] Å
        angles=(torch.rand((1, 3), generator=generator) * 30 + 75.0),    # [75, 105]°
        y=torch.tensor([1.0]),                                            # scaled property
    )


def make_pair_batch(n_pairs: int = 4, n_atoms: int = 3, seed: int = 0):
    g_w = torch.Generator().manual_seed(seed)
    g_l = torch.Generator().manual_seed(seed + 100)
    winners = Batch.from_data_list([make_struct(n_atoms, 10, g_w) for _ in range(n_pairs)])
    losers = Batch.from_data_list([make_struct(n_atoms, 10, g_l) for _ in range(n_pairs)])
    return winners, losers


def make_noise(struct, seed: int):
    g = torch.Generator().manual_seed(seed)
    rand_l = torch.randn(struct.num_graphs, 3, 3, generator=g)
    rand_x = torch.randn(struct.num_nodes, 3, generator=g)
    rand_t = torch.randn(struct.num_nodes, MAX_ATOMIC_NUM, generator=g)
    return rand_l, rand_x, rand_t


def make_batch(winners, losers, t, m, y, noise_w, noise_l, ref_w, ref_l):
    """Bundle into a SimpleNamespace-like object that dpo_step expects."""
    rand_l_w, rand_x_w, rand_t_w = noise_w
    rand_l_l, rand_x_l, rand_t_l = noise_l
    return types.SimpleNamespace(
        winner=winners, loser=losers, t=t, m=m, y=y,
        rand_l_w=rand_l_w, rand_x_w=rand_x_w, rand_t_w=rand_t_w,
        rand_l_l=rand_l_l, rand_x_l=rand_x_l, rand_t_l=rand_t_l,
        ref_loss_w=ref_w, ref_loss_l=ref_l,
    )


def build_dpo(beta: float = 200.0):
    print("[setup] loading SFT checkpoint...")
    sft, _, _ = load_model(Path("models/superconductor_generator").resolve(), load_data=False)
    sft.cpu().eval()
    dpo = DPODiffusion(policy=sft, beta=beta, total_timesteps=1000)
    return dpo


def compute_ref_losses(dpo, struct, t, m, y, noise):
    rand_l, rand_x, rand_t = noise
    with torch.no_grad():
        mse_l, mse_f, mse_a = dpo.per_component_mse(struct, t, rand_l, rand_x, rand_t, m, y)
    return torch.stack([mse_l, mse_f, mse_a], dim=1)


def test_log2_identity(dpo):
    print("\n[TEST 1] log-2 identity at θ = θ_ref")
    winners, losers = make_pair_batch(n_pairs=4, n_atoms=3, seed=0)
    t = 500
    m = torch.ones(4)
    y = torch.ones(4) * 1.2218
    noise_w = make_noise(winners, seed=42)
    noise_l = make_noise(losers,  seed=43)

    # ref losses = θ losses (since model unchanged) → z = 0
    ref_w = compute_ref_losses(dpo, winners, t, m, y, noise_w)
    ref_l = compute_ref_losses(dpo, losers,  t, m, y, noise_l)
    batch = make_batch(winners, losers, t, m, y, noise_w, noise_l, ref_w, ref_l)
    out = dpo.dpo_step(batch)
    loss_val = out["loss"].item()
    expected = math.log(2)
    print(f"  loss = {loss_val:.6f}   expected = {expected:.6f}   |diff| = {abs(loss_val - expected):.2e}")
    assert abs(loss_val - expected) < 1e-4, f"log-2 identity violated: {loss_val} vs {expected}"
    print(f"  σ(z) mean = {out['sigma_z'].mean().item():.6f}   (expected 0.5)")
    print("  ✓ pass")


def test_antisymmetry(dpo):
    print("\n[TEST 2] forward & gradient antisymmetry under (winner, loser) swap")
    # Forward antisymmetry: z(l, w) = −z(w, l) holds *everywhere* (it's how z is defined).
    # Gradient antisymmetry: only holds at θ = θ_ref (i.e., z = 0). Derivation:
    #     ∂loss(w,l)/∂θ = −(1 − σ(z))·∂z/∂θ
    #     ∂loss(l,w)/∂θ = +σ(−z)·∂(−z)/∂θ = −σ(z)·∂z/∂θ
    #   sum = (2σ(z) − 1)·∂z/∂θ, which equals 0 only when σ(z) = 0.5, i.e. z = 0.
    # So we test it at θ = θ_ref where z = 0 by construction; ∂z/∂θ is still nonzero,
    # so each individual gradient is nonzero and the sum-zero check is meaningful.
    winners, losers = make_pair_batch(n_pairs=4, n_atoms=3, seed=1)
    t = 750
    m = torch.ones(4)
    y = torch.ones(4) * 1.2218
    noise_w = make_noise(winners, seed=11)
    noise_l = make_noise(losers,  seed=12)
    ref_w = compute_ref_losses(dpo, winners, t, m, y, noise_w)
    ref_l = compute_ref_losses(dpo, losers,  t, m, y, noise_l)

    # --- forward antisymmetry: z(w, l) = −z(l, w) at θ = θ_ref (both = 0) and in general ---
    batch_a = make_batch(winners, losers, t, m, y, noise_w, noise_l, ref_w, ref_l)
    batch_b = make_batch(losers, winners, t, m, y, noise_l, noise_w, ref_l, ref_w)
    z_a = dpo.dpo_step(batch_a)["z"]
    z_b = dpo.dpo_step(batch_b)["z"]
    fwd_err = (z_a + z_b).abs().max().item()
    print(f"  forward: max |z(w,l) + z(l,w)| = {fwd_err:.2e}   (expected ≈ 0)")
    assert fwd_err < 1e-5, f"Forward antisymmetry violated on z: {fwd_err}"

    # --- gradient antisymmetry: at θ = θ_ref, ∂loss(w,l)/∂θ = −∂loss(l,w)/∂θ ---
    trainable = [p for p in dpo.policy.parameters() if p.requires_grad]
    print(f"  {len(trainable)} trainable tensors, {sum(p.numel() for p in trainable):,} params")

    for p in trainable:
        if p.grad is not None:
            p.grad.zero_()
    out_a = dpo.dpo_step(batch_a)
    out_a["loss"].backward()
    grads_a = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in trainable]

    for p in trainable:
        if p.grad is not None:
            p.grad.zero_()
    out_b = dpo.dpo_step(batch_b)
    out_b["loss"].backward()
    grads_b = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in trainable]

    max_abs = 0.0
    max_rel = 0.0
    nonzero_max = 0.0
    for ga, gb in zip(grads_a, grads_b):
        diff = (ga + gb).abs().max().item()
        scale = max(ga.abs().max().item(), gb.abs().max().item(), 1e-12)
        max_abs = max(max_abs, diff)
        max_rel = max(max_rel, diff / scale)
        nonzero_max = max(nonzero_max, ga.abs().max().item())
    print(f"  loss(w,l) = {out_a['loss'].item():.6f}   loss(l,w) = {out_b['loss'].item():.6f}   (both = log 2 at θ=θ_ref)")
    print(f"  max |grad_a|              = {nonzero_max:.2e}   (gradient is nonzero, so the test is meaningful)")
    print(f"  max |grad_a + grad_b|     = {max_abs:.2e}")
    print(f"  max |grad_a + grad_b|/scale = {max_rel:.2e}")
    assert nonzero_max > 1e-8, "Gradient is zero — test is degenerate"
    assert max_rel < 1e-3, f"Antisymmetry violated: max relative error {max_rel:.2e}"
    print("  ✓ pass")


def test_smoke_training(dpo, n_steps: int = 50):
    print(f"\n[TEST 3] {n_steps}-step training on synthetic pair (loss should decrease, no NaN)")
    winners, losers = make_pair_batch(n_pairs=4, n_atoms=3, seed=2)
    t_value = 500
    m = torch.ones(4)
    y = torch.ones(4) * 1.2218
    noise_w = make_noise(winners, seed=21)
    noise_l = make_noise(losers,  seed=22)
    ref_w = compute_ref_losses(dpo, winners, t_value, m, y, noise_w)
    ref_l = compute_ref_losses(dpo, losers,  t_value, m, y, noise_l)
    batch = make_batch(winners, losers, t_value, m, y, noise_w, noise_l, ref_w, ref_l)

    dpo.train()
    opt = torch.optim.Adam([p for p in dpo.policy.parameters() if p.requires_grad], lr=1e-4)
    losses, sigmas = [], []
    for step in range(n_steps):
        out = dpo.dpo_step(batch)
        loss = out["loss"]
        if torch.isnan(loss):
            raise RuntimeError(f"NaN at step {step}")
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        sigmas.append(out["sigma_z"].mean().item())
    print(f"  loss[0]   = {losses[0]:.4f}   σ(z)[0]   = {sigmas[0]:.4f}")
    print(f"  loss[24]  = {losses[24]:.4f}   σ(z)[24]  = {sigmas[24]:.4f}")
    print(f"  loss[-1]  = {losses[-1]:.4f}   σ(z)[-1]  = {sigmas[-1]:.4f}")
    assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
    assert sigmas[-1] > 0.5, f"σ(z) did not rise above 0.5: {sigmas[-1]:.4f}"
    print("  ✓ pass")


def main():
    torch.set_grad_enabled(False)
    dpo = build_dpo(beta=200.0)
    torch.set_grad_enabled(True)
    test_log2_identity(dpo)
    test_antisymmetry(dpo)
    test_smoke_training(dpo)
    print("\n=== all DPO loss tests passed ===")


if __name__ == "__main__":
    main()
