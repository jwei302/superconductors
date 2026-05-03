"""End-to-end DPO pipeline smoke test on synthetic pairs (CPU, ~1 min).

Builds a fake preference dataset (10 random crystals, fake "stability" scores), constructs
pairs via top/bottom-quartile selection, pre-caches π_ref's per-component MSEs, saves to
disk, loads via PreferenceDataset + DataLoader, runs 50 DPO grad steps. End-to-end
exercise of every code path Day 2 will use.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval_utils import load_model   # noqa: E402
from diffcsp.pl_modules.dpo_module import DPODiffusion   # noqa: E402
from diffcsp.pl_modules.diffusion_w_type import MAX_ATOMIC_NUM   # noqa: E402
from diffcsp.pl_data.preference_dataset import make_loader   # noqa: E402


def make_struct(n_atoms: int, max_z: int, generator: torch.Generator) -> Data:
    return Data(
        num_atoms=torch.tensor([n_atoms]),
        num_nodes=n_atoms,
        atom_types=torch.randint(1, max_z + 1, (n_atoms,), generator=generator),
        frac_coords=torch.rand((n_atoms, 3), generator=generator),
        lengths=(torch.rand((1, 3), generator=generator) * 5 + 3.0),
        angles=(torch.rand((1, 3), generator=generator) * 30 + 75.0),
        y=torch.tensor([1.2218]),       # scaled Tc=10K
    )


def synthesize_pool(n: int = 10, n_atoms: int = 3, seed: int = 0):
    """Generate n synthetic crystals + a fake 'stability' score (lower=better)."""
    g = torch.Generator().manual_seed(seed)
    structs = [make_struct(n_atoms, 10, g) for _ in range(n)]
    # Fake score = norm of frac_coords + lattice trace — just to give pairs *something* to rank by.
    scores = []
    for s in structs:
        score = float(s.frac_coords.norm() + s.lengths.sum() * 0.1)
        scores.append(score)
    return structs, scores


def build_pairs(structs, scores, top_frac: float = 0.4, bottom_frac: float = 0.4,
                gap_margin: float = 0.0):
    """Top-fraction = winners (lower score = better stability), bottom-fraction = losers."""
    n = len(structs)
    order = sorted(range(n), key=lambda i: scores[i])
    n_top = max(1, int(n * top_frac))
    n_bot = max(1, int(n * bottom_frac))
    winners_idx = order[:n_top]
    losers_idx = order[-n_bot:]
    pairs = []
    for w in winners_idx:
        for l in losers_idx:
            gap = scores[l] - scores[w]
            if gap >= gap_margin:
                pairs.append((w, l, gap))
    return pairs


def cache_pair_entries(dpo, structs, pairs, t_value: int = 500, m_value: int = 1, k_draws: int = 2):
    """For each pair, sample k_draws random noise tuples and cache π_ref's per-component MSEs."""
    entries = []
    g = torch.Generator().manual_seed(7)
    for w_idx, l_idx, gap in pairs:
        for k in range(k_draws):
            sw = structs[w_idx]
            sl = structs[l_idx]
            # Noise tensors (same shape as in CSPDiffusion.forward, per-graph).
            rand_l_w = torch.randn(1, 3, 3, generator=g).squeeze(0)
            rand_x_w = torch.randn(sw.num_nodes, 3, generator=g)
            rand_t_w = torch.randn(sw.num_nodes, MAX_ATOMIC_NUM, generator=g)
            rand_l_l = torch.randn(1, 3, 3, generator=g).squeeze(0)
            rand_x_l = torch.randn(sl.num_nodes, 3, generator=g)
            rand_t_l = torch.randn(sl.num_nodes, MAX_ATOMIC_NUM, generator=g)

            # Compute π_ref per-component MSE (single-pair forward with no_grad).
            from torch_geometric.data import Batch
            wbatch = Batch.from_data_list([sw])
            lbatch = Batch.from_data_list([sl])
            with torch.no_grad():
                m_t = torch.tensor([float(m_value)])
                y_t = torch.tensor([1.2218])
                ml_w, mf_w, ma_w = dpo.per_component_mse(
                    wbatch, t_value, rand_l_w.unsqueeze(0), rand_x_w, rand_t_w, m_t, y_t)
                ml_l, mf_l, ma_l = dpo.per_component_mse(
                    lbatch, t_value, rand_l_l.unsqueeze(0), rand_x_l, rand_t_l, m_t, y_t)
            ref_w = torch.stack([ml_w[0], mf_w[0], ma_w[0]])
            ref_l = torch.stack([ml_l[0], mf_l[0], ma_l[0]])

            entries.append({
                "winner": sw, "loser": sl,
                "t": t_value, "m": m_value,
                "rand_l_w": rand_l_w, "rand_x_w": rand_x_w, "rand_t_w": rand_t_w,
                "rand_l_l": rand_l_l, "rand_x_l": rand_x_l, "rand_t_l": rand_t_l,
                "ref_loss_w": ref_w, "ref_loss_l": ref_l,
                "reward_gap": gap,
            })
    return entries


def main():
    print("=== Building synthetic pool of 10 crystals ===")
    structs, scores = synthesize_pool(n=10, n_atoms=3, seed=0)
    print(f"  scores range: [{min(scores):.3f}, {max(scores):.3f}]")

    print("\n=== Forming pairs (top-40% vs bottom-40%) ===")
    pairs = build_pairs(structs, scores, top_frac=0.4, bottom_frac=0.4, gap_margin=0.0)
    print(f"  {len(pairs)} pairs")
    if not pairs:
        raise RuntimeError("no pairs!")

    print("\n=== Loading SFT model as π_ref + π_θ initial weights ===")
    sft, _, _ = load_model(Path("models/superconductor_generator").resolve(), load_data=False)
    sft.cpu().eval()
    # β=1 (small) for a synthetic test — the "reward signal" here (frac_coord norm + lattice trace)
    # isn't physically meaningful, so we just want to demonstrate the pipeline runs without instability.
    dpo = DPODiffusion(policy=sft, beta=1.0, total_timesteps=1000)

    print("\n=== Caching π_ref per-component MSEs (k=2 draws/pair) ===")
    entries = cache_pair_entries(dpo, structs, pairs, t_value=500, m_value=1, k_draws=2)
    print(f"  {len(entries)} cached entries")
    out_path = Path("dpo_artifacts/test_pairs_v0.pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(entries, out_path)
    print(f"  saved → {out_path}")

    print("\n=== Loading via PreferenceDataset / DataLoader (full-batch) ===")
    # Full-batch (32 entries, all share same t and m) so gradient averages cleanly.
    loader = make_loader(out_path, batch_size=len(entries), shuffle=False)
    print(f"  loader has {len(loader)} batches")

    print("\n=== 50 DPO grad steps ===")
    dpo.train()
    opt = torch.optim.Adam([p for p in dpo.policy.parameters() if p.requires_grad], lr=1e-4)
    steps = 0
    losses, sigmas = [], []
    while steps < 50:
        for batch in loader:
            out = dpo.dpo_step(batch)
            loss = out["loss"]
            if torch.isnan(loss):
                raise RuntimeError(f"NaN at step {steps}")
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
            sigmas.append(out["sigma_z"].mean().item())
            steps += 1
            if steps >= 50:
                break
    print(f"  loss[0]   = {losses[0]:.4f}   σ(z)[0]   = {sigmas[0]:.4f}   (initial; expected ≈ log 2 = 0.6931)")
    print(f"  loss[24]  = {losses[24]:.4f}   σ(z)[24]  = {sigmas[24]:.4f}")
    print(f"  loss[-1]  = {losses[-1]:.4f}   σ(z)[-1]  = {sigmas[-1]:.4f}")
    # Synthetic-reward test: the data has no real signal so the optimizer just memorizes the
    # batch. Success criterion: loss decreases and σ(z) ends above 0.5 (model "prefers" winners).
    assert max(losses) < 100, f"Loss exploded: max = {max(losses):.2f}"
    assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
    assert sigmas[-1] > 0.5, f"σ(z) did not move above 0.5: {sigmas[-1]:.4f}"
    print(f"  Δσ(z) = {sigmas[-1] - sigmas[0]:+.4f}   Δloss = {losses[-1] - losses[0]:+.4f}")
    print("\n=== pipeline test passed: dataset → loader → DPO step → backward all work ===")


if __name__ == "__main__":
    main()
