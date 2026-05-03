"""Sample crystals from a DPO checkpoint (Lightning DPODiffusion format).

The DPODiffusion module wraps a CSPDiffusion as `policy`. This script extracts the
policy and runs the existing sample-loop machinery — same code path as
`scripts/generation.py` but loading from a DPO ckpt instead of a Hydra run dir.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import DataLoader
from pymatgen.io.cif import CifWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval_utils import load_model, lattices_to_params_shape, get_crystals_list   # noqa: E402
from generation import SampleDataset, get_pymatgen, diffusion   # noqa: E402
from diffcsp.pl_modules.dpo_module import DPODiffusion   # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_model_path", required=True, type=Path,
                    help="Hydra run dir of π_ref (provides hparams.yaml + scalers + base CSPDiffusion to load DPO weights into).")
    ap.add_argument("--dpo_ckpt", required=True, type=Path,
                    help="Path to a DPO Lightning checkpoint (.ckpt). If empty/None, samples from π_ref.")
    ap.add_argument("--save_path", required=True, type=Path)
    ap.add_argument("--dataset", default="supccomb_12")
    ap.add_argument("--band_gap", type=float, default=1.2218)
    ap.add_argument("--guide_w", type=float, default=2.0)
    ap.add_argument("--batch_size", type=int, default=100)
    ap.add_argument("--num_batches_to_samples", type=int, default=20)
    ap.add_argument("--step_lr", type=float, default=-1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", default="")
    return ap.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    # 1. Load π_ref as the baseline policy structure
    print(f"[load] base CSPDiffusion ← {args.ref_model_path}")
    sft, _, _ = load_model(args.ref_model_path.resolve(), load_data=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sft.to(device)

    # 2. If a DPO checkpoint is given, overwrite policy weights from it
    if args.dpo_ckpt and str(args.dpo_ckpt).lower() not in ("none", ""):
        print(f"[load] DPO weights ← {args.dpo_ckpt}")
        ckpt = torch.load(str(args.dpo_ckpt), map_location=device, weights_only=False)
        state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        # DPODiffusion stores policy weights under the 'policy.' prefix.
        policy_state = {k[len("policy."):]: v for k, v in state.items() if k.startswith("policy.")}
        missing, unexpected = sft.load_state_dict(policy_state, strict=False)
        if missing:
            print(f"  missing keys: {len(missing)} (first 5: {missing[:5]})")
        if unexpected:
            print(f"  unexpected keys: {len(unexpected)} (first 5: {unexpected[:5]})")
    else:
        print("[load] no DPO ckpt provided → sampling from π_ref baseline")

    sft.eval()

    # 3. Sample
    band_gap_t = torch.tensor([args.band_gap], device=device).repeat(args.batch_size)
    test_set = SampleDataset(args.dataset, args.batch_size * args.num_batches_to_samples)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)
    if args.step_lr < 0:
        from eval_utils import recommand_step_lr
        step_lr = recommand_step_lr["gen"][args.dataset]
    else:
        step_lr = args.step_lr
    print(f"[sample] {len(test_set)} crystals at scaled y={args.band_gap}, guide_w={args.guide_w}, step_lr={step_lr}")

    t0 = time.time()
    frac_coords, atom_types, lattices, lengths, angles, num_atoms = diffusion(
        test_loader, sft, step_lr, band_gap_t, args.guide_w)
    print(f"[sample] done in {time.time()-t0:.1f}s")

    # 4. Convert to pymatgen + write CIFs
    args.save_path.mkdir(parents=True, exist_ok=True)
    crystal_list = get_crystals_list(frac_coords, atom_types, lengths, angles, num_atoms)
    n_written = 0
    for i, cd in enumerate(crystal_list):
        try:
            s = get_pymatgen(cd)
            CifWriter(s).write_file(str(args.save_path / f"{i+1}.cif"))
            n_written += 1
        except Exception as e:
            print(f"  [warn] failed to write structure {i+1}: {e}")
    print(f"[done] wrote {n_written}/{len(crystal_list)} CIFs to {args.save_path}")


if __name__ == "__main__":
    main()
