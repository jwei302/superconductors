#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from pymatgen.core import Structure


REPO_ROOT = Path(__file__).resolve().parents[1]
MEGNET_ROOT = REPO_ROOT / "external" / "megnet"
if str(MEGNET_ROOT) not in sys.path:
    sys.path.insert(0, str(MEGNET_ROOT))

from megnet.utils.models import load_model  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cif", required=True, help="Path to a CIF file to score.")
    parser.add_argument(
        "--formation-model",
        default="Eform_MP_2019",
        help="Pretrained MEGNet model name for formation energy.",
    )
    parser.add_argument(
        "--bandgap-model",
        default="Bandgap_MP_2018",
        help="Pretrained MEGNet model name for band gap.",
    )
    parser.add_argument(
        "--metal-threshold",
        type=float,
        default=0.05,
        help="If predicted band gap is <= this value, mark as metallic-like.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cif_path = Path(args.cif).resolve()
    if not cif_path.is_file():
        raise FileNotFoundError(f"CIF not found: {cif_path}")

    structure = Structure.from_file(cif_path)

    started = time.perf_counter()
    formation_model = load_model(args.formation_model)
    formation_load_s = time.perf_counter() - started

    started = time.perf_counter()
    bandgap_model = load_model(args.bandgap_model)
    bandgap_load_s = time.perf_counter() - started

    started = time.perf_counter()
    formation_energy = float(formation_model.predict_structure(structure).ravel()[0])
    formation_predict_s = time.perf_counter() - started

    started = time.perf_counter()
    band_gap = float(bandgap_model.predict_structure(structure).ravel()[0])
    bandgap_predict_s = time.perf_counter() - started

    print(f"CIF: {cif_path}")
    print(f"Formula: {structure.composition.reduced_formula}")
    print(f"Sites: {len(structure)}")
    print(f"Formation model: {args.formation_model}")
    print(f"Band gap model: {args.bandgap_model}")
    print(f"Predicted formation energy (eV/atom): {formation_energy:.6f}")
    print(f"Predicted band gap (eV): {band_gap:.6f}")
    print(f"Metallic-like (Eg <= {args.metal_threshold} eV): {band_gap <= args.metal_threshold}")
    print(f"Formation model load time (s): {formation_load_s:.4f}")
    print(f"Band gap model load time (s): {bandgap_load_s:.4f}")
    print(f"Formation prediction time (s): {formation_predict_s:.4f}")
    print(f"Band gap prediction time (s): {bandgap_predict_s:.4f}")


if __name__ == "__main__":
    main()
