"""Final project figures.

Outputs (all to dpo_artifacts/figures/):
  fig1_beta_sweep.{pdf,png}        — main result: hit-rate vs β with chemistry-collapse warning
  fig2_energy_dist.{pdf,png}       — E/atom distribution shift
  fig3_chemistry_audit.{pdf,png}   — top-15 chemsys for baseline vs β=0.1 (reward-hacking exposure)
  fig4_training_components.{pdf,png} — per-component logit shares over training
  fig5_training_sigma.{pdf,png}    — σ(z) trajectories over training
  fig6_compute_matched.{pdf,png}   — DPO@N vs SFT best-of-N at matched compute
  table1_main_results.tex          — main results LaTeX table
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from collections import Counter

import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------- styling ----------------
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.8,
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
    "pdf.fonttype": 42,
})

OUT = Path("dpo_artifacts/figures")
OUT.mkdir(parents=True, exist_ok=True)

# Color scheme — gray for baseline; red flag for the reward-hacking β; greens for honest βs
COLOR = {
    "baseline": "#888888",
    "0.1":      "#d62728",  # red — collapsed
    "0.5":      "#ff7f0e",  # orange — borderline
    "1":        "#2ca02c",  # green — sweet spot
    "5":        "#1f77b4",  # blue
    "25":       "#9467bd",  # purple
}
BETAS = ["0.1", "0.5", "1", "5", "25"]


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    print(f"  wrote {OUT}/{name}.{{pdf,png}}")


# ---------------- load data ----------------
def load_pool(path):
    rows = json.loads(Path(path).read_text())
    return np.array([r["energy_per_atom"] for r in rows if r["valid"]])

baseline = load_pool("dpo_artifacts/day1_baseline_pool/scores_y10k.json")
sft8k    = load_pool("dpo_artifacts/eval/scores_sft_8k_seed1.json")
pools    = {b: load_pool(f"dpo_artifacts/eval/scores_dpo_b{b}.json") for b in BETAS}

T_25 = np.percentile(baseline, 25)
T_10 = np.percentile(baseline, 10)
T_05 = np.percentile(baseline, 5)

audit = json.loads(Path("dpo_artifacts/audit_summary.json").read_text())


# ============================================================
# FIG 1: β-sweep main result, hit-rate at three thresholds + chemistry-diversity overlay
# ============================================================
def fig_beta_sweep():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2),
                                   gridspec_kw={"wspace": 0.30})

    betas_f = [float(b) for b in BETAS]
    hr25 = [(pools[b] < T_25).mean() for b in BETAS]
    hr10 = [(pools[b] < T_10).mean() for b in BETAS]
    hr05 = [(pools[b] < T_05).mean() for b in BETAS]

    # Left: hit-rate vs β
    ax1.plot(betas_f, hr25, "o-", color="#2ca02c", label=f"hit@25th-pct (T={T_25:.2f})")
    ax1.plot(betas_f, hr10, "s-", color="#1f77b4", label=f"hit@10th-pct (T={T_10:.2f})")
    ax1.plot(betas_f, hr05, "^-", color="#9467bd", label=f"hit@5th-pct (T={T_05:.2f})")
    # Baseline horizontal lines
    for hr_b, ls, c in [(0.25, "--", "#2ca02c"), (0.10, "--", "#1f77b4"), (0.05, "--", "#9467bd")]:
        ax1.axhline(hr_b, color=c, ls=ls, alpha=0.4, lw=1)
    ax1.set_xscale("log")
    ax1.set_xlabel(r"DPO regularization strength $\beta$")
    ax1.set_ylabel("Hit-rate (fraction of pool below threshold)")
    ax1.set_title("Hit-rate vs $\\beta$ (5-point sweep, n=2000 each)")
    ax1.legend(loc="upper right", frameon=False)
    ax1.set_ylim(0, 1.0)
    ax1.grid(True, alpha=0.25)

    # Annotate the reward-hacking warning at β=0.1
    ax1.annotate(
        "chemistry-collapsed\n(reward hacking)",
        xy=(0.1, hr25[0]), xytext=(0.25, 0.78),
        fontsize=9, color="#d62728", ha="left",
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2),
    )

    # Right: chemistry-diversity diagnostic (this is what disqualifies β=0.1)
    n_chemsys = [audit[f"dpo_b{b}"]["n_chemsys"] for b in BETAS]
    top10_share = [audit[f"dpo_b{b}"]["top10_chemsys_share"] for b in BETAS]
    base_n = audit["baseline"]["n_chemsys"]
    base_top10 = audit["baseline"]["top10_chemsys_share"]

    ax2b = ax2.twinx()
    ax2.plot(betas_f, n_chemsys, "o-", color="#444444", label="# unique chemsys")
    ax2.axhline(base_n, color="#444444", ls="--", alpha=0.5)
    ax2.text(25, base_n - 30, "baseline", color="#444444", fontsize=9, ha="right")
    ax2b.plot(betas_f, top10_share, "s-", color="#d62728", label="top-10 chemsys share")
    ax2b.axhline(base_top10, color="#d62728", ls="--", alpha=0.5)

    ax2.set_xscale("log")
    ax2.set_xlabel(r"DPO regularization strength $\beta$")
    ax2.set_ylabel("# unique chemical systems (out of 2000 samples)", color="#444444")
    ax2b.set_ylabel("share of pool in top-10 chemsys", color="#d62728")
    ax2.set_title("Chemistry diversity (the audit)")
    ax2.tick_params(axis="y", labelcolor="#444444")
    ax2b.tick_params(axis="y", labelcolor="#d62728")
    ax2.grid(True, alpha=0.25)
    # combine legends
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="center right", frameon=False)

    fig.suptitle("Diffusion-DPO β-sweep: hit-rate gain at small β is largely chemistry collapse",
                 fontsize=12, y=1.02)
    save(fig, "fig1_beta_sweep")
    plt.close(fig)


# ============================================================
# FIG 2: energy distribution KDEs
# ============================================================
def fig_energy_dist():
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bins = np.linspace(-13, -2, 80)

    # Baseline as filled histogram
    ax.hist(baseline, bins=bins, alpha=0.30, color=COLOR["baseline"], density=True,
            label=f"baseline (n={len(baseline)})", edgecolor="none")

    for b in BETAS:
        e = pools[b]
        n_uniq = audit[f"dpo_b{b}"]["n_chemsys"]
        ls = "-" if b != "0.1" else (0, (3, 2))   # dashed for the collapsed one
        lw = 2.0 if b == "1" else 1.5
        ax.hist(e, bins=bins, histtype="step", density=True,
                color=COLOR[b], linestyle=ls, linewidth=lw,
                label=fr"$\beta$={b} (n={len(e)}, #cs={n_uniq})")

    ax.axvline(T_25, color="black", ls=":", alpha=0.5)
    ax.text(T_25, ax.get_ylim()[1]*0.95, " T_25", fontsize=9, va="top")
    ax.set_xlabel("CHGNet energy per atom (eV/atom)")
    ax.set_ylabel("density")
    ax.set_title("Energy-per-atom distribution: β=0.1's shift is driven by chemistry, not stability")
    ax.legend(loc="upper left", frameon=False)
    save(fig, "fig2_energy_dist")
    plt.close(fig)


# ============================================================
# FIG 3: chemistry distribution audit
# ============================================================
def fig_chemistry_audit():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), gridspec_kw={"wspace": 0.45})

    for ax, name, color in [
        (axes[0], "baseline", COLOR["baseline"]),
        (axes[1], "dpo_b1",   COLOR["1"]),
        (axes[2], "dpo_b0.1", COLOR["0.1"]),
    ]:
        top = audit[name]["chemsys_top20"][:15]
        labels = [t[0] for t in top]
        counts = [t[1] for t in top]
        ax.barh(range(len(labels))[::-1], counts, color=color, alpha=0.85)
        ax.set_yticks(range(len(labels))[::-1])
        ax.set_yticklabels(labels, fontsize=9)
        n_cs = audit[name]["n_chemsys"]
        share = audit[name]["top10_chemsys_share"]
        title = name.replace("dpo_b", r"DPO $\beta$=") if "dpo" in name else name
        ax.set_title(f"{title}\n#chemsys={n_cs}, top-10 share={share:.1%}")
        ax.set_xlabel("# samples (out of 2000)")

    fig.suptitle("Top-15 chemical systems per pool — β=0.1 concentrates on Ta/Re-rich (5d transition metals)",
                 fontsize=11, y=1.02)
    save(fig, "fig3_chemistry_audit")
    plt.close(fig)


# ============================================================
# FIG 4: per-component logit shares over training
# ============================================================
def fig_training_components():
    fig, axes = plt.subplots(1, len(BETAS), figsize=(15, 3.8), sharey=True,
                             gridspec_kw={"wspace": 0.05})
    for ax, b in zip(axes, BETAS):
        df = pd.read_csv(f"dpo_artifacts/dpo_b{b}/dpo/version_0/metrics.csv")
        sl = df["train/share_lattice"].dropna().values
        sf = df["train/share_coord"].dropna().values
        sa = df["train/share_type"].dropna().values
        x = np.arange(len(sl))
        # Smooth
        win = 50
        smooth = lambda a: np.convolve(a, np.ones(win)/win, mode="valid")
        ax.plot(x[win-1:], smooth(sl), color="#1f77b4", label="lattice")
        ax.plot(x[win-1:], smooth(sf), color="#2ca02c", label="coord")
        ax.plot(x[win-1:], smooth(sa), color="#d62728", label="type")
        ax.set_title(fr"$\beta$={b}")
        ax.set_xlabel("training step")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.2)
    axes[0].set_ylabel("share of |logit|")
    axes[-1].legend(loc="upper right", frameon=False, fontsize=8)
    fig.suptitle("Per-component DPO logit share over training (50-step rolling avg)",
                 fontsize=11, y=1.04)
    save(fig, "fig4_training_components")
    plt.close(fig)


# ============================================================
# FIG 5: σ(z) trajectory over training
# ============================================================
def fig_training_sigma():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"wspace": 0.30})
    for b in BETAS:
        df = pd.read_csv(f"dpo_artifacts/dpo_b{b}/dpo/version_0/metrics.csv")
        sz = df["train/sigma_z_mean"].dropna().values
        tl = df["train/loss"].dropna().values
        win = 50
        smooth = lambda a: np.convolve(a, np.ones(win)/win, mode="valid")
        x = np.arange(len(sz))[win-1:]
        axes[0].plot(x, smooth(sz), color=COLOR[b], label=fr"$\beta$={b}")
        axes[1].plot(x, smooth(tl), color=COLOR[b], label=fr"$\beta$={b}")
    axes[0].axhline(0.5, ls=":", color="black", alpha=0.5)
    axes[0].text(0, 0.51, " σ(z)=0.5 (no preference)", fontsize=9)
    axes[0].set_xlabel("training step"); axes[0].set_ylabel(r"$\sigma(z)$ (winner score)")
    axes[0].set_title(r"σ(z) trajectory: rises with training, saturates faster at large $\beta$")
    axes[0].set_ylim(0.4, 1.0)
    axes[0].legend(frameon=False, ncol=2)
    axes[0].grid(True, alpha=0.25)

    axes[1].set_xlabel("training step"); axes[1].set_ylabel("DPO training loss")
    axes[1].set_title("DPO training loss (50-step rolling avg)")
    axes[1].set_yscale("log")
    axes[1].legend(frameon=False, ncol=2)
    axes[1].grid(True, alpha=0.25)

    save(fig, "fig5_training_sigma")
    plt.close(fig)


# ============================================================
# FIG 6: compute-matched comparison — DPO@N vs SFT best-of-N
# ============================================================
def fig_compute_matched():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), gridspec_kw={"wspace": 0.32})

    pool_data = [
        ("SFT (1×)",     baseline,    COLOR["baseline"]),
        ("SFT (4×)",     sft8k,       "#666666"),
        ("DPO β=1 (1×)", pools["1"],  COLOR["1"]),
    ]

    names   = [p[0] for p in pool_data]
    colors  = [p[2] for p in pool_data]
    hits    = [(p[1] < T_25).sum() for p in pool_data]
    rates   = [(p[1] < T_25).mean() for p in pool_data]

    x = np.arange(len(names))
    axes[0].bar(x, rates, color=colors, alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, fontsize=10)
    axes[0].set_ylabel("hit-rate")
    axes[0].set_title("Hit-rate at $T_{25}$")
    for xi, r in zip(x, rates):
        axes[0].text(xi, r + 0.01, f"{r:.1%}", ha="center", fontsize=9)
    axes[0].set_ylim(0, max(rates) * 1.20)

    K = 500
    top_meds = [np.median(np.sort(p[1])[:K]) for p in pool_data]
    axes[1].bar(x, top_meds, color=colors, alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, fontsize=10)
    axes[1].set_ylabel("top-500 median (eV/atom)")
    axes[1].set_title("Top-500 screening quality")
    axes[1].invert_yaxis()
    for xi, m in zip(x, top_meds):
        axes[1].text(xi, m, f"{m:.2f}", ha="center", va="bottom", fontsize=9)

    save(fig, "fig6_compute_matched")
    plt.close(fig)


# ============================================================
# TABLE 1: main results LaTeX
# ============================================================
def table_main():
    rows = []
    def srow(name, e):
        rows.append({
            "name": name,
            "n": len(e),
            "median": np.median(e),
            "hr25": (e < T_25).mean(),
            "hr10": (e < T_10).mean(),
            "hr05": (e < T_05).mean(),
            "top500": np.median(np.sort(e)[:500]),
        })
    srow("baseline (SFT)", baseline)
    for b in BETAS:
        srow(fr"DPO $\beta$={b}", pools[b])

    # also include a chemsys flag column
    nm_to_audit = {f"DPO $\\beta$={b}": f"dpo_b{b}" for b in BETAS}
    nm_to_audit["baseline (SFT)"] = "baseline"

    out = []
    out.append(r"\begin{tabular}{lrrrrrrr}\toprule")
    out.append(r"pool & n & median $E$ & hr@25\% & hr@10\% & hr@5\% & top-500 med & \#chemsys \\")
    out.append(r"\midrule")
    for r in rows:
        cs = audit[nm_to_audit[r['name']]]["n_chemsys"]
        flag = r" \textcolor{red}{$\dagger$}" if cs < 300 else ""
        out.append(
            f"{r['name']}{flag} & {r['n']} & {r['median']:.3f} & "
            f"{r['hr25']:.2%} & {r['hr10']:.2%} & {r['hr05']:.2%} & "
            f"{r['top500']:.3f} & {cs} \\\\"
        )
    out.append(r"\bottomrule\end{tabular}")
    out.append(r"% $\dagger$ chemistry-collapsed (top-10 share > 60\%); see audit.")

    tex = "\n".join(out).replace("%", r"\%")
    Path(OUT / "table1_main_results.tex").write_text(tex)
    print(f"  wrote {OUT}/table1_main_results.tex")


# ---------------- run all ----------------
print("Generating figures...")
fig_beta_sweep()
fig_energy_dist()
fig_chemistry_audit()
fig_training_components()
fig_training_sigma()
fig_compute_matched()
table_main()
print(f"All figures and table written to {OUT}/")
