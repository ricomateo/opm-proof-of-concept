"""Diagnostic plots for the SPE9 dataset.

Reads dataset.csv and runs_log.csv from the project root and writes a handful
of PNG figures into plots/. The point is to make the variance, the lever
sensitivities, and the per-simulation trajectories visually obvious.

Usage:
    python scripts/plot_dataset.py
    python scripts/plot_dataset.py --dataset other.csv --log other_log.csv --out plots_b
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LEVERS = [
    "qwinj_rate",
    "qo_rate_high",
    "k_mult",
    "phi_mult",
    "p_init",
    "pb_shift",
]

DYNAMIC_COLS = [
    "Caudal_Prod_Petroleo_bbl",
    "Caudal_Prod_Gas_Mpc",
    "Caudal_Iny_Agua_bbl",
    "Prod_Acumulada_Petroleo",
    "Prod_Acumulada_Gas",
    "Prod_Acumulada_Agua",
    "Iny_Acumulada_Agua",
    "Presion_Reservorio_psi",
]

CORRELATION_COLS = [
    "Porosidad",
    "Permeabilidad_mD",
    "Presion_Burbuja_psi",
    "Bo_rb_stb",
    "Bg_rb_scf",
    "Rs_scf_stb",
    "Caudal_Prod_Petroleo_bbl",
    "Caudal_Prod_Gas_Mpc",
    "Caudal_Iny_Agua_bbl",
    "Prod_Acumulada_Petroleo",
    "Prod_Acumulada_Gas",
    "Prod_Acumulada_Agua",
    "Iny_Acumulada_Agua",
    "Presion_Reservorio_psi",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot the SPE9 dataset")
    p.add_argument("--dataset", default="dataset.csv")
    p.add_argument("--log", default="runs_log.csv")
    p.add_argument("--out", default="plots")
    return p.parse_args()


def add_step_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["step"] = df.groupby("sim_id").cumcount()
    return df


def plot_pressure_trajectories(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for _, grp in df.groupby("sim_id"):
        ax.plot(grp["step"], grp["Presion_Reservorio_psi"], lw=0.6, alpha=0.35, color="steelblue")
    median = df.groupby("step")["Presion_Reservorio_psi"].median()
    ax.plot(median.index, median.values, color="black", lw=2.0, label="median across sims")
    ax.set_xlabel("Report step (each ~10 days)")
    ax.set_ylabel("FPR (psi)")
    ax.set_title("Reservoir pressure trajectories - 100 simulations")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "pressure_trajectories.png", dpi=120)
    plt.close(fig)


def plot_cumulative_production(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    titles = {
        "Prod_Acumulada_Petroleo": "Cumulative oil produced (STB)",
        "Prod_Acumulada_Gas": "Cumulative gas produced (SCF)",
        "Prod_Acumulada_Agua": "Cumulative water produced (STB)",
        "Iny_Acumulada_Agua": "Cumulative water injected (STB)",
    }
    for ax, (col, title) in zip(axes.flat, titles.items()):
        for _, grp in df.groupby("sim_id"):
            ax.plot(grp["step"], grp[col], lw=0.5, alpha=0.3, color="darkgreen")
        median = df.groupby("step")[col].median()
        ax.plot(median.index, median.values, color="black", lw=1.8, label="median")
        ax.set_title(title)
        ax.set_xlabel("Report step")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Cumulative production and injection per simulation")
    fig.tight_layout()
    fig.savefig(out / "cumulative_production.png", dpi=120)
    plt.close(fig)


def plot_lever_distributions(log: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, lever in zip(axes.flat, LEVERS):
        ax.hist(log[lever], bins=20, color="slategray", edgecolor="black")
        ax.set_title(lever)
        ax.set_ylabel("count")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Lever distributions across the LHS sample")
    fig.tight_layout()
    fig.savefig(out / "lever_distributions.png", dpi=120)
    plt.close(fig)


def plot_lever_vs_fpr(log: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    for ax, lever in zip(axes.flat, LEVERS):
        ax.scatter(log[lever], log["fpr_min_psi"], s=18, alpha=0.7, label="FPR min")
        ax.scatter(log[lever], log["fpr_max_psi"], s=18, alpha=0.7, label="FPR max")
        ax.set_xlabel(lever)
        ax.set_ylabel("FPR (psi)")
        ax.grid(True, alpha=0.3)
    axes.flat[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Lever sensitivity: FPR min/max per simulation vs each lever")
    fig.tight_layout()
    fig.savefig(out / "lever_vs_fpr.png", dpi=120)
    plt.close(fig)


def plot_feature_correlations(df: pd.DataFrame, out: Path) -> None:
    cols = [c for c in CORRELATION_COLS if c in df.columns]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticklabels(cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}",
                    ha="center", va="center", color="black", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.04)
    ax.set_title("Pearson correlation - schema features (pooled rows)")
    fig.tight_layout()
    fig.savefig(out / "feature_correlations.png", dpi=120)
    plt.close(fig)


def plot_pvt_curves(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    sample_ids = df["sim_id"].drop_duplicates().sample(n=min(8, df["sim_id"].nunique()), random_state=0)
    palette = plt.cm.viridis(np.linspace(0, 1, len(sample_ids)))
    for color, sid in zip(palette, sample_ids):
        sub = df[df["sim_id"] == sid].sort_values("Presion_Reservorio_psi")
        axes[0].plot(sub["Presion_Reservorio_psi"], sub["Bo_rb_stb"], color=color, lw=1.0, alpha=0.9)
        axes[1].plot(sub["Presion_Reservorio_psi"], sub["Bg_rb_scf"], color=color, lw=1.0, alpha=0.9)
        axes[2].plot(sub["Presion_Reservorio_psi"], sub["Rs_scf_stb"], color=color, lw=1.0, alpha=0.9)
    axes[0].set_title("Bo (rb/stb) vs FPR")
    axes[1].set_title("Bg (rb/scf) vs FPR")
    axes[2].set_title("Rs (scf/stb) vs FPR")
    for ax in axes:
        ax.set_xlabel("FPR (psi)")
        ax.grid(True, alpha=0.3)
    fig.suptitle("PVT-derived columns vs reservoir pressure (8 random sims)")
    fig.tight_layout()
    fig.savefig(out / "pvt_curves.png", dpi=120)
    plt.close(fig)


def plot_per_sim_panel(df: pd.DataFrame, out: Path) -> None:
    """Pick three sims (low/med/high FPR_min) and plot rates + cumulatives + Pr together."""
    last_pr = df.groupby("sim_id")["Presion_Reservorio_psi"].min().sort_values()
    chosen = [last_pr.index[0], last_pr.index[len(last_pr) // 2], last_pr.index[-1]]
    fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharex=True)
    for col_i, sid in enumerate(chosen):
        sub = df[df["sim_id"] == sid]
        axes[0, col_i].plot(sub["step"], sub["Presion_Reservorio_psi"], color="firebrick")
        axes[0, col_i].set_title(f"sim_{sid:04d} - FPR")
        axes[0, col_i].set_ylabel("psi")
        axes[1, col_i].plot(sub["step"], sub["Caudal_Prod_Petroleo_bbl"], label="oil prod", color="goldenrod")
        axes[1, col_i].plot(sub["step"], sub["Caudal_Iny_Agua_bbl"], label="water inj", color="steelblue")
        axes[1, col_i].set_ylabel("STB/day")
        axes[1, col_i].legend(fontsize=7)
        axes[2, col_i].plot(sub["step"], sub["Prod_Acumulada_Petroleo"], label="Np", color="goldenrod")
        axes[2, col_i].plot(sub["step"], sub["Iny_Acumulada_Agua"], label="Winj", color="steelblue")
        axes[2, col_i].set_ylabel("STB cumulative")
        axes[2, col_i].set_xlabel("Report step")
        axes[2, col_i].legend(fontsize=7)
        for r in range(3):
            axes[r, col_i].grid(True, alpha=0.3)
    fig.suptitle("Three simulations: lowest, median, and highest final FPR")
    fig.tight_layout()
    fig.savefig(out / "per_sim_panel.png", dpi=120)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    dataset_path = (PROJECT_ROOT / args.dataset) if not Path(args.dataset).is_absolute() else Path(args.dataset)
    log_path = (PROJECT_ROOT / args.log) if not Path(args.log).is_absolute() else Path(args.log)
    out_dir = (PROJECT_ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(exist_ok=True)

    df = pd.read_csv(dataset_path)
    log = pd.read_csv(log_path)
    df = add_step_index(df)

    print(f"[plot] dataset {df.shape}, log {log.shape}, output -> {out_dir}")

    plot_pressure_trajectories(df, out_dir)
    plot_cumulative_production(df, out_dir)
    plot_lever_distributions(log, out_dir)
    plot_lever_vs_fpr(log, out_dir)
    plot_feature_correlations(df, out_dir)
    plot_pvt_curves(df, out_dir)
    plot_per_sim_panel(df, out_dir)

    for png in sorted(out_dir.glob("*.png")):
        print(f"  wrote {png.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
