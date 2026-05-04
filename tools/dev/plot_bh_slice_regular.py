#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render regular Cartesian BH slices from CSV (pivot + imshow)."
    )
    parser.add_argument("--csv", default="/tmp/bh_cartesian_slice64.csv")
    parser.add_argument("--alpha-png", default="/tmp/bh_alpha_slice_regular.png")
    parser.add_argument(
        "--ricci-png", default="/tmp/bh_ricci_trace_slice_regular.png"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    alpha = df.pivot(index="y", columns="x", values="alpha").values
    ricci_trace = df.pivot(index="y", columns="x", values="ricci_trace").values
    extent = [df["x"].min(), df["x"].max(), df["y"].min(), df["y"].max()]

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    im = ax.imshow(
        alpha,
        extent=extent,
        origin="lower",
        cmap="magma",
        interpolation="bilinear",
        vmin=0.0,
        vmax=1.0,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("alpha")
    ax.set_title("Alpha slice z=0 (64x64, Cartesian)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.grid(True, color="white", linestyle="--", linewidth=0.35, alpha=0.25)
    fig.savefig(args.alpha_png, dpi=180)

    finite = ricci_trace[np.isfinite(ricci_trace)]
    q = np.percentile(np.abs(finite), 99.0) if finite.size else 1.0
    q = max(float(q), 1e-12)
    norm = colors.SymLogNorm(
        linthresh=q * 1e-4, linscale=1.0, vmin=-q, vmax=q, base=10.0
    )

    fig2, ax2 = plt.subplots(figsize=(8, 7), constrained_layout=True)
    im2 = ax2.imshow(
        ricci_trace,
        extent=extent,
        origin="lower",
        cmap="RdBu_r",
        interpolation="nearest",
        norm=norm,
    )
    cbar2 = fig2.colorbar(im2, ax=ax2, fraction=0.046, pad=0.03)
    cbar2.set_label("Ricci trace")
    ax2.set_title("Ricci trace slice z=0 (64x64, Cartesian)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_aspect("equal")
    ax2.grid(True, color="black", linestyle="--", linewidth=0.3, alpha=0.2)
    fig2.savefig(args.ricci_png, dpi=180)

    print(f"alpha PNG: {args.alpha_png}")
    print(f"ricci PNG: {args.ricci_png}")


if __name__ == "__main__":
    main()
