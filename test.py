#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_slice(df, field, out=None, levels=128):
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    v = df[field].to_numpy()

    fig, ax = plt.subplots(figsize=(7, 6))

    # Pour grille spectrale / Chebyshev, tricontourf marche bien même si les points
    # ne sont pas uniformément espacés.
    cntr = ax.tricontourf(x, y, v, levels=levels)

    cbar = fig.colorbar(cntr, ax=ax)
    cbar.set_label(field)

    # Points réels de collocation, utile pour voir la résolution réelle
    ax.scatter(x, y, s=8, alpha=0.35)

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Bowen-York puncture z-slice: {field}")

    if field == "residual":
        idx = df["residual"].abs().idxmax()
        row = df.loc[idx]
        ax.scatter([row["x"]], [row["y"]], marker="x", s=100)
        ax.text(
            row["x"],
            row["y"],
            f" max |res|\n{row['residual']:.3e}",
            fontsize=9,
            ha="left",
            va="bottom",
        )

    fig.tight_layout()

    if out:
        fig.savefig(out, dpi=250)
        print(f"wrote {out}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument(
        "--field",
        default="psi",
        choices=["psi", "u", "psi_singular", "residual", "r_puncture"],
    )
    parser.add_argument("--out", default=None)
    parser.add_argument("--levels", type=int, default=128)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    print("rows:", len(df))
    print("grid approx:", int(np.sqrt(len(df))), "x", int(np.sqrt(len(df))))
    print("z unique:", df["z"].unique())
    print(f"{args.field} min/max:", df[args.field].min(), df[args.field].max())

    plot_slice(df, args.field, out=args.out, levels=args.levels)


if __name__ == "__main__":
    main()
