#!/usr/bin/env python
"""Figure for the POST-HOC gap_weight = 0 ablation (descriptive): mean nRF with 95 % bootstrap CI
at the chosen heights for PEA (frozen), PEA with gap_weight = 0 and msaresid TRUE, per model.
Colour = model (same as fig1), marker = variant."""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stats_plots import MODEL_COLOR, MODEL_LABEL, MODEL_TINTS, boot_ci

ABL = "pea_gw0"
VARIANTS = [("pea", "PEA (frozen settings)"), (ABL, "PEA, gap_weight = 0"),
            ("msaresid_true", "residue dist., TRUE MSA")]
OFFSET = {"pea": -0.22, ABL: 0.0, "msaresid_true": 0.22}


def marker_kw(variant, model):
    c = MODEL_COLOR[model]
    if variant == "pea":
        return dict(marker="o", ms=7, color=c, mfc=c, mec="white", mew=0.8)
    if variant == ABL:
        return dict(marker="o", ms=7, color=c, mfc="white", mec=c, mew=1.6)
    t = MODEL_TINTS[model]["msaresid_true"]
    return dict(marker="s", ms=6, color=t, mfc=t, mec="white", mew=0.8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", nargs="+", required=True)
    ap.add_argument("--main", required=True)
    ap.add_argument("--heights", nargs="+", type=float, required=True)
    ap.add_argument("--out", required=True, help="output path without extension")
    a = ap.parse_args()

    abl = pd.concat([pd.read_csv(f, sep="\t") for f in a.metrics], ignore_index=True)
    main = pd.read_csv(a.main, sep="\t")
    df = pd.concat([abl, main[main.method.isin(["pea", "msaresid_true"])]], ignore_index=True)
    models = [m for m in MODEL_COLOR if m in set(abl.model)]

    fig, axes = plt.subplots(1, len(a.heights), figsize=(3.2 * len(a.heights), 3.4), sharey=True,
                             squeeze=False)
    for ax, h in zip(axes[0], a.heights):
        d = df[df.height == h]
        for i, mo in enumerate(models):
            for v, _ in VARIANTS:
                x = d[(d.method == v) & (d.model == mo)].nRF.values
                lo, hi = boot_ci(x)
                kw = marker_kw(v, mo)
                ax.errorbar(i + OFFSET[v], x.mean(), yerr=[[x.mean() - lo], [hi - x.mean()]], lw=0,
                            elinewidth=1.4, capsize=0, ecolor=kw["color"], **{k: kw[k] for k in kw if k != "color"})
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([MODEL_LABEL[m] for m in models])
        ax.set_xlim(-0.6, len(models) - 0.4)
        ax.grid(axis="y", color="#e1e0d9", lw=0.6)
        ax.set_axisbelow(True)
        ax.set_title(f"tree height {h:g} subs/site")
    axes[0, 0].set_ylabel("Mean normalised RF distance\n(95 % bootstrap CI)")
    fig.supxlabel("Embedding (colour = model; ESM-2 frozen gap_weight is already 0, so its two PEA "
                  "points coincide)", fontsize=7, y=0.13, color="#52514e")
    handles = [plt.Line2D([], [], ls="", **{k: v for k, v in marker_kw(v, "esm2_t12_35M_UR50D").items()
                                            if k != "color"}) for v, _ in VARIANTS]
    for hnd in handles:  # neutral legend glyphs: shape identifies the variant, colour the model
        hnd.set_markerfacecolor("#52514e" if hnd.get_markerfacecolor() != "white" else "white")
        hnd.set_markeredgecolor("#52514e" if hnd.get_markerfacecolor() == "white" else "white")
    fig.legend(handles, [lab for _, lab in VARIANTS], loc="lower center", ncol=3, fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.14, right=0.99, top=0.80, bottom=0.27, wspace=0.08)
    fig.suptitle("Post-hoc ablation: indel term of PEA (descriptive, not tuning)", fontsize=9, y=0.97)
    for ext in ("pdf", "png"):
        fig.savefig(f"{a.out}.{ext}", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
