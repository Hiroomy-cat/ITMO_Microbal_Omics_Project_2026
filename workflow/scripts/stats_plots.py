#!/usr/bin/env python
"""Hypothesis tests + publication-ready figures from results/summary.tsv.

Tests (paired over datasets, Wilcoxon signed-rank, Holm correction over all tests):
  H1   nRF(pea) < nRF(meanpool)            one-sided, per model x height
  ref  nRF(pea) vs nRF(msaresid_mafft)     two-sided (how close to the MSA-based upper bound)
  cls  nRF(pea) vs nRF(bionj), nRF(ml)     two-sided (vs classical pipelines)
  H1b  Spearman rho between tree height and [nRF(pea) - nRF(bionj)] per model
       (negative rho = PEA gains relative to MSA-based NJ as divergence grows)
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300, "pdf.fonttype": 42})

ORDER = ["ml", "bionj", "msaresid_true", "msaresid_mafft", "pea", "meanpool", "kmer"]
LABEL = {"ml": "IQ-TREE ML (MAFFT)", "bionj": "BIONJ, ML dist. (MAFFT)",
         "msaresid_true": "PLM residue dist., TRUE MSA", "msaresid_mafft": "PLM residue dist., MAFFT",
         "pea": "PEA: PLM pairwise alignment (no MSA)", "meanpool": "PLM mean-pool", "kmer": "3-mer cosine"}


def holm(p):
    p = np.asarray(p, float)
    ok = ~np.isnan(p)
    out = np.full_like(p, np.nan)
    idx = np.argsort(p[ok])
    m = ok.sum()
    adj = np.maximum.accumulate((m - np.arange(m)) * p[ok][idx])
    tmp = np.empty(m)
    tmp[idx] = np.minimum(adj, 1.0)
    out[ok] = tmp
    return out


def paired(df, a, b, model_a, model_b, metric):
    x = df[(df.method == a) & (df.model == model_a)].set_index("dataset")[metric]
    y = df[(df.method == b) & (df.model == model_b)].set_index("dataset")[metric]
    j = x.index.intersection(y.index)
    return x.loc[j], y.loc[j]


def run_tests(df, metric):
    rows = []
    models = sorted(m for m in df.model.unique() if m != "none")
    for mo in models:
        for h in sorted(df.height.unique()):
            d = df[df.height == h]
            for other, omodel, alt, tag in [("meanpool", mo, "less", "H1"),
                                            ("msaresid_mafft", mo, "two-sided", "ref"),
                                            ("bionj", "none", "two-sided", "cls"),
                                            ("ml", "none", "two-sided", "cls"),
                                            ("kmer", "none", "two-sided", "cls")]:
                x, y = paired(d, "pea", other, mo, omodel, metric)
                if len(x) < 5 or np.allclose(x.values, y.values):
                    p = np.nan
                else:
                    p = wilcoxon(x, y, alternative=alt).pvalue
                rows.append(dict(test=tag, model=mo, height=h, comparison=f"pea vs {other}",
                                 alternative=alt, n=len(x), median_pea=np.median(x) if len(x) else np.nan,
                                 median_other=np.median(y) if len(y) else np.nan,
                                 median_diff=np.median(x.values - y.values) if len(x) else np.nan, p=p))
        x, y = paired(df, "pea", "bionj", mo, "none", metric)
        if len(x) >= 5:
            hmap = df.drop_duplicates("dataset").set_index("dataset").height
            rho, p = spearmanr(hmap.loc[x.index].values, (x - y).values)
            rows.append(dict(test="H1b", model=mo, height="all", comparison="rho(height, pea - bionj)",
                             alternative="two-sided", n=len(x), median_pea=np.nan, median_other=np.nan,
                             median_diff=rho, p=p))
    res = pd.DataFrame(rows)
    res["p_holm"] = holm(res.p.values)
    return res


def boot_ci(v, n=2000, rng=np.random.default_rng(0)):
    v = np.asarray(v)
    if len(v) < 2:
        return np.nan, np.nan
    bs = rng.choice(v, (n, len(v))).mean(1)
    return np.percentile(bs, [2.5, 97.5])


# --- fig1 style: colour = model (validated blue/orange pair), PEA bright & thick, other PLM
# methods = muted tints of the model hue, classical methods = black/greys; method also gets its own
# line style + marker so identity never rests on colour alone. No colour is used twice.
MODEL_COLOR = {"esm2_t12_35M_UR50D": "#2a78d6", "onehot": "#eb6834"}
MODEL_TINTS = {"esm2_t12_35M_UR50D": {"msaresid_true": "#1f528f", "msaresid_mafft": "#74a6e3",
                                      "meanpool": "#9ec1ea"},
               "onehot": {"msaresid_true": "#9d4726", "msaresid_mafft": "#f19c7a", "meanpool": "#f4b9a1"}}
MODEL_LABEL = {"esm2_t12_35M_UR50D": "ESM-2 35M", "onehot": "one-hot"}
CLASSIC = {"ml": ("#0b0b0b", "-", "^"), "bionj": ("#52514e", "--", "v"), "kmer": ("#898781", ":", "P")}
METHOD_STYLE = {"pea": ("-", "o"), "msaresid_true": ("--", "s"), "msaresid_mafft": ("-.", "D"),
                "meanpool": (":", "X")}
SHORT = {"pea": "PEA (no MSA)", "meanpool": "mean-pool (no MSA)", "msaresid_true": "residue dist., TRUE MSA",
         "msaresid_mafft": "residue dist., MAFFT", "ml": "IQ-TREE ML (MAFFT)", "bionj": "BIONJ (MAFFT)",
         "kmer": "3-mer cosine (no MSA)"}
# legend columns (filled column-wise): MSA-free PLM | MSA-based PLM | classical
LEGEND_ORDER = [("pea", "esm2_t12_35M_UR50D"), ("pea", "onehot"), ("meanpool", "esm2_t12_35M_UR50D"),
                ("meanpool", "onehot"), ("msaresid_true", "esm2_t12_35M_UR50D"), ("msaresid_true", "onehot"),
                ("msaresid_mafft", "esm2_t12_35M_UR50D"), ("msaresid_mafft", "onehot"),
                ("ml", "none"), ("bionj", "none"), ("kmer", "none")]


def series_style(method, model):
    """(colour, linestyle, marker, linewidth, label, zorder) for one method x model series."""
    if model == "none":
        c, ls, mk = CLASSIC[method]
        return c, ls, mk, 1.3, SHORT[method], 3
    ls, mk = METHOD_STYLE[method]
    lab = f"{SHORT[method]}, {MODEL_LABEL.get(model, model)}"
    if method == "pea":
        return MODEL_COLOR[model], ls, mk, 2.8, lab, 5
    return MODEL_TINTS[model][method], ls, mk, 1.2, lab, 2


def fig_accuracy(df, metric, out):
    ns = sorted(df.n_taxa.unique())
    fig, axes = plt.subplots(1, len(ns), figsize=(3.6 * len(ns), 4.4), sharey=True, squeeze=False)
    present = set(zip(df.method, df.model))
    keys = [k for k in LEGEND_ORDER if k in present]
    for ax, n in zip(axes[0], ns):
        d = df[df.n_taxa == n]
        for me, mo in keys:
            g = d[(d.method == me) & (d.model == mo)].groupby("height")[metric]
            hs = np.array(sorted(g.groups))
            mu = np.array([g.get_group(h).mean() for h in hs])
            c, ls, mk, lw, lab, z = series_style(me, mo)
            ax.plot(hs, mu, ls=ls, marker=mk, ms=4 if me == "pea" else 3.5, lw=lw, color=c, label=lab,
                    zorder=z, markeredgecolor="white" if me == "pea" else c, markeredgewidth=0.6)
            if me == "pea":  # CI band only for the method under test, to keep the panel readable
                ci = np.array([boot_ci(g.get_group(h).values) for h in hs])
                ax.fill_between(hs, ci[:, 0], ci[:, 1], color=c, alpha=0.15, lw=0, zorder=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(d.height.unique()))
        ax.set_xticklabels([f"{h:g}" for h in sorted(d.height.unique())])
        ax.minorticks_off()
        ax.grid(axis="y", color="#e1e0d9", lw=0.6)
        ax.set_axisbelow(True)
        ax.set_title(f"{n} taxa")
    axes[0, 0].set_ylabel({"nRF": "Normalised RF distance to true tree",
                           "nQD": "Normalised quartet distance"}[metric])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if len(handles) == 11:  # 4 rows x 3 columns: pad the classical column
        handles.append(plt.Line2D([], [], alpha=0))
        labels.append("")
    # layout from the bottom up: legend, then the single shared x label, then the panels
    fig.subplots_adjust(left=0.08, right=0.99, top=0.92, bottom=0.34, wspace=0.08)
    fig.supxlabel("Mean root-to-tip distance (subs/site, log scale)", fontsize=9, y=0.245)
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, 0.0), columnspacing=1.5, handlelength=3.2)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight")
    plt.close(fig)


def fig_paired(df, metric, out):
    models = sorted(m for m in df.model.unique() if m != "none")
    fig, axes = plt.subplots(1, len(models), figsize=(3.2 * len(models), 2.8), sharey=True, squeeze=False)
    for ax, mo in zip(axes[0], models):
        hs = sorted(df.height.unique())
        data = []
        for h in hs:
            x, y = paired(df[df.height == h], "pea", "meanpool", mo, mo, metric)
            data.append((x - y).values)
        ax.boxplot(data, tick_labels=[str(h) for h in hs], showfliers=False)
        for i, v in enumerate(data, 1):
            ax.scatter(np.full(len(v), i) + np.random.default_rng(i).uniform(-.15, .15, len(v)), v, s=4, alpha=.4)
        ax.axhline(0, color="k", lw=.8)
        ax.set_title(mo)
        ax.set_xlabel("Tree height (subs/site)")
    axes[0, 0].set_ylabel(f"Δ{metric}: PEA − mean-pool\n(<0 = PEA better)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight")
    plt.close(fig)


# fig3: short point labels; offsets (points) chosen so labels in the two dense clusters do not overlap
SHORT_TAG = {"pea": "PEA", "meanpool": "MP", "msaresid_true": "TRUE", "msaresid_mafft": "MAFFT",
             "ml": "ML", "bionj": "BIONJ", "kmer": "3-mer"}
MODEL_TAG = {"esm2_t12_35M_UR50D": "ESM", "onehot": "1hot"}
TAG_OFFSET = {("pea", "onehot"): (-4, -16), ("msaresid_mafft", "onehot"): (10, 10),
              ("msaresid_true", "onehot"): (-30, 12), ("meanpool", "onehot"): (7, 3),
              ("pea", "esm2_t12_35M_UR50D"): (14, 14), ("msaresid_mafft", "esm2_t12_35M_UR50D"): (-44, 16),
              ("msaresid_true", "esm2_t12_35M_UR50D"): (8, -14), ("meanpool", "esm2_t12_35M_UR50D"): (7, 3),
              ("ml", "none"): (-30, -12), ("bionj", "none"): (-38, 4), ("kmer", "none"): (6, 5)}


def fig_runtime(df, metric, out):
    g = df.groupby(["method", "model"]).agg(acc=(metric, "mean"), sec=("seconds", "median")).reset_index()
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    for r in g.itertuples():
        c, _, mk, _, _, z = series_style(r.method, r.model)
        big = r.method == "pea"
        ax.scatter(r.sec, r.acc, s=60 if big else 32, marker=mk, color=c, edgecolor="white",
                   linewidth=0.7, zorder=3 if big else 4)  # big PEA marker below, so close points stay visible
        tag = SHORT_TAG[r.method] + ("" if r.model == "none" else "-" + MODEL_TAG.get(r.model, r.model))
        dx, dy = TAG_OFFSET.get((r.method, r.model), (6, 4))
        far = abs(dx) > 9 or abs(dy) > 9
        ax.annotate(tag, (r.sec, r.acc), xytext=(dx, dy), textcoords="offset points", fontsize=7,
                    fontweight="bold" if big else "normal", color="#0b0b0b" if big else "#52514e",
                    ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", color="#898781", lw=0.5, shrinkA=0, shrinkB=4) if far else None)
    ax.set_xscale("log")
    ax.set_ylim(min(0.16, g.acc.min() - 0.04), None)  # room for labels under the lowest points
    ax.grid(color="#e1e0d9", lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("Median wall-clock per dataset, s (log scale)")
    ax.set_ylabel(f"Mean {metric} (lower = better)")
    fig.text(0.01, -0.02, "PEA = pairwise embedding alignment, MP = mean-pool, TRUE / MAFFT = residue distance on "
             "the TRUE / MAFFT MSA;\nESM = ESM-2 35M, 1hot = one-hot. Colour = embedding model as in Fig. 1, "
             "black/grey = classical methods.", fontsize=6, color="#52514e", va="top")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--metric", default="nRF")
    ap.add_argument("--stats-out", required=True)
    ap.add_argument("--fig-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.fig_dir, exist_ok=True)
    df = pd.read_csv(a.summary, sep="\t")
    run_tests(df, a.metric).to_csv(a.stats_out, sep="\t", index=False)
    fig_accuracy(df, a.metric, os.path.join(a.fig_dir, "fig1_accuracy_vs_divergence"))
    fig_paired(df, a.metric, os.path.join(a.fig_dir, "fig2_pea_minus_meanpool"))
    fig_runtime(df, a.metric, os.path.join(a.fig_dir, "fig3_runtime_vs_accuracy"))


if __name__ == "__main__":
    main()
