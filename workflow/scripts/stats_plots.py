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


def fig_accuracy(df, metric, out):
    ns = sorted(df.n_taxa.unique())
    fig, axes = plt.subplots(1, len(ns), figsize=(3.4 * len(ns), 3.0), sharey=True, squeeze=False)
    cmap = plt.get_cmap("tab10")
    mm = [m for m in df.method_model.unique()]
    mm.sort(key=lambda s: (ORDER.index(s.split("__")[0]) if s.split("__")[0] in ORDER else 99, s))
    for ax, n in zip(axes[0], ns):
        d = df[df.n_taxa == n]
        for k, key in enumerate(mm):
            g = d[d.method_model == key].groupby("height")[metric]
            hs = np.array(sorted(g.groups))
            mu = np.array([g.get_group(h).mean() for h in hs])
            ci = np.array([boot_ci(g.get_group(h).values) for h in hs])
            me, mo = key.split("__")
            lab = LABEL.get(me, me) + ("" if mo == "none" else f" [{mo}]")
            ls = "-" if me == "pea" else ("--" if mo == "none" else ":")
            lw = 2.2 if me == "pea" else 1.2
            ax.plot(hs, mu, ls, marker="o", ms=3, lw=lw, color=cmap(k % 10), label=lab)
            ax.fill_between(hs, ci[:, 0], ci[:, 1], color=cmap(k % 10), alpha=0.12, lw=0)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Mean root-to-tip distance (subs/site)")
        ax.set_title(f"{n} taxa")
    axes[0, 0].set_ylabel({"nRF": "Normalised RF distance to true tree",
                           "nQD": "Normalised quartet distance"}[metric])
    axes[0, -1].legend(fontsize=7, frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
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


def fig_runtime(df, metric, out):
    g = df.groupby("method_model").agg(acc=(metric, "mean"), sec=("seconds", "median")).reset_index()
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.scatter(g.sec, g.acc)
    for _, r in g.iterrows():
        ax.annotate(r.method_model.replace("__none", ""), (r.sec, r.acc), fontsize=6, xytext=(3, 3),
                    textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Median wall-clock per dataset, s (log)")
    ax.set_ylabel(f"Mean {metric}")
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
