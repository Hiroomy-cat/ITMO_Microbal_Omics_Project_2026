#!/usr/bin/env python
"""POST-HOC ablation report (descriptive only, no hypothesis tests): PEA with gap_weight = 0
vs the main PEA and msaresid TRUE, paired over the same test datasets, per model and height.

Columns: mean nRF of each variant, paired mean difference with a 95 % percentile bootstrap CI
(fixed seed), and how many datasets got better / equal / worse.
"""
import argparse

import numpy as np
import pandas as pd

ABL = "pea_gw0"


def boot_ci(d, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.choice(d, size=(n, len(d)), replace=True).mean(axis=1)
    return np.percentile(m, [2.5, 97.5])


def paired(a, b):
    """a - b over the same datasets (negative = a has lower nRF = better)."""
    d = (a - b).to_numpy()
    lo, hi = boot_ci(d)
    return dict(mean=d.mean(), lo=lo, hi=hi, better=int((d < 0).sum()), equal=int((d == 0).sum()),
                worse=int((d > 0).sum()), max_abs=float(np.abs(d).max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", nargs="+", required=True, help="results_ablation/metrics/*.tsv")
    ap.add_argument("--main", required=True, help="results/summary.tsv (read-only)")
    ap.add_argument("--heights", nargs="+", type=float, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    abl = pd.concat([pd.read_csv(f, sep="\t") for f in a.metrics], ignore_index=True)
    main = pd.read_csv(a.main, sep="\t")
    df = pd.concat([abl, main[main.method.isin(["pea", "msaresid_true"])]], ignore_index=True)
    w = df.pivot_table(index=["dataset", "height", "model"], columns="method", values="nRF").reset_index()
    assert w[[ABL, "pea", "msaresid_true"]].notna().all().all(), "missing trees for some datasets"

    rows = []
    for (model, h), g in w[w.height.isin(a.heights)].groupby(["model", "height"]):
        r = dict(model=model, height=h, n_datasets=len(g), nRF_pea=g.pea.mean(),
                 nRF_pea_gw0=g[ABL].mean(), nRF_msaresid_true=g.msaresid_true.mean())
        for name, other in [("gw0_minus_pea", "pea"), ("gw0_minus_msaresid_true", "msaresid_true"),
                            ("pea_minus_msaresid_true", None)]:
            p = paired(g[ABL] if other else g.pea, g[other] if other else g.msaresid_true)
            r.update({f"{name}_{k}": v for k, v in p.items()})
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(a.out, sep="\t", index=False, float_format="%.4f")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
