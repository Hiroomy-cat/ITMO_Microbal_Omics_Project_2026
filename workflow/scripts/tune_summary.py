#!/usr/bin/env python
"""Average the PEA tuning grid over tuning datasets and pick the best setting PER MODEL.

Criterion: lowest mean nRF over all tuning datasets. Ties -> lowest mean nQD (computed only for
the tied settings) -> fewest changes from the current defaults -> first in sorted order.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from common import normalized_quartet_distance

KEYS = ["model", "layer", "center", "gap_open", "gap_extend", "zscore", "free_end_gaps", "gap_weight"]
PEA_KEYS = ["gap_open", "gap_extend", "zscore", "free_end_gaps", "gap_weight"]


def n_changes(row, defaults, last_layer):
    default_layer = last_layer if defaults["layer"] == -1 else defaults["layer"]
    n = int(row["center"] != False) + int(row["layer"] != default_layer)  # noqa: E712
    return n + sum(row[k] != defaults["pea"][k] for k in PEA_KEYS)


def mean_nqd(df, setting, ref_dir):
    sel = df.loc[(df[KEYS] == pd.Series(setting)).all(axis=1)]
    q = [normalized_quartet_distance(open(os.path.join(ref_dir, r.dataset, "true.nwk")).read().strip(),
                                     r.newick) for r in sel.itertuples()]
    return float(np.mean(q))


def pick(df, summ, model, defaults, ref_dir):
    s = summ[summ.model == model].copy()
    tied = s[np.isclose(s.nRF_mean, s.nRF_mean.min(), rtol=0, atol=1e-12)].copy()
    tied["nQD_mean"] = np.nan
    if len(tied) > 1:
        tied["nQD_mean"] = [mean_nqd(df, r[KEYS].to_dict(), ref_dir) for _, r in tied.iterrows()]
        tied = tied[np.isclose(tied.nQD_mean, tied.nQD_mean.min(), rtol=0, atol=1e-12)].copy()
    last = df.loc[df.model == model, "layer"].max()
    tied["n_changes"] = [n_changes(r, defaults, last) for _, r in tied.iterrows()]
    tied = tied[tied.n_changes == tied.n_changes.min()].sort_values(KEYS)
    best = tied.iloc[0].copy()
    best["n_tied_nRF"] = int(np.isclose(s.nRF_mean, s.nRF_mean.min(), rtol=0, atol=1e-12).sum())
    d = dict(defaults["pea"], center=False, layer=last if defaults["layer"] == -1 else defaults["layer"])
    ref_row = s.loc[(s[list(d)] == pd.Series(d)).all(axis=1)]
    best["nRF_mean_defaults"] = float(ref_row.nRF_mean.iloc[0]) if len(ref_row) else np.nan
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", nargs="+", required=True)
    ap.add_argument("--ref-dir", required=True, help="dir with <dataset>/true.nwk (tuning set)")
    ap.add_argument("--defaults", required=True, help='JSON {"pea": {...}, "layer": int}')
    ap.add_argument("--summary", required=True)
    ap.add_argument("--best", required=True)
    a = ap.parse_args()

    defaults = json.loads(a.defaults)
    df = pd.concat([pd.read_csv(f, sep="\t") for f in a.grid], ignore_index=True)
    g = df.groupby(KEYS)
    summ = g.nRF.agg(nRF_mean="mean", nRF_sd="std", n_datasets="count").reset_index()
    by_h = df.pivot_table(index=KEYS, columns="height", values="nRF", aggfunc="mean")
    by_h.columns = [f"nRF_h{h:g}" for h in by_h.columns]
    summ = summ.merge(by_h.reset_index(), on=KEYS)
    assert summ.n_datasets.nunique() == 1, "settings were evaluated on different numbers of datasets"
    summ = summ.sort_values(["model", "nRF_mean", *KEYS[1:]])
    summ["rank"] = summ.groupby("model").cumcount() + 1
    summ.to_csv(a.summary, sep="\t", index=False, float_format="%.6f")

    best = pd.DataFrame([pick(df, summ, m, defaults, a.ref_dir) for m in sorted(df.model.unique())])
    best.to_csv(a.best, sep="\t", index=False, float_format="%.6f")
    print(best.to_string(index=False))


if __name__ == "__main__":
    main()
