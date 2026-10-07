#!/usr/bin/env python
"""PEA grid search on ONE tuning dataset and ONE embedding (TUNING grid only, never the test set).

For every grid point the result is exactly what the main pipeline would give
(dist_emb.py --method pea -> write_dist -> nj.py -> nRF): same pea_pair, same %.8g rounding of
the distance file, same NJ and metric. gap_weight does not change the alignment, so each pair is
aligned once per (center, gap_open, gap_extend, zscore, free_end_gaps) and all gap_weight values
are derived from it.

Output TSV: one row per grid point with nRF and the NJ tree (nQD is computed later, only where
it is needed as a tie-break, because quartets are O(n^4)).
"""
import argparse
import itertools
import json
import re

import numpy as np

from common import neighbor_joining, normalized_rf
from dist_emb import _unit, center, pea_pair

ALIGN_KEYS = ["gap_open", "gap_extend", "zscore", "free_end_gaps"]


def pea_raw(U, names, go, ge, zscore, free_end):
    """Mean (1 - cos) over matched residues and unmatched fraction, for every pair."""
    n = len(names)
    D0 = np.zeros((n, n))
    UNM = np.zeros((n, n))
    for i, j in itertools.combinations(range(n), 2):
        a, b = U[names[i]], U[names[j]]
        d, k = pea_pair(a, b, go=go, ge=ge, zscore=zscore, free_end=free_end, gap_weight=0.0)
        D0[i, j] = D0[j, i] = d
        UNM[i, j] = UNM[j, i] = 1.0 - 2.0 * k / (a.shape[0] + b.shape[0])
    return D0, UNM


def with_gap_weight(D0, UNM, gw):
    """Same arithmetic as pea_pair (d + gap_weight * unmatched) and the same rounding as write_dist."""
    D = D0 + gw * UNM
    np.fill_diagonal(D, 0.0)
    return np.vectorize(lambda x: float(f"{x:.8g}"))(D)


def grid_rows(emb, names, ref, grid):
    for c in grid["center"]:
        U = {k: _unit(v) for k, v in (center(emb, names) if c else emb).items()}
        for go, ge, z, fe in itertools.product(*(grid[k] for k in ALIGN_KEYS)):
            D0, UNM = pea_raw(U, names, go, ge, z, fe)
            for gw in grid["gap_weight"]:
                nwk, _ = neighbor_joining(names, with_gap_weight(D0, UNM, gw))
                yield dict(center=c, gap_open=go, gap_extend=ge, zscore=z, free_end_gaps=fe,
                           gap_weight=gw, nRF=normalized_rf(ref, nwk), newick=nwk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--ref", required=True, help="true tree of the tuning dataset")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--grid", required=True, help="JSON: lists for center, gap_open, gap_extend, "
                                                  "zscore, free_end_gaps, gap_weight")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    grid = json.loads(a.grid)
    emb = dict(np.load(a.emb))
    names = sorted(emb)
    ref = open(a.ref).read().strip()
    n, h, r = re.match(r"n(\d+)_h([\d.]+)_r(\d+)", a.dataset).groups()
    cols = ["dataset", "n_taxa", "height", "rep", "model", "layer", "center", *ALIGN_KEYS,
            "gap_weight", "nRF", "newick"]
    with open(a.out, "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for row in grid_rows(emb, names, ref, grid):
            row.update(dataset=a.dataset, n_taxa=n, height=h, rep=r, model=a.model, layer=a.layer)
            row["nRF"] = f"{row['nRF']:.6f}"
            fh.write("\t".join(str(row[c]) for c in cols) + "\n")


if __name__ == "__main__":
    main()
