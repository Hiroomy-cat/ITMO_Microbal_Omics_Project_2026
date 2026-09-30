#!/usr/bin/env python
"""Compare every inferred tree of one dataset with the reference (true) tree.
Output TSV columns: dataset, n_taxa, height, rep, method, model, nRF, nQD"""
import argparse
import os
import re

from common import normalized_quartet_distance, normalized_rf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--trees", nargs="+", required=True)
    ap.add_argument("--quartets", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    m = re.match(r"n(\d+)_h([\d.]+)_r(\d+)", a.dataset)
    n, h, r = (m.group(1), m.group(2), m.group(3)) if m else ("NA", "NA", "NA")
    ref = open(a.ref).read().strip()
    with open(a.out, "w") as fh:
        fh.write("dataset\tn_taxa\theight\trep\tmethod\tmodel\tnRF\tnQD\n")
        for t in a.trees:
            method, model = os.path.basename(t)[:-4].split("__")
            tree = open(t).read().strip()
            rf = normalized_rf(ref, tree)
            qd = normalized_quartet_distance(ref, tree) if a.quartets else float("nan")
            fh.write(f"{a.dataset}\t{n}\t{h}\t{r}\t{method}\t{model}\t{rf:.6f}\t{qd:.6f}\n")


if __name__ == "__main__":
    main()
