#!/usr/bin/env python
"""Distance matrix (square TSV) -> NJ tree (Newick). Logs the number of truncated negative branches."""
import sys

from common import neighbor_joining, read_dist

if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    names, D = read_dist(src)
    nwk, n_neg = neighbor_joining(names, D)
    with open(dst, "w") as fh:
        fh.write(nwk + "\n")
    if n_neg:
        print(f"[nj] {src}: {n_neg} negative branch length(s) truncated to 0", file=sys.stderr)
