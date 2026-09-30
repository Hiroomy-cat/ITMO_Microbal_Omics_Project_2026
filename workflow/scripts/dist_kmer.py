#!/usr/bin/env python
"""Alignment-free k-mer baseline: amino-acid k-mer count vectors -> cosine distance.
(The simplest competitor without alignment and without a language model.
 Possible upgrade: CVTree-style Markov background subtraction.)"""
import argparse
from collections import Counter

import numpy as np

from common import read_fasta, write_dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    seqs = read_fasta(a.fasta)
    names = sorted(seqs)
    counts = [Counter(seqs[n][i:i + a.k] for i in range(len(seqs[n]) - a.k + 1)) for n in names]
    vocab = sorted(set().union(*counts))
    vi = {w: i for i, w in enumerate(vocab)}
    X = np.zeros((len(names), len(vocab)), dtype=np.float64)
    for r, c in enumerate(counts):
        for w, v in c.items():
            X[r, vi[w]] = v
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    write_dist(names, 1.0 - X @ X.T, a.out)


if __name__ == "__main__":
    main()
