#!/usr/bin/env python
"""Pairwise distance matrices from per-residue PLM embeddings.

Methods
  meanpool   sequence = mean of residue embeddings; distance = cosine (or euclidean).
             Negative baseline (Stakauskas & Gorecki 2026: signal is lost by pooling).
  msaresid   residues are matched through a multiple sequence alignment (MAFFT or the
             TRUE simulated one); distance = mean (1 - cos) over columns where both
             sequences have a residue. Reference / upper bound: it needs an MSA.
  pea        *** the method under test *** Pairwise Embedding Alignment, NO MSA.
             For every pair: residue-residue cosine similarity matrix -> (optionally
             z-score "signal enhancement", as in EBA, Pantolini et al. 2024) -> affine-gap
             global/semi-global DP alignment -> distance = mean (1 - cos) over matched
             residue pairs + gap_weight * unmatched fraction.
"""
import argparse
import itertools

import numpy as np

from common import read_fasta, write_dist

try:
    from numba import njit
except ImportError:  # pure-python fallback (slow, but correct)
    def njit(*a, **k):
        def deco(f):
            return f
        return deco if not (a and callable(a[0])) else a[0]


def _unit(E):
    E = E.astype(np.float32)
    n = np.linalg.norm(E, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return E / n


# ----------------------------------------------------------------------------- meanpool

def dist_meanpool(emb, names, metric="cosine"):
    V = np.stack([emb[k].astype(np.float32).mean(axis=0) for k in names])
    if metric == "euclidean":
        diff = V[:, None, :] - V[None, :, :]
        return np.sqrt((diff ** 2).sum(-1))
    U = _unit(V)
    return 1.0 - U @ U.T


# ----------------------------------------------------------------------------- msaresid

def dist_msaresid(emb, names, aln):
    n = len(names)
    cols = len(next(iter(aln.values())))
    D0 = next(iter(emb.values())).shape[1]
    X = np.zeros((n, cols, D0), dtype=np.float32)
    mask = np.zeros((n, cols), dtype=bool)
    for a, k in enumerate(names):
        U = _unit(emb[k])
        r = 0
        for c, ch in enumerate(aln[k]):
            if ch not in "-.":
                X[a, c] = U[r]
                mask[a, c] = True
                r += 1
        assert r == U.shape[0], f"{k}: alignment has {r} residues, embedding {U.shape[0]}"
    D = np.zeros((n, n))
    for i, j in itertools.combinations(range(n), 2):
        m = mask[i] & mask[j]
        if m.sum() == 0:
            D[i, j] = D[j, i] = np.nan
            continue
        d = 1.0 - (X[i, m] * X[j, m]).sum(-1).mean()
        D[i, j] = D[j, i] = d
    return D


# ----------------------------------------------------------------------------- PEA

@njit(cache=True)
def _gotoh(S, go, ge, free_end):
    """Affine-gap alignment maximising sum(S). Returns matched index arrays (ia, jb)."""
    L, M = S.shape
    NEG = -1e18
    H = np.full((L + 1, M + 1), NEG)
    E = np.full((L + 1, M + 1), NEG)  # consumes a residue of seq A (gap in B)
    F = np.full((L + 1, M + 1), NEG)  # consumes a residue of seq B (gap in A)
    TH = np.zeros((L + 1, M + 1), np.uint8)
    TE = np.zeros((L + 1, M + 1), np.uint8)
    TF = np.zeros((L + 1, M + 1), np.uint8)
    H[0, 0] = 0.0
    for i in range(1, L + 1):
        E[i, 0] = 0.0 if free_end else -(go + (i - 1) * ge)
        TE[i, 0] = 1 if i > 1 else 0
    for j in range(1, M + 1):
        F[0, j] = 0.0 if free_end else -(go + (j - 1) * ge)
        TF[0, j] = 2 if j > 1 else 0
    for i in range(1, L + 1):
        for j in range(1, M + 1):
            # match
            a, b, c = H[i - 1, j - 1], E[i - 1, j - 1], F[i - 1, j - 1]
            if a >= b and a >= c:
                H[i, j] = a + S[i - 1, j - 1]; TH[i, j] = 0
            elif b >= c:
                H[i, j] = b + S[i - 1, j - 1]; TH[i, j] = 1
            else:
                H[i, j] = c + S[i - 1, j - 1]; TH[i, j] = 2
            # gap in B (move i)
            a, b, c = H[i - 1, j] - go, E[i - 1, j] - ge, F[i - 1, j] - go
            if a >= b and a >= c:
                E[i, j] = a; TE[i, j] = 0
            elif b >= c:
                E[i, j] = b; TE[i, j] = 1
            else:
                E[i, j] = c; TE[i, j] = 2
            # gap in A (move j)
            a, b, c = H[i, j - 1] - go, F[i, j - 1] - ge, E[i, j - 1] - go
            if a >= b and a >= c:
                F[i, j] = a; TF[i, j] = 0
            elif b >= c:
                F[i, j] = b; TF[i, j] = 2
            else:
                F[i, j] = c; TF[i, j] = 1
    # choose end cell
    if free_end:
        bi, bj, best = L, M, H[L, M]
        for j in range(1, M + 1):
            if H[L, j] > best:
                best, bi, bj = H[L, j], L, j
        for i in range(1, L + 1):
            if H[i, M] > best:
                best, bi, bj = H[i, M], i, M
        s = 0
    else:
        bi, bj = L, M
        a, b, c = H[L, M], E[L, M], F[L, M]
        s = 0 if (a >= b and a >= c) else (1 if b >= c else 2)
    ia = np.empty(min(L, M), np.int64)
    jb = np.empty(min(L, M), np.int64)
    k = 0
    i, j = bi, bj
    while i > 0 and j > 0:
        if s == 0:
            ia[k] = i - 1; jb[k] = j - 1; k += 1
            s = TH[i, j]; i -= 1; j -= 1
        elif s == 1:
            s = TE[i, j]; i -= 1
        else:
            s = TF[i, j]; j -= 1
    return ia[:k][::-1].copy(), jb[:k][::-1].copy()


def _zscore_enhance(S):
    """EBA-style signal enhancement: average of row- and column-wise z-scores."""
    zr = (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-8)
    zc = (S - S.mean(0, keepdims=True)) / (S.std(0, keepdims=True) + 1e-8)
    return 0.5 * (zr + zc)


def pea_pair(Ua, Ub, go=2.0, ge=0.5, zscore=True, free_end=True, gap_weight=0.0):
    C = (Ua @ Ub.T).astype(np.float64)  # raw cosine similarities
    S = _zscore_enhance(C) if zscore else C
    ia, jb = _gotoh(S, go, ge, free_end)
    if len(ia) == 0:
        return np.nan, 0
    d = float((1.0 - C[ia, jb]).mean())
    unmatched = 1.0 - 2.0 * len(ia) / (Ua.shape[0] + Ub.shape[0])
    return d + gap_weight * unmatched, len(ia)


def dist_pea(emb, names, **kw):
    U = {k: _unit(emb[k]) for k in names}
    n = len(names)
    D = np.zeros((n, n))
    for i, j in itertools.combinations(range(n), 2):
        d, _ = pea_pair(U[names[i]], U[names[j]], **kw)
        D[i, j] = D[j, i] = d
    return D


# ----------------------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True, help=".npz from embed.py")
    ap.add_argument("--method", required=True, choices=["meanpool", "msaresid", "pea"])
    ap.add_argument("--aln", help="aligned FASTA (msaresid only)")
    ap.add_argument("--metric", default="cosine", choices=["cosine", "euclidean"])
    ap.add_argument("--gap-open", type=float, default=2.0)
    ap.add_argument("--gap-extend", type=float, default=0.5)
    ap.add_argument("--no-zscore", action="store_true")
    ap.add_argument("--global-ends", action="store_true", help="penalise terminal gaps")
    ap.add_argument("--gap-weight", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    emb = dict(np.load(a.emb))
    names = sorted(emb)
    if a.method == "meanpool":
        D = dist_meanpool(emb, names, a.metric)
    elif a.method == "msaresid":
        D = dist_msaresid(emb, names, read_fasta(a.aln))
    else:
        D = dist_pea(emb, names, go=a.gap_open, ge=a.gap_extend, zscore=not a.no_zscore,
                     free_end=not a.global_ends, gap_weight=a.gap_weight)
    write_dist(names, D, a.out)


if __name__ == "__main__":
    main()
