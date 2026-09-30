"""Shared helpers: FASTA I/O, distance-matrix I/O, neighbour joining,
Newick parsing and split-based tree metrics.

Kept dependency-light on purpose (numpy only) so every step is easy to audit.
"""
from __future__ import annotations

import itertools
import re
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------------- FASTA


def read_fasta(path: str) -> Dict[str, str]:
    seqs: Dict[str, List[str]] = {}
    name = None
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                name = line[1:].split()[0]
                seqs[name] = []
            else:
                seqs[name].append(line)
    return {k: "".join(v) for k, v in seqs.items()}


def write_fasta(seqs: Dict[str, str], path: str) -> None:
    with open(path, "w") as fh:
        for k, v in seqs.items():
            fh.write(f">{k}\n{v}\n")


GAP_CHARS = set("-.")


def degap(seq: str) -> str:
    return "".join(c for c in seq if c not in GAP_CHARS)


# ----------------------------------------------------------------------------- distance matrices


def write_dist(names: List[str], D: np.ndarray, path: str) -> None:
    """Square TSV with a header row; first column = names."""
    with open(path, "w") as fh:
        fh.write("\t" + "\t".join(names) + "\n")
        for n, row in zip(names, D):
            fh.write(n + "\t" + "\t".join(f"{x:.8g}" for x in row) + "\n")


def read_dist(path: str) -> Tuple[List[str], np.ndarray]:
    with open(path) as fh:
        names = fh.readline().rstrip("\n").split("\t")[1:]
        rows = [list(map(float, l.rstrip("\n").split("\t")[1:])) for l in fh if l.strip()]
    return names, np.asarray(rows, dtype=float)


def sanitize_dist(D: np.ndarray) -> np.ndarray:
    """Symmetrise, zero the diagonal, replace NaN/inf by 1.5 * max finite value."""
    D = np.array(D, dtype=float)
    finite = D[np.isfinite(D)]
    fill = (finite.max() * 1.5) if finite.size else 1.0
    D[~np.isfinite(D)] = fill
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)
    D[D < 0] = 0.0
    return D


# ----------------------------------------------------------------------------- neighbour joining


def neighbor_joining(names: List[str], D: np.ndarray) -> Tuple[str, int]:
    """Classic Saitou & Nei NJ. Returns (unrooted Newick, number of negative
    branch lengths that were truncated to 0)."""
    D = sanitize_dist(D)
    nodes = list(names)
    n_neg = 0
    while len(nodes) > 3:
        n = len(nodes)
        r = D.sum(axis=1)
        Q = (n - 2) * D - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        i, j = np.unravel_index(np.argmin(Q), Q.shape)
        if i > j:
            i, j = j, i
        li = 0.5 * D[i, j] + (r[i] - r[j]) / (2 * (n - 2))
        lj = D[i, j] - li
        n_neg += int(li < 0) + int(lj < 0)
        li, lj = max(li, 0.0), max(lj, 0.0)
        new = f"({nodes[i]}:{li:.8g},{nodes[j]}:{lj:.8g})"
        du = 0.5 * (D[i] + D[j] - D[i, j])
        keep = [k for k in range(n) if k not in (i, j)]
        D2 = np.zeros((n - 1, n - 1))
        D2[:-1, :-1] = D[np.ix_(keep, keep)]
        D2[-1, :-1] = du[keep]
        D2[:-1, -1] = du[keep]
        D = D2
        nodes = [nodes[k] for k in keep] + [new]
    if len(nodes) == 3:
        la = 0.5 * (D[0, 1] + D[0, 2] - D[1, 2])
        lb = D[0, 1] - la
        lc = D[0, 2] - la
        n_neg += sum(x < 0 for x in (la, lb, lc))
        la, lb, lc = (max(x, 0.0) for x in (la, lb, lc))
        return f"({nodes[0]}:{la:.8g},{nodes[1]}:{lb:.8g},{nodes[2]}:{lc:.8g});", n_neg
    if len(nodes) == 2:
        return f"({nodes[0]}:{D[0, 1] / 2:.8g},{nodes[1]}:{D[0, 1] / 2:.8g});", n_neg
    return f"({nodes[0]});", n_neg


# ----------------------------------------------------------------------------- Newick -> splits

_TOKEN = re.compile(r"\(|\)|,|:|;|\[[^\]]*\]|[^(),:;\[]+")


def parse_newick_leafsets(newick: str) -> Tuple[List[str], List[frozenset]]:
    """Return (leaf names, leaf sets below every internal node).
    Handles branch lengths, internal labels / support values and [comments]."""
    tokens = _TOKEN.findall(newick.strip())
    leaves: List[str] = []
    clades: List[frozenset] = []
    stack: List[set] = []
    expect_len = False
    last_closed = False
    for tok in tokens:
        t = tok.strip()
        if tok.startswith("["):
            continue
        if tok == "(":
            stack.append(set())
            last_closed = False
        elif tok == ",":
            expect_len = last_closed = False
        elif tok == ")":
            s = stack.pop()
            clades.append(frozenset(s))
            if stack:
                stack[-1] |= s
            last_closed, expect_len = True, False
        elif tok == ":":
            expect_len = True
        elif tok == ";":
            break
        elif t:
            if expect_len:
                expect_len = False
                continue
            if last_closed:  # internal node label or support value
                last_closed = False
                continue
            leaves.append(t)
            if stack:
                stack[-1].add(t)
    return leaves, clades


def splits_bitmask(newick: str, taxa: List[str]) -> set:
    """Non-trivial bipartitions as canonical int bitmasks over `taxa` order."""
    leaves, clades = parse_newick_leafsets(newick)
    if set(leaves) != set(taxa):
        raise ValueError(f"Tree taxa differ from reference taxa: {sorted(set(leaves) ^ set(taxa))[:5]}")
    idx = {t: i for i, t in enumerate(taxa)}
    n = len(taxa)
    full = (1 << n) - 1
    out = set()
    for c in clades:
        if len(c) < 2 or len(c) > n - 2:
            continue
        m = 0
        for t in c:
            m |= 1 << idx[t]
        if m & 1:  # canonical form: the side NOT containing taxon 0
            m = full ^ m
        out.add(m)
    return out


def normalized_rf(tree_a: str, tree_b: str) -> float:
    """Robinson–Foulds distance normalised by the total number of non-trivial splits."""
    taxa = sorted(parse_newick_leafsets(tree_a)[0])
    A, B = splits_bitmask(tree_a, taxa), splits_bitmask(tree_b, taxa)
    denom = len(A) + len(B)
    return len(A ^ B) / denom if denom else 0.0


def _quartet_topology(splits: set, a: int, b: int, c: int, d: int):
    """0 = ab|cd, 1 = ac|bd, 2 = ad|bc, None = unresolved."""
    ba, bb, bc, bd = 1 << a, 1 << b, 1 << c, 1 << d
    for s in splits:
        ia, ib, ic, idd = bool(s & ba), bool(s & bb), bool(s & bc), bool(s & bd)
        if ia == ib and ic == idd and ia != ic:
            return 0
        if ia == ic and ib == idd and ia != ib:
            return 1
        if ia == idd and ib == ic and ia != ib:
            return 2
    return None


def normalized_quartet_distance(tree_a: str, tree_b: str) -> float:
    """Fraction of 4-taxon subsets whose topology differs.
    O(n^4 * splits): fine for n <= ~40; use TreeDist/tqDist for bigger trees."""
    taxa = sorted(parse_newick_leafsets(tree_a)[0])
    A, B = splits_bitmask(tree_a, taxa), splits_bitmask(tree_b, taxa)
    diff = tot = 0
    for q in itertools.combinations(range(len(taxa)), 4):
        tot += 1
        if _quartet_topology(A, *q) != _quartet_topology(B, *q):
            diff += 1
    return diff / tot if tot else 0.0
