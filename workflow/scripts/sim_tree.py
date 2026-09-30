#!/usr/bin/env python
"""Simulate a random Yule tree, add lineage-specific rate heterogeneity
(uncorrelated lognormal multipliers per branch) and rescale so that the MEAN
root-to-tip distance equals --height (substitutions / site).

Rate heterogeneity makes trees non-clock-like, which is realistic for protein
families and creates the long-branch situations where distance methods struggle.
"""
import argparse

import numpy as np


class Node:
    __slots__ = ("children", "length", "name")

    def __init__(self, name=None):
        self.children = []
        self.length = 0.0
        self.name = name


def yule_tree(n: int, rng: np.random.Generator) -> Node:
    root = Node()
    active = [Node(), Node()]
    root.children = active[:]
    while len(active) < n:
        dt = rng.exponential(1.0 / len(active))
        for nd in active:
            nd.length += dt
        parent = active.pop(rng.integers(len(active)))
        a, b = Node(), Node()
        parent.children = [a, b]
        active += [a, b]
    dt = rng.exponential(1.0 / len(active))
    for nd in active:
        nd.length += dt
    for i, nd in enumerate(active, 1):
        nd.name = f"t{i}"
    return root


def all_nodes(root):
    stack = [root]
    while stack:
        nd = stack.pop()
        yield nd
        stack.extend(nd.children)


def root_to_tip(root):
    out = []
    stack = [(root, 0.0)]
    while stack:
        nd, d = stack.pop()
        if not nd.children:
            out.append(d)
        for c in nd.children:
            stack.append((c, d + c.length))
    return np.array(out)


def to_newick(nd) -> str:
    if not nd.children:
        return f"{nd.name}:{nd.length:.8g}"
    inner = ",".join(to_newick(c) for c in nd.children)
    return f"({inner}):{nd.length:.8g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--height", type=float, required=True, help="mean root-to-tip, subs/site")
    ap.add_argument("--rate-sigma", type=float, default=0.3, help="lognormal sigma of per-branch rate multipliers")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    root = yule_tree(a.n, rng)
    for nd in all_nodes(root):
        if nd is not root:
            nd.length *= rng.lognormal(mean=0.0, sigma=a.rate_sigma)
    scale = a.height / root_to_tip(root).mean()
    for nd in all_nodes(root):
        nd.length *= scale
    # unrooted-style output: drop root length
    nwk = "(" + ",".join(to_newick(c) for c in root.children) + ");"
    with open(a.out, "w") as fh:
        fh.write(nwk + "\n")


if __name__ == "__main__":
    main()
