#!/usr/bin/env python
"""Evaluation of the EMPIRICAL trees of one protein family (no true tree exists).

Reference 1 — taxonomic monophyly: a group (NCBI phylum / class / order with >= 2 species in
the set and >= 2 outside) is recovered if its species form one side of a split of the unrooted
tree. Reference 2 — supported ML splits: fraction of reference splits with support >= threshold
that are missing from a tree. Robustness: random species subsets; trees are pruned, not rebuilt
(the splits of a pruned tree are exactly the restrictions of the original splits).

Outputs (TSV, all with a `family` column):
  groups.tsv      one row per group x tree: recovered 0/1
  monophyly.tsv   per tree: fraction recovered per rank and over all ranks (identical species
                  sets at several ranks counted once), with and without the dropped phylum
  supported.tsv   per tree: number of supported reference splits, number / fraction missing
  robustness.tsv  per tree: mean and 2.5-97.5 % interval of the all-ranks fraction over replicates
"""
import argparse
import os
import re

import numpy as np
import pandas as pd

_TOKEN = re.compile(r"\(|\)|,|:|;|\[[^\]]*\]|[^(),:;\[]+")


# ----------------------------------------------------------------------------- Newick -> splits

def parse(newick):
    """Return (leaf names, {split: support or None}); a split = frozenset of the side WITHOUT the
    alphabetically first leaf. Branch lengths are dropped; a label right after ')' is read as
    the support of that clade (IQ-TREE -B writes UFBoot there; for 'a/b' the last value)."""
    newick = re.sub(r":[-+0-9.eE]+", "", newick)
    stack, leaves, clades, after_close = [[]], [], [], False
    for tok in _TOKEN.findall(newick):
        if tok == "(":
            stack.append([])
            after_close = False
        elif tok == ")":
            below = stack.pop()
            clades.append([frozenset(below), None])
            stack[-1].extend(below)
            after_close = True
        elif tok in (",", ":", ";") or tok.startswith("["):
            after_close = False
        else:
            tok = tok.strip()
            if not tok:
                continue
            if after_close:
                try:
                    clades[-1][1] = float(tok.split("/")[-1])
                except ValueError:
                    pass
                after_close = False
            else:
                leaves.append(tok)
                stack[-1].append(tok)
    all_leaves = frozenset(leaves)
    root = min(all_leaves)
    out = {}
    for side, sup in clades:
        s = side if root not in side else all_leaves - side
        if 2 <= len(s) <= len(all_leaves) - 2 and out.get(s) is None:
            out[s] = sup
    return sorted(leaves), out


def restrict(splits, subset):
    """Splits of the tree pruned to `subset` (normalised the same way, trivial ones dropped)."""
    sub = frozenset(subset)
    root = min(sub)
    out = set()
    for s in splits:
        a = s & sub
        side = a if root not in a else sub - a
        if 2 <= len(side) <= len(sub) - 2:
            out.add(side)
    return out


def recovered(group, splits, taxa):
    """Is `group` one side of a split (unrooted monophyly)? Splits must be normalised on `taxa`."""
    g = frozenset(group)
    root = min(taxa)
    side = g if root not in g else frozenset(taxa) - g
    return side in splits


# ----------------------------------------------------------------------------- groups

def eligible_groups(sp, taxa, ranks):
    """[(rank, name, frozenset(taxa))] with >= 2 inside and >= 2 outside `taxa`."""
    taxa = set(taxa)
    out = []
    for rank in ranks:
        for name, g in sp[sp.taxon.isin(taxa)].groupby(rank):
            members = frozenset(g.taxon)
            if name != "NA" and len(members) >= 2 and len(taxa - members) >= 2:
                out.append((rank, name, members))
    return out


def fraction(groups, splits, taxa, drop_phylum_taxa=frozenset()):
    """All-ranks fraction recovered; identical species sets counted once."""
    sets = {m for _, _, m in groups if not (m <= drop_phylum_taxa)}
    if not sets:
        return np.nan
    return np.mean([recovered(m, splits, taxa) for m in sets])


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--species", required=True)
    ap.add_argument("--ref", required=True, help="reference ML tree with UFBoot labels")
    ap.add_argument("--trees", nargs="+", required=True, help="<method>__<model>.nwk")
    ap.add_argument("--ranks", nargs="+", default=["phylum", "class", "order"])
    ap.add_argument("--min-support", type=float, default=95)
    ap.add_argument("--drop-phylum", default="Chlamydiota")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--per-phylum", type=int, default=4)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()

    sp = pd.read_csv(a.species, sep="\t", comment="#")
    taxa = sorted(sp.taxon)
    drop = frozenset(sp.loc[sp.phylum == a.drop_phylum, "taxon"])
    groups = eligible_groups(sp, taxa, a.ranks)

    trees = {"ml_ref__none": open(a.ref).read().strip()}
    for t in a.trees:
        trees[os.path.basename(t)[:-4]] = open(t).read().strip()
    parsed = {}
    for key, nwk in trees.items():
        leaves, spl = parse(nwk)
        assert sorted(leaves) == taxa, f"{key}: taxa differ from species.tsv"
        parsed[key] = spl

    ref_sup = {s for s, v in parsed["ml_ref__none"].items() if v is not None and v >= a.min_support}

    g_rows, m_rows, s_rows = [], [], []
    for key, spl in parsed.items():
        method, model = key.split("__")
        splits = set(spl)
        for rank, name, members in groups:
            g_rows.append(dict(family=a.family, method=method, model=model, rank=rank, group=name,
                               n_species=len(members), recovered=int(recovered(members, splits, taxa))))
        for scope, dropped in (("all", frozenset()), (f"without_{a.drop_phylum}", drop)):
            for rank in [*a.ranks, "all_ranks"]:
                gs = groups if rank == "all_ranks" else [g for g in groups if g[0] == rank]
                gs = [g for g in gs if not (g[2] <= dropped)]
                n_sets = len({m for _, _, m in gs})
                m_rows.append(dict(family=a.family, method=method, model=model, scope=scope, rank=rank,
                                   n_groups=n_sets, fraction=fraction(gs, splits, taxa)))
        if key != "ml_ref__none":
            miss = len(ref_sup - splits)
            s_rows.append(dict(family=a.family, method=method, model=model, n_supported=len(ref_sup),
                               n_missing=miss, frac_missing=miss / len(ref_sup) if ref_sup else np.nan))

    # robustness: identical random subsets for every tree
    rng = np.random.default_rng(a.seed)
    by_phylum = {p: sorted(g.taxon) for p, g in sp.groupby("phylum")}
    subsets = [sorted(t for p in sorted(by_phylum) for t in rng.choice(by_phylum[p], a.per_phylum,
                                                                        replace=False))
               for _ in range(a.reps)]
    r_rows = []
    for key, spl in parsed.items():
        method, model = key.split("__")
        for scope, dropped in (("all", frozenset()), (f"without_{a.drop_phylum}", drop)):
            vals = []
            for sub in subsets:
                rs = restrict(spl, sub)
                gs = eligible_groups(sp, sub, a.ranks)
                vals.append(fraction(gs, rs, sub, dropped & frozenset(sub)))
            v = np.array(vals, dtype=float)
            r_rows.append(dict(family=a.family, method=method, model=model, scope=scope,
                               reps=len(v), mean=np.nanmean(v), lo=np.nanpercentile(v, 2.5),
                               hi=np.nanpercentile(v, 97.5)))

    os.makedirs(a.outdir, exist_ok=True)
    for name, rows in (("groups", g_rows), ("monophyly", m_rows), ("supported", s_rows),
                       ("robustness", r_rows)):
        pd.DataFrame(rows).to_csv(os.path.join(a.outdir, f"{name}.tsv"), sep="\t", index=False,
                                  float_format="%.4f")


if __name__ == "__main__":
    main()
