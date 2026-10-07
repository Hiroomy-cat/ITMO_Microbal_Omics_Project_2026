"""Fast unit tests for the core logic. Run: pytest -q tests/"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "workflow", "scripts"))

from common import neighbor_joining, normalized_quartet_distance, normalized_rf  # noqa: E402
from dist_emb import _unit, pea_pair  # noqa: E402
from embed import embed_onehot  # noqa: E402

TRUE = "((a:1,b:2):1,(c:1,d:3):2,(e:1,f:1):1);"




def test_nj_recovers_additive_tree():
    names = list("abcdef")
    # path lengths in TRUE
    leaf = {"a": 1, "b": 2, "c": 1, "d": 3, "e": 1, "f": 1}
    grp = {"a": ("x", 1), "b": ("x", 1), "c": ("y", 2), "d": ("y", 2), "e": ("z", 1), "f": ("z", 1)}
    D = np.zeros((6, 6))
    for i, p in enumerate(names):
        for j, q in enumerate(names):
            if i == j:
                continue
            if grp[p][0] == grp[q][0]:
                D[i, j] = leaf[p] + leaf[q]
            else:
                D[i, j] = leaf[p] + leaf[q] + grp[p][1] + grp[q][1]
    nwk, n_neg = neighbor_joining(names, D)
    assert n_neg == 0
    assert normalized_rf(TRUE, nwk) == 0.0


def test_rf_and_quartets():
    other = "((a:1,c:1):1,(b:1,d:1):1,(e:1,f:1):1);"
    assert normalized_rf(TRUE, TRUE) == 0.0
    assert normalized_quartet_distance(TRUE, TRUE) == 0.0
    assert 0 < normalized_rf(TRUE, other) <= 1
    assert 0 < normalized_quartet_distance(TRUE, other) <= 1


def test_pea_identity_and_indel():
    s1 = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWAAHP"
    s2 = s1[:50] + "GGGGG" + s1[50:]  # 5-aa insertion
    E = embed_onehot({"x": s1, "y": s2})
    U1, U2 = _unit(E["x"]), _unit(E["y"])
    d_same, m_same = pea_pair(U1, U1)
    assert d_same < 1e-6 and m_same == len(s1)
    d_ins, m_ins = pea_pair(U1, U2)
    assert d_ins < 0.02, d_ins            # insertion must be absorbed by a gap, not by mismatches
    assert m_ins >= len(s1) - 2


def test_center_removes_dataset_mean():
    from dist_emb import center
    rng = np.random.default_rng(0)
    emb = {"a": rng.normal(5, 1, (30, 8)), "b": rng.normal(5, 1, (40, 8))}
    C = center(emb, ["a", "b"])
    assert np.allclose(np.concatenate([C["a"], C["b"]]).mean(0), 0, atol=1e-5)
    assert C["a"].shape == (30, 8) and C["b"].shape == (40, 8)


def test_tune_pea_matches_pipeline(tmp_path):
    """One grid point of tune_pea.py == dist_emb.py --method pea -> write_dist -> nj.py -> nRF."""
    from common import read_dist, write_dist
    from dist_emb import center, dist_pea
    from tune_pea import grid_rows

    rng = np.random.default_rng(1)
    base = "".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), 60))
    seqs = {}
    for i in range(6):
        s = list(base)
        for p in rng.choice(60, 6 + 3 * i, replace=False):
            s[p] = rng.choice(list("ACDEFGHIKLMNPQRSTVWY"))
        seqs[f"t{i + 1}"] = "".join(s)[: 55 + i]
    emb = embed_onehot(seqs)
    names = sorted(emb)
    ref = "((t1,t2),(t3,t4),(t5,t6));"
    grid = dict(center=[True, False], gap_open=[1.0, 4.0], gap_extend=[0.5], zscore=[True, False],
                free_end_gaps=[False], gap_weight=[0.0, 1.0])
    rows = list(grid_rows(emb, names, ref, grid))
    assert len(rows) == 16
    for r in rows:
        e = center(emb, names) if r["center"] else emb
        D = dist_pea(e, names, go=r["gap_open"], ge=r["gap_extend"], zscore=r["zscore"],
                     free_end=r["free_end_gaps"], gap_weight=r["gap_weight"])
        f = tmp_path / "d.tsv"
        write_dist(names, D, str(f))
        nwk, _ = neighbor_joining(*read_dist(str(f)))
        assert nwk == r["newick"]
        assert normalized_rf(ref, nwk) == r["nRF"]


def test_empirical_parse_support_and_monophyly():
    from empirical_eval import parse, recovered, restrict

    nwk = "(a:0.1,b:0.2,((c:0.1,d:0.1)98:0.3,(e:0.1,f:0.1)/72:0.2)100:0.1);"
    leaves, spl = parse(nwk)
    assert leaves == list("abcdef")
    sup = {tuple(sorted(s)): v for s, v in spl.items()}
    assert sup[("c", "d")] == 98 and sup[("e", "f")] == 72 and sup[("c", "d", "e", "f")] == 100
    taxa = list("abcdef")
    assert recovered({"c", "d"}, set(spl), taxa)
    assert recovered({"a", "b"}, set(spl), taxa)          # complement of {c,d,e,f}: unrooted
    assert not recovered({"a", "c"}, set(spl), taxa)
    # pruning to {a,c,e,f}: {e,f} | {a,c} survives, {c,d} becomes trivial
    sub = ["a", "c", "e", "f"]
    assert restrict(set(spl), sub) == {frozenset({"e", "f"})}
    assert recovered({"a", "c"}, restrict(set(spl), sub), sub)


def test_empirical_group_rule():
    import pandas as pd

    from empirical_eval import eligible_groups

    sp = pd.DataFrame(dict(taxon=["s1", "s2", "s3", "s4", "s5", "s6"],
                           phylum=["P", "P", "P", "Q", "Q", "Q"],
                           **{"class": ["A", "A", "B", "C", "C", "C"]},
                           order=["o1", "o2", "o3", "o4", "o4", "o5"]))
    g = {(r, n): sorted(m) for r, n, m in eligible_groups(sp, list(sp.taxon), ["phylum", "class", "order"])}
    assert g[("phylum", "P")] == ["s1", "s2", "s3"]
    assert ("class", "B") not in g and ("order", "o1") not in g     # < 2 species
    assert g[("order", "o4")] == ["s4", "s5"]
    # in a subset of 3 taxa, a 2-member group has < 2 outside -> not eligible
    assert eligible_groups(sp, ["s1", "s2", "s4"], ["phylum"]) == []
