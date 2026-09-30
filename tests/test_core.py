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
