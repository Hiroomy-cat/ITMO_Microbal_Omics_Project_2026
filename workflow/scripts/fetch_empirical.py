#!/usr/bin/env python
"""Download one protein family for the species in species.tsv from UniProt; headers -> taxon ids.

Fails unless every species gets exactly one sequence of standard amino-acid letters; X (unknown
residue, e.g. 1 position in T. pallidum RecA) is allowed and reported — every downstream tool treats
it as unknown. Sequences are never edited.
"""
import argparse
import io
import sys
import urllib.parse
import urllib.request

import pandas as pd

from common import read_fasta, write_fasta

STREAM = "https://rest.uniprot.org/uniprotkb/stream"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True)
    ap.add_argument("--gene", required=True, help="column acc_<gene> in species.tsv")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sp = pd.read_csv(a.species, sep="\t", comment="#")
    acc2taxon = dict(zip(sp[f"acc_{a.gene}"], sp.taxon))
    query = "accession:(" + " OR ".join(acc2taxon) + ")"
    url = f"{STREAM}?{urllib.parse.urlencode(dict(query=query, format='fasta'))}"
    with urllib.request.urlopen(url, timeout=120) as r:
        text = r.read().decode()

    tmp = io.StringIO(text)
    seqs = {}
    name = None
    for line in tmp:
        line = line.strip()
        if line.startswith(">"):
            name = line[1:].split("|")[1]          # >sp|P0A7G6|RECA_ECOLI ...
            seqs[name] = []
        elif line:
            seqs[name].append(line)
    seqs = {acc2taxon[k]: "".join(v) for k, v in seqs.items()}
    missing = set(sp.taxon) - set(seqs)
    assert not missing, f"no sequence for {sorted(missing)}"
    n_x = {k: v.count("X") for k, v in seqs.items() if "X" in v}
    if n_x:
        print(f"[fetch] {a.gene}: unknown residues X kept: {n_x}", file=sys.stderr)
    bad = {k: set(v) - set("ACDEFGHIKLMNPQRSTVWYX") for k, v in seqs.items()}
    bad = {k: v for k, v in bad.items() if v}
    assert not bad, f"non-standard residues: {bad}"
    write_fasta({t: seqs[t] for t in sp.taxon}, a.out)
    assert len(read_fasta(a.out)) == len(sp)


if __name__ == "__main__":
    main()
