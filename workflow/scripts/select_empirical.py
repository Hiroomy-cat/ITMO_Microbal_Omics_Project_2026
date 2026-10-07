#!/usr/bin/env python
"""Deterministic choice of bacterial species for the empirical check (Phase 3).

Candidates: organisms with a REVIEWED (Swiss-Prot) entry for EVERY requested gene, inside a
UniProt reference proteome (keyword KW-1185). Per phylum, species are picked round-robin over
classes, then orders (sorted by name), the lowest NCBI taxid first; one strain per species.
The UniProt release is recorded, because the candidate set changes between releases.

Output TSV: taxon (short id used in trees), phylum, class, order, species, organism_id,
and one accession column per gene.
"""
import argparse
import csv
import io
import re
import time
import urllib.parse
import urllib.request
from collections import OrderedDict

API = "https://rest.uniprot.org/uniprotkb/search"
RANKS = ("phylum", "class", "order")


def fetch(query, fields):
    """All rows of a UniProtKB query as dicts (follows pagination)."""
    url = f"{API}?{urllib.parse.urlencode(dict(query=query, fields=fields, format='tsv', size=500))}"
    rows, release = [], None
    while url:
        for attempt in range(5):
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    release = r.headers.get("X-UniProt-Release")
                    text = r.read().decode()
                    link = r.headers.get("Link", "")
                break
            except OSError:
                time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"UniProt query failed: {query}")
        rows += list(csv.DictReader(io.StringIO(text), delimiter="\t"))
        m = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = m.group(1) if m else None
    return rows, release


def lineage_ranks(lineage):
    """'Bacteria (superkingdom), Pseudomonadota (phylum), ...' -> {rank: name}."""
    out = {}
    for part in lineage.split(", "):
        m = re.match(r"(.+) \((\w[\w ]*)\)$", part.strip())
        if m and m.group(2) in RANKS:
            out[m.group(2)] = m.group(1)
    return out


def candidates(phylum_taxid, genes):
    per_gene, release = {}, None
    for g in genes:
        q = f"gene_exact:{g} AND taxonomy_id:{phylum_taxid} AND reviewed:true AND keyword:KW-1185"
        rows, release = fetch(q, "accession,organism_id,organism_name,lineage,length")
        per_gene[g] = {}
        for r in sorted(rows, key=lambda r: r["Entry"]):
            per_gene[g].setdefault(int(r["Organism (ID)"]), r)  # first accession per organism
    common = sorted(set.intersection(*(set(v) for v in per_gene.values())))
    out = []
    for oid in common:
        r = per_gene[genes[0]][oid]
        ranks = lineage_ranks(r["Taxonomic lineage"])
        out.append(dict(organism_id=oid, organism=r["Organism"],
                        species=" ".join(r["Organism"].split()[:2]),
                        **{k: ranks.get(k, "NA") for k in RANKS},
                        **{f"acc_{g}": per_gene[g][oid]["Entry"] for g in genes}))
    return out, release


def pick(cands, k):
    """Round-robin: each round takes one species per class (classes sorted by name); inside a
    class the orders take turns. Lowest taxid first; one strain per species."""
    seen, tree = set(), OrderedDict()
    for c in sorted(cands, key=lambda c: (c["class"], c["order"], c["organism_id"])):
        if c["species"] not in seen:
            seen.add(c["species"])
            tree.setdefault(c["class"], OrderedDict()).setdefault(c["order"], []).append(c)
    chosen = []
    while len(chosen) < k and any(q for orders in tree.values() for q in orders.values()):
        for orders in tree.values():
            for o in list(orders):
                if orders[o]:
                    chosen.append(orders[o].pop(0))
                    orders.move_to_end(o)  # this order goes last in the class's next turn
                    break
            if len(chosen) == k:
                break
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phyla", nargs="+", required=True, help="name:taxid pairs")
    ap.add_argument("--genes", nargs="+", required=True)
    ap.add_argument("--per-phylum", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry", action="store_true", help="only print candidate counts per phylum")
    a = ap.parse_args()

    rows, release = [], None
    for item in a.phyla:
        name, taxid = item.split(":")
        cands, release = candidates(int(taxid), a.genes)
        n_cls = len({c["class"] for c in cands})
        n_ord = len({c["order"] for c in cands})
        n_sp = len({c["species"] for c in cands})
        print(f"{name:18s} candidates={len(cands):3d} species={n_sp:3d} classes={n_cls} orders={n_ord}")
        if a.dry:
            continue
        for c in pick(cands, a.per_phylum):
            c["phylum"] = name
            rows.append(c)
    if a.dry:
        return
    cols = ["taxon", "phylum", "class", "order", "species", "organism", "organism_id"] + \
           [f"acc_{g}" for g in a.genes]
    with open(a.out, "w") as fh:
        fh.write(f"# UniProt release {release}; reviewed entries in reference proteomes (KW-1185)\n")
        fh.write("\t".join(cols) + "\n")
        for i, r in enumerate(rows, 1):
            r["taxon"] = f"s{i:02d}"
            fh.write("\t".join(str(r[c]) for c in cols) + "\n")


if __name__ == "__main__":
    main()
