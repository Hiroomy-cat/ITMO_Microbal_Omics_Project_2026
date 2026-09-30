#!/usr/bin/env python
"""Concatenate per-dataset metric tables + attach wall-clock times from Snakemake benchmarks."""
import argparse
import glob
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", nargs="+", required=True)
    ap.add_argument("--bench-dir", default="benchmarks")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    df = pd.concat([pd.read_csv(f, sep="\t") for f in a.metrics], ignore_index=True)

    # benchmarks/<dataset>/<step>.tsv ; column "s" = seconds
    rows = []
    for f in glob.glob(os.path.join(a.bench_dir, "*", "*.tsv")):
        ds = os.path.basename(os.path.dirname(f))
        step = os.path.basename(f)[:-4]
        rows.append((ds, step, pd.read_csv(f, sep="\t")["s"].iloc[0]))
    b = pd.DataFrame(rows, columns=["dataset", "step", "s"]).set_index(["dataset", "step"])["s"] if rows else None

    def t(ds, step):
        try:
            return float(b.loc[(ds, step)])
        except Exception:
            return float("nan")

    def runtime(r):
        ds, me, mo = r.dataset, r.method, r.model
        if me in ("ml", "bionj"):
            return t(ds, "iqtree") + t(ds, "mafft")
        if me == "kmer":
            return t(ds, "dist_kmer__none")
        base = t(ds, f"dist_{me}__{mo}") + t(ds, f"embed__{mo}")
        return base + (t(ds, "mafft") if me == "msaresid_mafft" else 0.0)

    df["seconds"] = df.apply(runtime, axis=1)
    df["method_model"] = df["method"] + "__" + df["model"]
    df.to_csv(a.out, sep="\t", index=False)


if __name__ == "__main__":
    main()
