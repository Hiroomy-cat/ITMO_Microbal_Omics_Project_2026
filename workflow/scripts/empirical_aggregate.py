#!/usr/bin/env python
"""Concatenate per-family empirical tables and build the group x method table (1 = recovered)."""
import argparse
import os

import pandas as pd

TABLES = ("groups", "monophyly", "supported", "robustness")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True, help="per-family metric dirs")
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()

    for name in TABLES:
        df = pd.concat([pd.read_csv(os.path.join(d, f"{name}.tsv"), sep="\t") for d in a.dirs],
                       ignore_index=True)
        df.to_csv(os.path.join(a.outdir, f"{name}.tsv"), sep="\t", index=False, float_format="%.4f")
        if name == "groups":
            df["tree"] = df.method + "__" + df.model
            wide = df.pivot_table(index=["family", "rank", "group", "n_species"], columns="tree",
                                  values="recovered", aggfunc="first").reset_index()
            order = {"phylum": 0, "class": 1, "order": 2}
            wide = wide.sort_values(["family", "rank", "group"], key=lambda c: c.map(order) if c.name == "rank" else c)
            wide.to_csv(os.path.join(a.outdir, "group_by_method.tsv"), sep="\t", index=False)


if __name__ == "__main__":
    main()
