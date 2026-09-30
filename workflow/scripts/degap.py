#!/usr/bin/env python
"""Remove gap characters from an aligned FASTA -> unaligned FASTA (same order/ids)."""
import sys

from common import degap, read_fasta, write_fasta

if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    seqs = {k: degap(v) for k, v in read_fasta(src).items()}
    empty = [k for k, v in seqs.items() if not v]
    if empty:
        sys.exit(f"Empty sequences after degapping: {empty}")
    write_fasta(seqs, dst)
