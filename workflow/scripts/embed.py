#!/usr/bin/env python
"""Per-residue embeddings for every sequence of an UNALIGNED FASTA.

Output: .npz, key = sequence id, value = float16 array (L x D), row i = residue i.

Models
  onehot            20-dim one-hot (+ zero vector for non-standard aa). CONTROL: with it,
                    the PLM-based methods collapse to classical identity-based distances,
                    so any gain of a real PLM over `onehot` is the contribution of the model.
  esm2_t6_8M_UR50D, esm2_t12_35M_UR50D, esm2_t30_150M_UR50D, esm2_t33_650M_UR50D
                    fair-esm; `--layer -1` = last layer.
  esmc_300m, esmc_600m
                    EvolutionaryScale `esm` package (optional, see README).
"""
import argparse
import sys

import numpy as np

from common import read_fasta

AA = "ACDEFGHIKLMNPQRSTVWY"


def embed_onehot(seqs):
    idx = {a: i for i, a in enumerate(AA)}
    out = {}
    for k, s in seqs.items():
        E = np.zeros((len(s), len(AA)), dtype=np.float16)
        for i, c in enumerate(s.upper()):
            j = idx.get(c)
            if j is not None:
                E[i, j] = 1.0
        out[k] = E
    return out


def embed_esm2(seqs, model_name, layer, device, max_len=1022):
    import esm  # fair-esm
    import torch

    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model.eval().to(device)
    conv = alphabet.get_batch_converter()
    L = model.num_layers if layer < 0 else layer
    out = {}
    with torch.no_grad():
        for k, s in seqs.items():
            chunks = [s[i:i + max_len] for i in range(0, len(s), max_len)]
            if len(chunks) > 1:
                print(f"[embed] {k}: length {len(s)} > {max_len}, embedding in {len(chunks)} chunks", file=sys.stderr)
            parts = []
            for ch in chunks:
                _, _, toks = conv([(k, ch)])
                rep = model(toks.to(device), repr_layers=[L])["representations"][L]
                parts.append(rep[0, 1:len(ch) + 1].float().cpu().numpy())  # strip BOS/EOS
            out[k] = np.concatenate(parts).astype(np.float16)
    return out


def embed_esmc(seqs, model_name, layer, device):
    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMProtein, LogitsConfig

    client = ESMC.from_pretrained(model_name).to(device)
    out = {}
    for k, s in seqs.items():
        t = client.encode(ESMProtein(sequence=s))
        res = client.logits(t, LogitsConfig(sequence=True, return_embeddings=True, return_hidden_states=layer >= 0))
        E = res.hidden_states[layer][0] if layer >= 0 else res.embeddings[0]
        out[k] = E[1:len(s) + 1].float().cpu().numpy().astype(np.float16)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, default=-1)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    seqs = read_fasta(a.fasta)
    if a.model == "onehot":
        emb = embed_onehot(seqs)
    else:
        import torch
        dev = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
        if a.model.startswith("esm2_"):
            emb = embed_esm2(seqs, a.model, a.layer, dev)
        elif a.model.startswith("esmc_"):
            emb = embed_esmc(seqs, a.model, a.layer, dev)
        else:
            sys.exit(f"Unknown model {a.model}")
    for k, s in seqs.items():
        assert emb[k].shape[0] == len(s), f"{k}: {emb[k].shape[0]} rows vs {len(s)} residues"
    np.savez_compressed(a.out, **emb)


if __name__ == "__main__":
    main()
