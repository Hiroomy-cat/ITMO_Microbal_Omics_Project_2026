# Design decisions (append-only)

Format: date — decision — why.

- 2026-09-30 — Evaluate on **simulated** families first. — Only simulations give a known true tree;
  empirical "reference" trees are themselves inferred and would favour MSA-based methods.
- 2026-09-30 — Simulator: Yule topology + uncorrelated lognormal branch-rate multipliers (σ = 0.3),
  AliSim LG+G4(α = 0.6) with realistic indels (ins 0.03, del 0.09, Zipf 1.7, max 50). —
  Non-clock trees and indels are where MSA errors appear, which H1b is about.
- 2026-09-30 — Divergence axis = mean root-to-tip distance 0.25…4 subs/site. — Covers easy to
  saturated/twilight-zone alignments.
- 2026-09-30 — Include **one-hot pseudo-embeddings** as a control. — With one-hot, PEA ≈ pairwise
  identity distance, so "PLM minus one-hot" isolates what the language model contributes.
- 2026-09-30 — Include **msaresid on the TRUE alignment**. — Separates signal loss caused by the
  embedding from loss caused by alignment error.
- 2026-09-30 — ML reference uses LG+G4 (the simulating model family). — Gives ML its best case;
  a PLM method that approaches it is convincing.
- 2026-09-30 — Distances in PEA use RAW cosine of matched residues, alignment uses z-scored
  similarity. — z-scoring helps the DP find homologous positions (EBA), but raw cosine keeps a
  scale that is comparable across pairs.
- 2026-09-30 — Metrics: normalised RF (primary), normalised quartet distance (secondary). —
  Quartets are less sensitive to a single misplaced taxon.
- 2026-09-30 — IQ-TREE runs with `-keep-ident`. — At low divergence (h = 0.25) AliSim produces
  identical sequences; by default IQ-TREE drops them and re-adds them only to `.treefile`, so the
  `.bionj` tree missed taxa (n16_h0.25_r17: 14/16) and `compare` failed. Keeping them makes ML and
  BIONJ trees contain all taxa and treats every dataset the same; finished ML runs are recomputed.
