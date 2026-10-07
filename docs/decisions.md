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
- 2026-10-06 — **Hypothesis (written before the tuning run): ESM-2 embeddings are anisotropic.** —
  In the first full run (untuned PEA defaults) ESM-2 was worse than one-hot. A plausible cause is a
  large component shared by all residues, which makes every cosine high and compresses the range
  that carries phylogenetic signal. Test: add *centering* to the tuning grid — subtract the mean
  vector over all residues of all sequences of the dataset, then L2-normalise (`_unit`). Expected:
  centering helps ESM-2, and has little effect on one-hot (there it only removes the dataset's
  amino-acid composition). Centering uses only the unaligned sequences of the dataset itself, no
  tree or alignment, so it does not leak.
- 2026-10-06 — **PEA tuning protocol (Phase 1).** — Separate tuning grid (`workflow/tuning.smk`,
  outputs in `data_tuning/`, `results_tuning/`): same simulation settings as the test grid, 5 reps
  per cell (50 datasets), seeds `seed + sim.tuning_seed_offset + index` (no overlap with test
  seeds). Grid: gap_open {1,2,4} × gap_extend {0.25,0.5,1} × zscore {T,F} × free_end_gaps {T,F} ×
  gap_weight {0,0.5,1} × centering {T,F}; for ESM-2 additionally layer {4,6,8,12}. Parameters are
  chosen **per model** (one-hot control tuned as fairly as ESM-2) by mean nRF over all 50 datasets;
  ties → mean nQD → fewest changes from the current defaults. Layer and centering are properties
  of the embedding, so the chosen values are applied to every method of that model (meanpool,
  msaresid, pea), not only to PEA.
