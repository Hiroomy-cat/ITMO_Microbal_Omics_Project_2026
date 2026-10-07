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
- 2026-10-06 — **Frozen tuned settings** (tuning grid, 50 datasets, 1080 settings; mean nRF;
  `results_tuning/pea_grid_summary.tsv`, `pea_best.tsv`). Unique winners, no ties:
  - ESM-2 35M: **layer 4, centered**, gap_open 4, gap_extend 0.25, zscore, free end gaps,
    gap_weight 0 → nRF 0.209 (untuned defaults, last layer: 0.342).
  - one-hot: **centered**, gap_open 2, gap_extend 0.25, no zscore, global ends, gap_weight 0.5 →
    nRF 0.200 (defaults: 0.332).
  Applied per protocol: layer 4 and centering to all ESM-2 methods; centering to all one-hot
  methods (meanpool, msaresid, pea). —
  Observations: (1) the layer is the dominant ESM-2 factor (best nRF per layer: 4: 0.209,
  6: 0.262, 12: 0.282, 8: 0.360); (2) centering helps ESM-2 at every layer (−0.006 to −0.030),
  consistent with the anisotropy hypothesis but a smaller effect than the layer; (3) centering
  also helps one-hot (−0.010), against our expectation; (4) the top settings of each model lie
  within ~0.01 nRF of each other, so the exact winner is partly selection noise; (5) layer 4 is
  the lower edge of the tested range, so earlier layers might be better still — **not** explored,
  to avoid open-ended tuning (limitation); (6) on the tuning set tuned ESM-2 ≈ tuned one-hot.
- 2026-10-07 — **POST-HOC ablation (explanation, NOT tuning): PEA with gap_weight = 0.** —
  On the test grid tuned PEA beat MSA-based methods at h = 4 (mean nRF: PEA one-hot 0.396,
  PEA ESM-2 0.439, msaresid TRUE one-hot 0.489, ML 0.559). Question: how much of this comes from
  the indel term (gap_weight × unmatched fraction), which no MSA-residue method has? Recompute PEA
  on the main grid with all frozen settings except gap_weight = 0 (`workflow/ablation.smk`,
  outputs in `results_ablation/`, main `results/` untouched) and compare nRF at h = 2 and h = 4 with
  the main PEA and msaresid TRUE. Note: the frozen ESM-2 setting already has gap_weight = 0, so for
  ESM-2 the ablation must reproduce the main PEA exactly (a reproducibility check); only one-hot
  changes. The ablation result is descriptive and does not change any frozen parameter.
- 2026-10-07 — **POST-HOC ablation result** (`results_ablation/ablation_gw0.tsv`; 40 datasets per
  height; paired differences, 95 % bootstrap CI; descriptive, no tests, nothing re-tuned). —
  - ESM-2: gap_weight = 0 reproduces the main PEA exactly (max |ΔnRF| = 0 at h = 2 and 4) —
    reproducibility check passed.
  - one-hot: removing the indel term worsens PEA by ΔnRF +0.055 [0.035, 0.075] at h = 2 and
    +0.098 [0.066, 0.129] at h = 4. Without it, PEA one-hot at h = 4 equals the TRUE-MSA identity
    distance (0.494 vs msaresid TRUE 0.489; Δ +0.005 [−0.041, 0.050]); with it, PEA was better
    (Δ −0.093 [−0.140, −0.044]).
  - ESM-2 PEA is worse than msaresid TRUE with ESM-2 at both heights (h = 4: 0.439 vs 0.360;
    Δ +0.079 [0.042, 0.120]).
  Interpretation: one-hot PEA's advantage over MSA-residue distances at high divergence comes
  from the gap_weight × unmatched-fraction term, i.e. from indel information that the
  MSA-residue distance (matched columns only) discards — not from better residue matching.
  ESM-2 PEA does not exceed its own MSA upper bound; its lead over IQ-TREE ML at h = 4 reflects
  ML degrading on MAFFT alignments (msaresid TRUE ESM-2 also beats ML there). Limitation: the
  MSA-based baselines get no comparable indel term, so the PEA vs msaresid comparison is not
  like-for-like at high divergence.
