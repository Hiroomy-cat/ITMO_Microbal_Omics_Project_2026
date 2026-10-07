# todo.md — plan & status

Deadline: **14.10.2026** (defence, 5 min, 3–4 slides). Idea registered: 30.09.
Legend: `[ ]` todo · `[~]` in progress · `[x]` done · (H) = needs the human

## Phase 0 — setup (30.09–01.10)
- [x] Repo skeleton, config, Snakefile, core scripts, unit tests
- [x] Smoke run end-to-end with one-hot control (16 taxa, 2 heights × 5 reps) — works
- [ ] (H) Register idea in the course spreadsheet
- [ ] (H) Create GitHub repo, first commit from the human's account
- [ ] Create conda env from `environment.yml`, run `pytest` + smoke run on own machine
- [ ] Check ESM-2 35M embedding on CPU for one dataset (time, memory); decide on GPU/Colab

## Phase 1 — tuning, NOT on the test set (02.10–04.10)
- [x] Add a `tuning` grid (`workflow/tuning.smk`): same generator, seeds offset by `sim.tuning_seed_offset`,
      output under `data_tuning/`, `results_tuning/` (keep test outputs untouched)
- [x] Grid-search PEA on tuning set: gap_open ∈ {1,2,4}, gap_extend ∈ {0.25,0.5,1},
      zscore ∈ {T,F}, free_end_gaps ∈ {T,F}, gap_weight ∈ {0,0.5,1}, centering ∈ {T,F};
      ESM-2 layer ∈ {4,6,8,12}
- [x] Freeze best settings in `config.yaml`; log them in `docs/decisions.md`
- [x] Recompute embedding-based methods on the main grid with the frozen settings

## Phase 2 — main simulation benchmark (05.10–08.10)
- [ ] Full grid: n ∈ {16, 32}, height ∈ {0.25…4}, 20 reps (→ 50 if time allows)
- [ ] Models: onehot (control), esm2_t12_35M; optional esm2_t30_150M / esmc_300m
- [ ] Check: no NaN, negative-branch counts in `logs/nj/`, runtimes plausible
- [ ] Read `results/stats/tests.tsv`: H1, H1b, distance to MSA upper bound

## Phase 3 — empirical microbial check (08.10–10.10)
- [ ] Pick ≥ 3 bacterial marker families (e.g. GTDB bac120: RpL2, RpS3, RecA/GyrB), 30–60 genomes
      spanning several phyla → `data/empirical/<family>.fa`
- [ ] Reference 1: GTDB taxonomy — fraction of genera/families recovered as monophyletic
- [ ] Reference 2: IQ-TREE ML (MFP, UFBoot 1000) — compare only splits with UFBoot ≥ 95
- [ ] Generalise Snakefile to empirical datasets (reference ≠ true tree)

## Phase 4 — write-up (11.10–13.10)
- [ ] Final figures (pdf) + captions understandable without the speaker
- [ ] README "Results" + "Limitations" (branch lengths not evaluated in depth, NJ only, sim model = LG)
- [ ] Slides in the course deck: problem / methods (mermaid from docs/pipeline.mmd) / results + GitHub link
- [ ] (H) Rehearse 5-minute talk

## Nice-to-have (only if ahead of schedule)
- [ ] Branch-length agreement (branch score / KF distance), as in Stakauskas & Górecki 2026
- [ ] Soft alignment (Sinkhorn / optimal transport) instead of hard DP
- [ ] Compare with Phyloformer checkpoint
- [ ] CVTree-style background subtraction for the k-mer baseline
