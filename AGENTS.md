# AGENTS.md — instructions for coding agents

Read this file, `README.md` and `todo.md` before doing anything. `CLAUDE.md` / `.cursor/rules`
may simply point here.

## 1. Project in one paragraph

We test whether phylogenetic signal in residue-level protein-language-model (PLM) embeddings
can be used **without a multiple sequence alignment**. The method under test is **PEA**
(Pairwise Embedding Alignment distances, `workflow/scripts/dist_emb.py::pea_pair`): per pair of
proteins, align residues by dynamic programming on the embedding cosine-similarity matrix, then
average (1 − cos) over matched residues; build NJ trees from these distances. Everything is
benchmarked on simulated protein families with a known true tree (AliSim), against mean-pooled
embeddings (negative baseline), MSA-based residue distances (upper bound), IQ-TREE ML, BIONJ and
3-mers. Hypotheses H0/H1/H1b are in `README.md` — **do not change them without the human.**

## 2. Repository map

| Path | Role |
|---|---|
| `config/config.yaml` | the single source of truth for parameters; no magic numbers in scripts |
| `config/smoke.yaml` | tiny grid; must always finish in < 10 min on CPU |
| `workflow/Snakefile` | pipeline; every output file is produced by a rule |
| `workflow/scripts/common.py` | FASTA/dist I/O, NJ, Newick → splits, nRF, quartet distance |
| `workflow/scripts/dist_emb.py` | meanpool / msaresid / **PEA** |
| `workflow/scripts/embed.py` | onehot, ESM-2 (fair-esm), ESM-C (separate env) |
| `workflow/scripts/stats_plots.py` | tests + figures |
| `tests/` | unit tests (pytest) |
| `docs/decisions.md` | append-only log of design decisions |

## 3. Naming conventions (the pipeline depends on them)

- Dataset id: `n{taxa}_h{height}_r{replicate}` (e.g. `n32_h2_r7`).
- Distance / tree files: `results/{dist|trees}/{ds}/{method}__{model}.{tsv|nwk}`;
  `model = none` for methods that do not use embeddings. Double underscore is the separator —
  never use `__` inside method or model names.
- Taxon names come from the simulator (`t1..tN`); never rename them.
- Adding a method = new rule producing `results/dist/{ds}/<method>__{model}.tsv`, add it to
  `EMB_METHODS` or `OTHER_METHODS` and to the `nj` rule's `method` constraint, add a label in
  `stats_plots.LABEL`, add a unit test.

## 4. Hard rules

1. **Never tune on the test set.** PEA gap penalties / z-score / gap_weight / model layer are
   chosen on the *tuning* grid (seed + `sim.tuning_seed_offset`) and then frozen in
   `config.yaml` with a note in `docs/decisions.md`. Reporting a test-set-tuned result is a bug.
2. Do not change evaluation code (`compare.py`, metric functions in `common.py`) in the same
   commit as a method change.
3. Do not edit anything under `results/` or `data/` by hand; regenerate with Snakemake.
4. All randomness goes through seeds derived from `config.seed`; runs must be bit-reproducible
   on CPU.
5. Keep the one-hot control and the mean-pool baseline in every run.
6. If a result looks too good (e.g. PEA beats ML with TRUE MSA everywhere), stop and look for
   leakage (true alignment/tree used where it should not be) before celebrating.
7. Negative results are acceptable and must be reported honestly — do not add ad-hoc tweaks
   until PEA "wins".
8. Small commits with messages `<area>: <what> (<why>)`; the human's commit history is graded.
9. Ask the human before: changing hypotheses, adding large dependencies, downloading > 1 GB,
   running jobs longer than ~2 h.

## 5. How to verify your work (run before every commit)

```bash
pytest -q tests/
snakemake -s workflow/Snakefile --configfile config/smoke.yaml --cores 4 -n    # dry-run DAG
snakemake -s workflow/Snakefile --configfile config/smoke.yaml --cores 4       # when logic changed
```

Sanity expectations on the smoke grid (16 taxa): mean-pool(onehot) is clearly the worst;
ML ≈ MSA-residue methods at low divergence; every nRF is in [0, 1]; no NaN in `summary.tsv`.

## 6. Tool notes

- AliSim: `iqtree2 --alisim PREFIX -t TREE -m 'LG+G4{0.6}' --indel 0.03,0.09 --indel-size 'POW{1.7/50},POW{1.7/50}' -af fasta`
  writes `PREFIX.fa` (true MSA, headers padded with spaces — `read_fasta` strips them).
- IQ-TREE writes `.treefile`, `.bionj`, `.mldist`; we use the first two.
- `fair-esm` and EvolutionaryScale `esm` share the import name `esm` → separate envs.
- PEA's DP is numba-jitted (`cache=True`); the first call per process compiles (~1–2 s).
- Quartet distance is O(n⁴); switch `compare.quartets: false` or use R `TreeDist` for n > 40.

## 7. Definition of done for the mini-project

- [ ] Full grid run, `results/summary.tsv`, `results/stats/tests.tsv`, figures 1–3 committed (`git add -f`).
- [ ] Empirical check on ≥ 3 bacterial protein families (see `todo.md`).
- [ ] README results section: numbers, effect sizes, adjusted p-values, honest limitations.
- [ ] 3–4 slides in the course deck: problem → methods (mermaid) → results (+ GitHub link).
