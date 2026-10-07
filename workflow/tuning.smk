# =============================================================================
# PEA TUNING grid (Phase 1) — separate datasets, never the test set.
#   same simulation settings as the main grid, seeds offset by sim.tuning_seed_offset,
#   fewer replicates; outputs only under data_tuning/, results_tuning/, logs_tuning/,
#   benchmarks_tuning/ (the main results/ are not touched).
# Run:  snakemake -s workflow/tuning.smk --cores 8 --resources gpu=1
# =============================================================================
import json
import re

configfile: "config/config.yaml"

S = config["sim"]
T = config["tuning"]
SCRIPTS = "workflow/scripts"


def fmt_h(h):
    return f"{float(h):g}"


DATASETS = [f"n{n}_h{fmt_h(h)}_r{r}"
            for n in S["n_taxa"] for h in S["tree_heights"] for r in range(T["replicates"])]
DS_INDEX = {d: i for i, d in enumerate(DATASETS)}


def seed(ds):
    return config["seed"] + S["tuning_seed_offset"] + DS_INDEX[ds]


# embedding label -> (model, layer); onehot has no layers
EMBS = {}
for m in config["embedding"]["models"]:
    if m == "onehot":
        EMBS[m] = (m, -1)
    else:
        for L in T["layers"][m]:
            EMBS[f"{m}_L{L}"] = (m, L)

wildcard_constraints:
    ds=r"n\d+_h[\d.]+_r\d+",
    emb="|".join(EMBS),


def ds_param(ds, what):
    n, h, r = re.match(r"n(\d+)_h([\d.]+)_r(\d+)", ds).groups()
    return {"n": int(n), "h": float(h), "r": int(r)}[what]


rule all:
    input:
        "results_tuning/pea_grid_summary.tsv",
        "results_tuning/pea_best.tsv",


rule tune_sim_tree:
    output: "data_tuning/sim/{ds}/true.nwk"
    params: n=lambda w: ds_param(w.ds, "n"), h=lambda w: ds_param(w.ds, "h"), seed=lambda w: seed(w.ds)
    shell:
        "python {SCRIPTS}/sim_tree.py --n {params.n} --height {params.h} "
        "--rate-sigma {S[rate_sigma]} --seed {params.seed} --out {output}"


rule tune_alisim:
    input: "data_tuning/sim/{ds}/true.nwk"
    output: "data_tuning/sim/{ds}/true_aln.fa"
    log: "logs_tuning/alisim/{ds}.log"
    params: prefix="data_tuning/sim/{ds}/alisim", seed=lambda w: seed(w.ds)
    shell:
        """
        {config[iqtree_bin]} --alisim {params.prefix} -t {input} -m '{S[model]}' \
            --length {S[length]} --indel {S[indel_rates]} --indel-size '{S[indel_size]}' \
            --seed {params.seed} -af fasta -redo > {log} 2>&1
        mv {params.prefix}.fa {output}
        rm -f {params.prefix}.unaligned.fa
        """


rule tune_degap:
    input: "data_tuning/sim/{ds}/true_aln.fa"
    output: "data_tuning/sim/{ds}/unaligned.fa"
    shell: "python {SCRIPTS}/degap.py {input} {output}"


rule tune_embed:
    input: "data_tuning/sim/{ds}/unaligned.fa"
    output: "data_tuning/emb/{ds}/{emb}.npz"
    benchmark: "benchmarks_tuning/{ds}/embed__{emb}.tsv"
    # as in the main Snakefile: run with --resources gpu=1 — parallel ESM jobs on the 4 GB laptop
    # GPU hit CUDA OOM (and 8 of them crashed the WSL VM)
    resources: gpu=lambda w: 0 if EMBS[w.emb][0] == "onehot" else 1
    params:
        model=lambda w: EMBS[w.emb][0],
        layer=lambda w: EMBS[w.emb][1],
        device=config["embedding"]["device"],
    shell:
        "python {SCRIPTS}/embed.py --fasta {input} --model {params.model} "
        "--layer {params.layer} --device {params.device} --out {output}"


rule tune_pea_grid:
    input:
        emb="data_tuning/emb/{ds}/{emb}.npz",
        ref="data_tuning/sim/{ds}/true.nwk",
    output: "results_tuning/grid/{ds}/{emb}.tsv"
    benchmark: "benchmarks_tuning/{ds}/tune_pea__{emb}.tsv"
    params:
        model=lambda w: EMBS[w.emb][0],
        layer=lambda w: EMBS[w.emb][1],
        grid=json.dumps(T["grid"]),
    shell:
        "python {SCRIPTS}/tune_pea.py --emb {input.emb} --ref {input.ref} --dataset {wildcards.ds} "
        "--model {params.model} --layer {params.layer} --grid '{params.grid}' --out {output}"


rule tune_summary:
    input: expand("results_tuning/grid/{ds}/{emb}.tsv", ds=DATASETS, emb=list(EMBS))
    output:
        summary="results_tuning/pea_grid_summary.tsv",
        best="results_tuning/pea_best.tsv",
    params: defaults=json.dumps({"pea": config["pea"]["default"], "layer": config["embedding"]["layer"]})
    shell:
        "python {SCRIPTS}/tune_summary.py --grid {input} --ref-dir data_tuning/sim "
        "--defaults '{params.defaults}' --summary {output.summary} --best {output.best}"
