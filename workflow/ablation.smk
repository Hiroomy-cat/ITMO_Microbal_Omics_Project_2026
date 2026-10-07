# =============================================================================
# POST-HOC ABLATION (explanation, not tuning): PEA with the frozen per-model settings but
# gap_weight = 0, on the MAIN test grid. Reads data/sim and data/emb of the main run (read-only),
# writes only results_ablation/ (+ benchmarks_ablation/, logs_ablation/). See docs/decisions.md.
# Run (after the main pipeline):  snakemake -s workflow/ablation.smk --cores 8
# =============================================================================
configfile: "config/config.yaml"

S = config["sim"]
EMB_MODELS = config["embedding"]["models"]
SCRIPTS = "workflow/scripts"
METHOD = "pea_gw0"   # PEA, frozen settings, gap_weight forced to 0


def fmt_h(h):
    return f"{float(h):g}"


DATASETS = [f"n{n}_h{fmt_h(h)}_r{r}"
            for n in S["n_taxa"] for h in S["tree_heights"] for r in range(S["replicates"])]

wildcard_constraints:
    ds=r"n\d+_h[\d.]+_r\d+",
    model="|".join(EMB_MODELS),


def pea_args_gw0(w):
    p = config["pea"].get(w.model, config["pea"]["default"])
    center = config["embedding"].get("center", {}).get(w.model, False)
    return (f"--gap-open {p['gap_open']} --gap-extend {p['gap_extend']} --gap-weight 0.0"
            + ("" if p["zscore"] else " --no-zscore") + ("" if p["free_end_gaps"] else " --global-ends")
            + (" --center" if center else ""))


rule all:
    input: "results_ablation/ablation_gw0.tsv", "results_ablation/fig_ablation_gw0.pdf"


rule abl_dist:
    input: "data/emb/{ds}/{model}.npz"
    output: "results_ablation/dist/{ds}/" + METHOD + "__{model}.tsv"
    benchmark: "benchmarks_ablation/{ds}/dist_" + METHOD + "__{model}.tsv"
    params: args=pea_args_gw0
    shell: "python {SCRIPTS}/dist_emb.py --emb {input} --method pea {params.args} --out {output}"


rule abl_nj:
    input: "results_ablation/dist/{ds}/" + METHOD + "__{model}.tsv"
    output: "results_ablation/trees/{ds}/" + METHOD + "__{model}.nwk"
    log: "logs_ablation/nj/{ds}/" + METHOD + "__{model}.log"
    shell: "python {SCRIPTS}/nj.py {input} {output} 2> {log}"


rule abl_compare:
    input:
        ref="data/sim/{ds}/true.nwk",
        trees=expand("results_ablation/trees/{{ds}}/" + METHOD + "__{model}.nwk", model=EMB_MODELS),
    output: "results_ablation/metrics/{ds}.tsv"
    params: q="--quartets" if config["compare"]["quartets"] else ""
    shell:
        "python {SCRIPTS}/compare.py --dataset {wildcards.ds} --ref {input.ref} "
        "--trees {input.trees} {params.q} --out {output}"


rule abl_report:
    input:
        metrics=expand("results_ablation/metrics/{ds}.tsv", ds=DATASETS),
        main="results/summary.tsv",
    output: "results_ablation/ablation_gw0.tsv"
    shell:
        "python {SCRIPTS}/ablation_report.py --metrics {input.metrics} --main {input.main} "
        "--heights 2 4 --out {output}"


rule abl_figure:
    input:
        metrics=expand("results_ablation/metrics/{ds}.tsv", ds=DATASETS),
        main="results/summary.tsv",
    output: pdf="results_ablation/fig_ablation_gw0.pdf", png="results_ablation/fig_ablation_gw0.png"
    shell:
        "python {SCRIPTS}/ablation_plot.py --metrics {input.metrics} --main {input.main} "
        "--heights 2 4 --out results_ablation/fig_ablation_gw0"
