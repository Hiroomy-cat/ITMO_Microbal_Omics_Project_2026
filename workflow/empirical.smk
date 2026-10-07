# =============================================================================
# EMPIRICAL check (Phase 3, reduced): RecA and RpsC of 42 bacteria from 7 phyla.
#   inputs (committed):  data/empirical/species.tsv, data/empirical/<family>.fa
#   outputs:             results_empirical/  (+ benchmarks_empirical/, logs_empirical/)
# All method settings are the FROZEN ones from config/config.yaml — nothing is tuned here.
# Run:  snakemake -s workflow/empirical.smk --cores 8 --resources gpu=1
# =============================================================================
configfile: "config/config.yaml"
configfile: "config/empirical.yaml"

E = config["empirical"]
FAMILIES = list(E["families"])
EMB_MODELS = config["embedding"]["models"]
EMB_METHODS = ["meanpool", "pea", "msaresid_mafft"]     # no TRUE alignment for real proteins
OTHER_METHODS = ["ml", "bionj", "kmer"]
SCRIPTS = "workflow/scripts"
R = "results_empirical"

wildcard_constraints:
    fam="|".join(FAMILIES),
    model=r"[A-Za-z0-9_]+",


def trees_for(fam):
    t = [f"{R}/trees/{fam}/{m}__{mo}.nwk" for m in EMB_METHODS for mo in EMB_MODELS]
    return t + [f"{R}/trees/{fam}/{m}__none.nwk" for m in OTHER_METHODS]


def center_flag(w):
    return "--center" if config["embedding"].get("center", {}).get(w.model, False) else ""


def pea_args(w):
    p = config["pea"].get(w.model, config["pea"]["default"])
    return (f"--gap-open {p['gap_open']} --gap-extend {p['gap_extend']} --gap-weight {p['gap_weight']}"
            + ("" if p["zscore"] else " --no-zscore") + ("" if p["free_end_gaps"] else " --global-ends"))


def fam_seed(fam):
    return config["seed"] + FAMILIES.index(fam)


rule all:
    input: expand(f"{R}/{{t}}.tsv", t=["groups", "monophyly", "supported", "robustness", "group_by_method"])


# ----------------------------------------------------------------------------- data (network)
rule emp_select:
    output: "data/empirical/species.tsv"
    params:
        phyla=" ".join(f"{k}:{v}" for k, v in E["phyla"].items()),
        genes=" ".join(E["families"].values()),
    shell:
        "python {SCRIPTS}/select_empirical.py --phyla {params.phyla} --genes {params.genes} "
        "--per-phylum {E[per_phylum]} --out {output}"


rule emp_fetch:
    input: "data/empirical/species.tsv"
    output: "data/empirical/{fam}.fa"
    params: gene=lambda w: E["families"][w.fam]
    shell: "python {SCRIPTS}/fetch_empirical.py --species {input} --gene {params.gene} --out {output}"


# ----------------------------------------------------------------------------- alignment & embeddings
rule emp_mafft:
    input: "data/empirical/{fam}.fa"
    output: f"{R}/aln/{{fam}}.mafft.fa"
    log: "logs_empirical/mafft/{fam}.log"
    shell: "{config[mafft_bin]} --auto --quiet --anysymbol {input} > {output} 2> {log}"


rule emp_embed:
    input: "data/empirical/{fam}.fa"
    output: f"{R}/emb/{{fam}}/{{model}}.npz"
    benchmark: "benchmarks_empirical/{fam}/embed__{model}.tsv"
    resources: gpu=lambda w: 0 if w.model == "onehot" else 1
    params: layer=config["embedding"]["layer"], device=config["embedding"]["device"]
    shell:
        "python {SCRIPTS}/embed.py --fasta {input} --model {wildcards.model} "
        "--layer {params.layer} --device {params.device} --out {output}"


# ----------------------------------------------------------------------------- distances & trees
rule emp_pea:
    input: f"{R}/emb/{{fam}}/{{model}}.npz"
    output: f"{R}/dist/{{fam}}/pea__{{model}}.tsv"
    params: args=pea_args, center=center_flag
    shell: "python {SCRIPTS}/dist_emb.py --emb {input} --method pea {params.args} {params.center} --out {output}"


rule emp_meanpool:
    input: f"{R}/emb/{{fam}}/{{model}}.npz"
    output: f"{R}/dist/{{fam}}/meanpool__{{model}}.tsv"
    params: center=center_flag
    shell:
        "python {SCRIPTS}/dist_emb.py --emb {input} --method meanpool "
        "--metric {config[meanpool][metric]} {params.center} --out {output}"


rule emp_msaresid:
    input: emb=f"{R}/emb/{{fam}}/{{model}}.npz", aln=f"{R}/aln/{{fam}}.mafft.fa"
    output: f"{R}/dist/{{fam}}/msaresid_mafft__{{model}}.tsv"
    params: center=center_flag
    shell:
        "python {SCRIPTS}/dist_emb.py --emb {input.emb} --method msaresid --aln {input.aln} "
        "{params.center} --out {output}"


rule emp_kmer:
    input: "data/empirical/{fam}.fa"
    output: f"{R}/dist/{{fam}}/kmer__none.tsv"
    shell: "python {SCRIPTS}/dist_kmer.py --fasta {input} --k {config[kmer][k]} --out {output}"


rule emp_nj:
    input: f"{R}/dist/{{fam}}/{{method}}__{{model}}.tsv"
    output: f"{R}/trees/{{fam}}/{{method}}__{{model}}.nwk"
    wildcard_constraints: method="meanpool|pea|msaresid_mafft|kmer"
    log: "logs_empirical/nj/{fam}/{method}__{model}.log"
    shell: "python {SCRIPTS}/nj.py {input} {output} 2> {log}"


rule emp_iqtree_method:   # the "ml"/"bionj" METHODS: same settings as on the simulated grid
    input: f"{R}/aln/{{fam}}.mafft.fa"
    output: ml=f"{R}/iqtree/{{fam}}/method.treefile", bionj=f"{R}/iqtree/{{fam}}/method.bionj"
    params: prefix=f"{R}/iqtree/{{fam}}/method", seed=lambda w: fam_seed(w.fam)
    shell:
        "{config[iqtree_bin]} -s {input} -m {config[iqtree][model]} {config[iqtree][extra]} "
        "--seed {params.seed} --prefix {params.prefix} -redo --quiet"


rule emp_collect:
    input: ml=f"{R}/iqtree/{{fam}}/method.treefile", bionj=f"{R}/iqtree/{{fam}}/method.bionj"
    output: ml=f"{R}/trees/{{fam}}/ml__none.nwk", bionj=f"{R}/trees/{{fam}}/bionj__none.nwk"
    shell: "cp {input.ml} {output.ml} && cp {input.bionj} {output.bionj}"


rule emp_iqtree_ref:      # the REFERENCE: model selection + UFBoot
    input: f"{R}/aln/{{fam}}.mafft.fa"
    output: f"{R}/iqtree/{{fam}}/ref.treefile"
    benchmark: "benchmarks_empirical/{fam}/iqtree_ref.tsv"
    threads: 2
    params: prefix=f"{R}/iqtree/{{fam}}/ref", seed=lambda w: fam_seed(w.fam)
    shell:
        "{config[iqtree_bin]} -s {input} -m {E[reference][iqtree_model]} -B {E[reference][ufboot]} "
        "-T {threads} -keep-ident --seed {params.seed} --prefix {params.prefix} -redo --quiet"


# ----------------------------------------------------------------------------- evaluation
rule emp_eval:
    input:
        species="data/empirical/species.tsv",
        ref=f"{R}/iqtree/{{fam}}/ref.treefile",
        trees=lambda w: trees_for(w.fam),
    output: expand(f"{R}/metrics/{{{{fam}}}}/{{t}}.tsv", t=["groups", "monophyly", "supported", "robustness"])
    params: ranks=" ".join(E["ranks"])
    shell:
        "python {SCRIPTS}/empirical_eval.py --family {wildcards.fam} --species {input.species} "
        "--ref {input.ref} --trees {input.trees} --ranks {params.ranks} "
        "--min-support {E[reference][min_support]} --drop-phylum {E[sensitivity_drop_phylum]} "
        "--reps {E[robustness][replicates]} --per-phylum {E[robustness][per_phylum]} "
        "--seed {config[seed]} --outdir {R}/metrics/{wildcards.fam}"


rule emp_aggregate:
    input: expand(f"{R}/metrics/{{fam}}/{{t}}.tsv", fam=FAMILIES, t=["groups", "monophyly", "supported", "robustness"])
    output: expand(f"{R}/{{t}}.tsv", t=["groups", "monophyly", "supported", "robustness", "group_by_method"])
    params: dirs=" ".join(f"{R}/metrics/{f}" for f in FAMILIES)
    shell: "python {SCRIPTS}/empirical_aggregate.py --dirs {params.dirs} --outdir {R}"
