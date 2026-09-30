"""Synthetic scientific document corpus generator.

Generates 500+ short, realistic-sounding scientific text chunks: protocol
excerpts, assay SOP snippets, and experimental result summaries. Content is
synthetically assembled from real, varied technical vocabulary (reagents,
instruments, organisms, assay types, units) via structured templates -- it is
NOT lorem-ipsum filler and NOT copied from any real lab's documents.

Deterministic (seeded) so the corpus and any "planted document" used for
retrieval sanity checks are stable across runs.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List

SEED = 42

ASSAYS = [
    "ELISA", "Western blot", "qPCR", "flow cytometry", "ELISPOT",
    "immunohistochemistry", "chromatin immunoprecipitation (ChIP)",
    "co-immunoprecipitation", "size-exclusion chromatography",
    "mass spectrometry proteomics", "CRISPR knockout screening",
    "single-cell RNA sequencing", "surface plasmon resonance",
    "microscale thermophoresis", "cell viability (MTT) assay",
    "luciferase reporter assay", "electrophoretic mobility shift assay",
]

ORGANISMS = [
    "HEK293T cells", "Jurkat T cells", "primary murine splenocytes",
    "zebrafish embryos", "Saccharomyces cerevisiae", "E. coli BL21(DE3)",
    "CHO-K1 cells", "iPSC-derived cardiomyocytes", "HeLa cells",
    "C. elegans (N2 strain)", "primary human PBMCs", "A549 lung epithelial cells",
]

REAGENTS = [
    "anti-CD3 monoclonal antibody", "recombinant human IL-2", "Lipofectamine 3000",
    "SYBR Green master mix", "protease inhibitor cocktail", "DAPI nuclear stain",
    "TRIzol reagent", "Ni-NTA agarose resin", "puromycin selection medium",
    "FBS (heat-inactivated, 10%)", "RIPA lysis buffer", "Cas9 ribonucleoprotein complex",
    "streptavidin-HRP conjugate", "phalloidin-AlexaFluor 488", "EDTA (0.5 M, pH 8.0)",
]

INSTRUMENTS = [
    "BD FACSCanto II flow cytometer", "Bio-Rad CFX96 real-time PCR system",
    "Thermo Fisher NanoDrop spectrophotometer", "Zeiss LSM 880 confocal microscope",
    "Agilent 1260 Infinity HPLC", "Beckman Coulter Optima ultracentrifuge",
    "Illumina NovaSeq 6000", "Biacore T200 SPR instrument",
    "Tecan Infinite M200 Pro plate reader", "Sartorius Ambr 250 bioreactor",
]

UNITS_TIME = ["15 minutes", "30 minutes", "1 hour", "2 hours", "overnight at 4°C",
              "45 minutes", "90 minutes", "4 hours", "10 minutes on ice"]
UNITS_TEMP = ["4°C", "room temperature", "37°C", "-20°C", "-80°C", "95°C"]
UNITS_CONC = ["1 µg/mL", "10 nM", "0.1% (v/v)", "5 mM", "100 U/mL", "50 µg/mL", "2.5 mM"]

TOPICS = [
    "apoptosis induction", "protein-protein interaction mapping",
    "gene knockdown efficiency", "cytokine secretion profiling",
    "cell cycle arrest", "mitochondrial membrane potential",
    "antibody titer determination", "transfection efficiency",
    "enzyme kinetics", "receptor binding affinity", "oxidative stress response",
    "autophagy flux", "epitope mapping", "biomarker validation",
    "drug dose-response", "viral transduction efficiency",
]


@dataclass
class Document:
    doc_id: str
    doc_type: str
    title: str
    text: str
    tags: List[str] = field(default_factory=list)


def _protocol_excerpt(rng: random.Random, i: int) -> Document:
    assay = rng.choice(ASSAYS)
    organism = rng.choice(ORGANISMS)
    reagent = rng.choice(REAGENTS)
    instrument = rng.choice(INSTRUMENTS)
    t = rng.choice(UNITS_TIME)
    temp = rng.choice(UNITS_TEMP)
    conc = rng.choice(UNITS_CONC)
    step_n = rng.randint(2, 7)
    title = f"Protocol excerpt: {assay} on {organism} (step {step_n})"
    text = (
        f"Step {step_n}: Incubate {organism} with {reagent} at a final "
        f"concentration of {conc} for {t} at {temp}. Following incubation, "
        f"proceed with {assay} using the {instrument}. Ensure all samples are "
        f"processed in technical triplicate and include a vehicle-only "
        f"negative control. Record raw output files immediately after "
        f"acquisition to avoid signal drift."
    )
    tags = ["protocol", assay.lower(), organism.lower()]
    return Document(f"protocol-{i}", "protocol_excerpt", title, text, tags)


def _assay_sop(rng: random.Random, i: int) -> Document:
    assay = rng.choice(ASSAYS)
    reagent = rng.choice(REAGENTS)
    instrument = rng.choice(INSTRUMENTS)
    conc = rng.choice(UNITS_CONC)
    temp = rng.choice(UNITS_TEMP)
    title = f"SOP: {assay} standard operating procedure"
    text = (
        f"Purpose: This SOP describes the standard procedure for performing "
        f"{assay} in the laboratory. Materials required include {reagent} "
        f"prepared at {conc} and access to the {instrument}. Safety: wear "
        f"appropriate PPE and handle all reagents at {temp} unless otherwise "
        f"specified. Quality control: run a known positive and negative "
        f"control on every plate/run. Deviations from this SOP must be "
        f"documented in the lab notebook and approved by the PI before the "
        f"data is used in any report."
    )
    tags = ["sop", assay.lower()]
    return Document(f"sop-{i}", "assay_sop", title, text, tags)


def _result_summary(rng: random.Random, i: int) -> Document:
    assay = rng.choice(ASSAYS)
    organism = rng.choice(ORGANISMS)
    topic = rng.choice(TOPICS)
    pct = rng.randint(12, 94)
    pval = rng.choice(["0.001", "0.01", "0.02", "0.045", "0.003"])
    fold = round(rng.uniform(1.2, 8.5), 1)
    title = f"Result summary: {topic} in {organism}"
    text = (
        f"In this experiment, {organism} were assessed for {topic} using "
        f"{assay}. Treated samples showed a {fold}-fold change relative to "
        f"untreated controls, corresponding to {pct}% of the maximal "
        f"response observed in the positive control group (p < {pval}, "
        f"two-tailed t-test, n=3 biological replicates). These results are "
        f"consistent with the hypothesis that the treatment modulates "
        f"{topic} in a dose-dependent manner and warrant follow-up "
        f"validation with an orthogonal assay."
    )
    tags = ["result", assay.lower(), topic.replace(" ", "_")]
    return Document(f"result-{i}", "result_summary", title, text, tags)


def build_corpus(n: int = 520, seed: int = SEED) -> List[Document]:
    """Build a deterministic synthetic corpus of n documents (n >= 500)."""
    rng = random.Random(seed)
    docs: List[Document] = []
    generators = [_protocol_excerpt, _assay_sop, _result_summary]
    for i in range(n):
        gen = generators[i % len(generators)]
        docs.append(gen(rng, i))

    # Planted "known-relevant" document used by the test suite as a sanity
    # check: a very specific, unambiguous query should retrieve this exact
    # document in the top-k results.
    planted = Document(
        doc_id="planted-qpcr-mitochondrial-0001",
        doc_type="protocol_excerpt",
        title="Protocol excerpt: qPCR quantification of mitochondrial DNA copy number in zebrafish embryos",
        text=(
            "Step 3: Extract total genomic DNA from dissociated zebrafish "
            "embryo tissue using a column-based kit. Quantify mitochondrial "
            "DNA copy number by qPCR, amplifying the mt-nd1 mitochondrial "
            "gene and normalizing to the single-copy nuclear gene rag1. Run "
            "each sample in technical triplicate on the Bio-Rad CFX96 "
            "real-time PCR system using SYBR Green master mix. A 2-fold "
            "increase in mitochondrial DNA copy number is expected in "
            "treated embryos relative to untreated controls."
        ),
        tags=["protocol", "qpcr", "mitochondrial-dna-copy-number", "zebrafish"],
    )
    docs.append(planted)
    return docs
