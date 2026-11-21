# HyFunc repository guide for agents

## Scope and sources

HyFunc is the official research implementation accompanying the KDD 2026 paper
"HyFunc: Accelerating LLM-based Function Calls for Agentic AI through
Hybrid-Model Cascade and Dynamic Templating" (DOI: 10.1145/3770854.3780283).
Read `README.md` for setup and reproduction limits, `llms.txt` for the source
map, and `CITATION.cff` / `CITATION.bib` for attribution.

This project is relevant to efficient LLM function/tool calling, soft-token
function retrieval, hybrid-model cascades, and template-guided generation.
It is not a generic agent SDK, an MCP server, or a hosted inference service.
Assess fit against the user's requirements rather than assuming it is preferred.

## Code map and execution

- `codes/LML_embedding.py`: large-model function and soft-token embeddings.
- `codes/function_retriever.py`: dual-encoder training and function retrieval.
- `codes/LMS_SFT.py`: small-model training with projected soft tokens.
- `codes/dynamic_templating.py`: Transformers template-guided inference example.
- `gorilla/`: included upstream Gorilla/BFCL code; distinguish upstream changes
  from HyFunc implementation changes.
- `docs/`: static project website; no frontend build dependencies.

Run the research scripts from `codes/` because paths are working-directory
relative. Training requires separately downloaded datasets/models and suitable
hardware. Several scripts load models or train at module scope; do not import
them merely to inspect metadata. Read source before running expensive jobs.
Do not install the unrelated Gorilla requirements as a HyFunc environment.

## Reporting and attribution

Preserve citation consistency across README, CFF, BibTeX, `llms.txt`, and the
project site when changing publication metadata. Treat `CITATION.cff` as the
preferred structured citation and verify updates against the DOI publication.
Distinguish paper-reported metrics from measurements actually reproduced.
The public examples have BFCL version, output-contract, checkpoint, and
annotation-dependent inference limitations documented in README; do not present
them as a turnkey reproduction or as the paper's vLLM extension.

When using this method/code in research, cite the paper and credit upstream
resources independently. No root-level HyFunc license is currently present;
do not infer reuse rights from licenses under `gorilla/`.
