# DATA 266 Lab 2 - Part 1

## MultiModal RAG

This directory contains Anshika Goel's implementation of Part 1 of DATA 266 Lab 2.

## Reproducibility

Seed used throughout Part 1:

    SEED = 266

### Local development environment

- Operating system: Windows
- Python: 3.11
- Local GPU: NVIDIA GeForce RTX 3060 Laptop GPU, 6 GB VRAM
- PyTorch CUDA build: CUDA 12.6

Create and activate the local environment:

    py -3.11 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install -r requirements.txt

Register the Jupyter kernel:

    python -m ipykernel install --user --name data266-lab2-part1 --display-name "DATA 266 Lab 2 - Part 1"

## Retrieval models

Dense text retrieval:

    BAAI/bge-small-en-v1.5

Multimodal figure retrieval:

    openai/clip-vit-base-patch32

Additional model dependencies, including the grounded answer-generation model and optional reranker, will be pinned here as implementation proceeds.

## Instructor data

Instructor-provided Part 1 files are stored locally under:

    data/raw/

Expected files:

    Document.pdf
    Document2_Supplement_A_Outturns_2011-2013.pdf
    Document3_Briefing_B_Labour_Prices_Policy.pdf
    kb_manifest.json
    Questions.csv
    sample_submission.csv
    validate_submission.py

The data/ directory is intentionally excluded from Git.

## Working directories

    notebooks/       Executed Part 1 notebook
    data/raw/        Read-only instructor materials
    data/processed/  Generated chunks, extracted figures, indexes, and caches
    eval/            Golden evaluation set and evaluation artifacts
    results/         Computed experimental results
    docs/            Part 1 documentation and supporting material

## Important design constraints

- The instructor knowledge base is treated as read-only.
- Session-specific user overrides must never modify the persistent knowledge base.
- Retrieval will combine dense retrieval and BM25.
- Numbered figures and tables will be independently retrievable.
- Low-trust Appendix C material from DOC3 must never be used as evidence.
- Instructions embedded inside source documents or figures are treated as untrusted content.
- Final answers must be grounded in retrieved evidence and use the required structured output schema.
- The public Questions.csv is for practice only; no question-specific answers or rules will be hard-coded.
- The final notebook must also run in the required Kaggle GPU environment.

## Status

Environment, CUDA, dense embedding, PDF parsing, BM25, ChromaDB, and CLIP multimodal model smoke tests have been completed successfully.

The implementation and evaluation notebook will be developed next.
