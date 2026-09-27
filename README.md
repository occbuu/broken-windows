# Broken Windows 311 replication materials

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22988376.svg)](https://doi.org/10.5281/zenodo.22988376)

This repository contains **code and generated results only** for the NYC–Chicago 311 analysis. It intentionally excludes the manuscript, reviewer files, raw microdata, analysis-ready Parquet files, virtual environments, caches, and temporary logs.

## Repository contents

| Path | Contents |
|---|---|
| `Paper6_00_Prep_LocalData.ipynb` | Documented raw-data preparation workflow |
| `Paper6_Analysis.ipynb` | Main NYC analysis and Chicago replication entry point |
| `Paper6_Colab_GeoAI_OSM.ipynb` | OSM feature engineering and optional GraphSAGE analysis |
| `scripts/` | Canonical preparation, analysis, robustness, runtime, and job-control code |
| `results/tables/` | Machine-readable CSV and JSON result tables |
| `results/figures/` | Generated figures used to inspect and report the analyses |
| `results/metadata/manifest.json` | Row counts, variables, sizes, and study-window metadata for derived datasets |
| `results/metadata/gnn_results.json` | Saved GeoAI model metrics |
| `data/RAW_DATA_LAYOUT.md` | Expected raw-data layout and public-source instructions |
| `RESULTS_GUIDE.md` | Map from hypotheses and analyses to output files |

## What is not included

- No paper, draft, appendix, review report, or submission document.
- No raw 311, crime, ACS, Zillow/Redfin, TIGER, or OpenStreetMap files.
- No derived event-level or tract-month Parquet files.
- No personally identifying addresses. The pipeline hashes address strings before tract-level analysis.

The omitted inputs are too large for Git and, where applicable, remain governed by their original data providers. `data/RAW_DATA_LAYOUT.md` identifies the required files and source locations. `results/metadata/manifest.json` documents the derived schemas without distributing the observations.

## Environment

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Optional GeoAI dependencies (`pyrosm`, `torch`, and `torch-geometric`) are listed at the end of `requirements.txt`. The main statistical analysis does not require a GPU. LightGBM and GraphSAGE use a compatible GPU when available and otherwise fall back to CPU.

## Reproduction workflow

1. Download the raw inputs and arrange them as described in `data/RAW_DATA_LAYOUT.md`.
2. Set the repository root if running outside this directory:

   ```bash
   export PAPER6_ROOT="$PWD"     # Windows PowerShell: $env:PAPER6_ROOT = (Get-Location).Path
   ```

3. Validate the raw inputs:

   ```bash
   python scripts/check_raw_data.py
   ```

4. Run the preparation and primary analysis:

   ```bash
   python scripts/run_prep.py
   python scripts/run_analysis.py
   ```

5. Run the optional GeoAI component and Chicago robustness suite:

   ```bash
   python scripts/run_geoai.py
   python scripts/analysis_chicago_v1_robust.py
   ```

For a staged end-to-end run, use `python scripts/run_pipeline.py`. Long jobs can be managed with `python scripts/run_job.py --help`. Generated intermediate datasets are written to `derived/`; tables and figures are written to `tables/` and `figures/` by the original pipeline. The archived publication outputs are under `results/` in this repository.

## Verification

Start with:

- `results/tables/T00_verification.csv` for automated consistency checks;
- `results/tables/results_headline.json` for headline sample and model quantities;
- `results/tables/chicago_v1/results_headline_robust.json` for the Chicago robustness run;
- `results/metadata/manifest.json` for derived-data provenance.

All notebooks in this archive have their execution output cleared so that the repository contains code rather than machine-specific logs or paths.

## Archived release and citation

The immutable Version 1.0.0 archive is available from Zenodo:

- DOI: [10.5281/zenodo.22988376](https://doi.org/10.5281/zenodo.22988376)
- GitHub release: [V1.0.0](https://github.com/occbuu/broken-windows/releases/tag/V1.0.0)

Preferred citation:

> Le, N. H. (2026). *Broken Windows 311: Reproducible code and results for NYC and Chicago municipal service-request analyses* (Version 1.0.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.22988376

