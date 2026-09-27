# Version 1.0.0 — initial reproducibility release

This is the first archived release of the Broken Windows 311 replication materials.

## Included

- Canonical Python preparation and analysis scripts.
- Three cleaned Jupyter notebooks with machine-specific outputs removed.
- NYC primary-analysis and Chicago replication/robustness workflows.
- Machine-readable CSV and JSON result tables.
- Generated result figures.
- Dependency specification and runtime helpers.
- Derived-data schema and row-count metadata.
- Raw-data layout and public-source instructions.
- Zenodo and software-citation metadata.

## Excluded by design

- Manuscript, submission files, reviewer reports, and author notes.
- Raw 311, crime, census, housing, boundary, and OpenStreetMap files.
- Analysis-ready event-level and tract-month Parquet datasets.
- Virtual environments, caches, checkpoints, temporary logs, and machine-specific paths.

## Reproducibility note

The archived result tables and figures are supplied for transparent inspection. Full reruns require the original public-source inputs arranged according to `data/RAW_DATA_LAYOUT.md`. Derived schemas are documented in `results/metadata/manifest.json`.
