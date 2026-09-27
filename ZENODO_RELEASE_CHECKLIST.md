# Zenodo release checklist

Version 1.0.0 has been published at https://doi.org/10.5281/zenodo.22988376 from GitHub release https://github.com/occbuu/broken-windows/releases/tag/V1.0.0.

## Before creating the GitHub release

- [ ] Review `.zenodo.json`, especially creator name, affiliation, ORCID, title, and version.
- [ ] Choose a license and add the corresponding license file(s) to the repository.
- [ ] Confirm that `.zenodo.json`, `CITATION.cff`, `ZENODO_METADATA.md`, and these release notes are committed to the default branch.
- [ ] Confirm that no manuscript, reviewer file, raw data, private path, credential, or large derived dataset has been added.
- [x] Create the Git tag `V1.0.0` from the exact commit to be archived.
- [x] Create the GitHub release titled `Version 1.0.0 — initial reproducibility release`.

## Zenodo–GitHub route

1. Sign in to Zenodo using the account that can access the GitHub repository.
2. Enable `occbuu/broken-windows` in the Zenodo GitHub integration.
3. Publish the GitHub release only after the integration is enabled.
4. Open the resulting Zenodo draft/record and verify every metadata field before publication.
5. Select the confirmed license and open-file visibility.
6. Publish the Zenodo record. Zenodo will assign the DOI at publication.

## Manual-upload route

1. Create a new Zenodo upload with resource type **Software**.
2. Upload the GitHub release ZIP or a clean ZIP of tag `v1.0.0`.
3. Copy the fields from `ZENODO_METADATA.md`.
4. Select the confirmed license and open-file visibility.
5. Use Zenodo's **Get a DOI now** option if the DOI must be inserted into repository files before publication.
6. Preview the record, then publish it.

## After Zenodo assigns the DOI

- [x] Add the DOI to `CITATION.cff` under `identifiers`.
- [x] Add a DOI badge and the preferred citation to `README.md`.
- [x] Add the DOI to the article's data/code availability statement.
- [x] Commit the post-publication DOI metadata to the Git repository.
- [x] Preserve the version-specific DOI for exact reproducibility.
