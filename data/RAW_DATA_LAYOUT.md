# DataPaper6 — raw inputs (not in git)

Keep this folder name and these relative paths. The prep notebook looks them up from `PAPER6_ROOT/DataPaper6`.

The 14.7 GB file that shipped with the Windows copy (`311-service-requests-from-2010-to-present-001.csv`) is a **duplicated 2019 extract**. Unique keys repeat across the whole file; sampled dates are only Sep–Dec 2019. Do **not** copy it to the lab. Re-download 2020–2024.

## Layout

```
DataPaper6/
  311-service-requests-from-2010-to-present-001.csv          # NYC 311, MUST cover 2020–2024
  NYPD_2020_2024/
    NYPD_Complaint_2020.csv
    NYPD_Complaint_2021.csv
    NYPD_Complaint_2022.csv
    NYPD_Complaint_2023.csv
    NYPD_Complaint_2024.csv
  Chicago_311_Service_Requests/
    311-service-requests-abandoned-vehicles.csv
    311-service-requests-alley-lights-out.csv
    311-service-requests-garbage-carts.csv
    311-service-requests-graffiti-removal.csv
    311-service-requests-pot-holes-reported.csv
    311-service-requests-rodent-baiting.csv
    311-service-requests-sanitation-code-complaints.csv
    311-service-requests-street-lights-all-out.csv
    311-service-requests-street-lights-one-out.csv
    311-service-requests-tree-debris.csv
    311-service-requests-tree-trims.csv
    311-service-requests-vacant-and-abandoned-buildings-reported.csv
  2020_2024_ACS5a_American_Community_Survey_2020–2024_5-Year_Data/
    nhgis0001_ds272_20245_tract.csv
    nhgis0001_ds272_20245_tract_codebook.txt
  Zillow&Redfin/
    Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv
    redfin_housing_market_monthly_top_50_cities_key_metrics_2023_Jan_to_2026_Jul.csv
  Open_Street_Map/
    new-york-260903.osm.pbf
    illinois-260903.osm.pbf
```

Copy NYPD, Chicago, ACS, Zillow/Redfin, and OSM with rsync/scp. Those files are large for git but fine on a lab disk.

## NYC 311 (required: 2020–2024)

Source: [311 Service Requests from 2010 to Present](https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2010-to-Present/erm2-nwe9) (dataset `erm2-nwe9`).

Save as `311-service-requests-from-2010-to-present-001.csv` so the prep notebook finds it. A filtered 2020–2024 extract is enough (typically several GB, not a 2019-only duplicate).

After the file lands, from the repo root:

```bash
python scripts/check_raw_data.py
```

The probe must list years inside 2020–2024. If it only reports 2019, stop — Prep would write an empty NYC panel.

## Study windows

| City | 311 window | Notes |
|---|---|---|
| NYC | 2020–2024 | Matches ACS 5-year vintage and the NYPD extract |
| Chicago | 2011–2019 (legacy files often end 2018) | RQ1–RQ2 replication only; no crime file |
