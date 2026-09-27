"""Sanity-check DataPaper6 before running the full pipeline."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = Path(os.environ.get("PAPER6_ROOT", ROOT)) / "DataPaper6"
YEAR_MIN, YEAR_MAX = 2020, 2026

REQUIRED = [
    RAW / "NYPD_2020_2024" / "NYPD_Complaint_2020.csv",
    RAW / "NYPD_2020_2024" / "NYPD_Complaint_2021.csv",
    RAW / "NYPD_2020_2024" / "NYPD_Complaint_2022.csv",
    RAW / "NYPD_2020_2024" / "NYPD_Complaint_2023.csv",
    RAW / "NYPD_2020_2024" / "NYPD_Complaint_2024.csv",
    RAW / "Zillow&Redfin" / "Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv",
    RAW / "Open_Street_Map" / "new-york-260903.osm.pbf",
    RAW / "Open_Street_Map" / "illinois-260903.osm.pbf",
]


def find_acs() -> Path | None:
    if not RAW.exists():
        return None
    for p in RAW.iterdir():
        if p.is_dir() and p.name.startswith("2020_2024_ACS5"):
            hits = list(p.glob("*_tract.csv"))
            return hits[0] if hits else p
    return None


def find_chicago() -> list[Path]:
    d = RAW / "Chicago_311_Service_Requests"
    return sorted(d.glob("*.csv")) if d.exists() else []


def find_nyc_monthly() -> list[Path]:
    d = RAW / "NYC_311_Service_Requests" / "nyc311"
    if not d.exists():
        return []
    keep = []
    for f in sorted(d.glob("nyc311_????-??.csv.gz")):
        try:
            y = int(f.stem.split("_")[1][:4])
        except (IndexError, ValueError):
            continue
        if YEAR_MIN <= y <= YEAR_MAX:
            keep.append(f)
    return keep


def gb(path: Path) -> str:
    return f"{path.stat().st_size / 1e9:.2f} GB"


def main() -> int:
    print("RAW =", RAW, "| exists =", RAW.exists())
    missing = [p for p in REQUIRED if not p.exists()]
    acs = find_acs()
    chi = find_chicago()
    monthly = find_nyc_monthly()

    if acs is None or (acs.is_dir() and not list(acs.glob("*_tract.csv"))):
        missing.append(RAW / "2020_2024_ACS5a_*_tract.csv")
    if len(chi) < 12:
        missing.append(RAW / "Chicago_311_Service_Requests" / f"(found {len(chi)} csv, need 12)")
    if not monthly:
        missing.append(RAW / "NYC_311_Service_Requests" / "nyc311" / "nyc311_YYYY-MM.csv.gz")

    if missing:
        print("MISSING:")
        for p in missing:
            print("  -", p)
    else:
        print("required files: all present")

    print("  ACS     ", acs)
    print("  Chicago ", len(chi), "csv")
    print("  NYC 311 ", len(monthly), "monthly gz",
          f"({sum(p.stat().st_size for p in monthly) / 1e9:.2f} GB compressed)")
    if monthly:
        years = sorted({int(p.stem.split("_")[1][:4]) for p in monthly})
        print("  years   ", years[0], "–", years[-1], "| files", monthly[0].name, "…", monthly[-1].name)

    if not monthly:
        print("FAIL: no monthly NYC 311 extracts for 2020–present.")
        return 1
    print("OK: NYC 311 monthly extracts cover the study window.")
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
