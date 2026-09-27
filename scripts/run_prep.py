#!/usr/bin/env python3
"""Paper 6 local prep — detached, resumable, CPU-bound.

Launch with `python scripts/run_job.py start prep` so the job keeps running
after Cursor is closed. Reopen Cursor and run `python scripts/run_job.py status`.
"""
from __future__ import annotations

import gc
import json
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from paper6_runtime import (  # noqa: E402
    RunStatus,
    cap_blas_threads,
    describe_compute,
    project_root,
)

cap_blas_threads()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import geopandas as gpd  # noqa: E402
import requests  # noqa: E402

# ---------------------------------------------------------------------------
YEAR_MIN, YEAR_MAX = 2020, 2026
NYC311_MONTHLY = Path("NYC_311_Service_Requests") / "nyc311"
CHI_YEAR_MIN, CHI_YEAR_MAX = 2011, 2019
COORD_DP = 4
NYC_COUNTIES = {"005", "047", "061", "081", "085"}
CHI_COUNTY = {"031"}
TIGER = {"36": "tl_2023_36_tract.zip", "17": "tl_2023_17_tract.zip"}
TIGER_BASE = "https://www2.census.gov/geo/tiger/TIGER2023/TRACT/"
TIGER_FTP_HOST = "ftp2.census.gov"
TIGER_FTP_DIR = "/geo/tiger/TIGER2023/TRACT"
UA = {"User-Agent": "Paper6-research/1.0 (academic; TIGER tract download)"}


def download_tiger_zip(fn: str, dst: Path) -> str:
    """HTTPS first; Census often 403s datacenter IPs, then FTP still works."""
    url = TIGER_BASE + fn
    try:
        r = requests.get(url, timeout=120, stream=True, headers=UA)
        if r.status_code == 200:
            with open(dst, "wb") as f:
                for blk in r.iter_content(1 << 20):
                    f.write(blk)
            return "https"
        print(f"  HTTPS {r.status_code} for {fn}; trying FTP...", flush=True)
    except Exception as exc:
        print(f"  HTTPS failed ({exc}); trying FTP...", flush=True)
    from ftplib import FTP
    ftp = FTP(TIGER_FTP_HOST, timeout=60)
    ftp.login()
    ftp.cwd(TIGER_FTP_DIR)
    with open(dst, "wb") as f:
        ftp.retrbinary(f"RETR {fn}", f.write, blocksize=1 << 20)
    ftp.quit()
    return "ftp"

DOMAIN_RULES = [
    ("physical_disorder", ["graffiti", "dirty condition", "illegal dumping", "derelict vehicle",
                           "abandoned vehicle", "street condition", "sidewalk condition", "curb condition",
                           "damaged tree", "dead tree", "overgrown tree", "vacant lot", "abandoned building",
                           "unsanitary pigeon", "litter basket", "dumpster", "broken parking meter",
                           "rodent", "standing water"]),
    ("social_disorder", ["noise", "drug activity", "illegal fireworks", "panhandling", "homeless",
                         "encampment", "disorderly youth", "drinking", "urinating in public",
                         "graffiti - vandalism", "illegal parking", "blocked driveway", "animal abuse",
                         "unlicensed vendor", "consumer complaint"]),
    ("housing", ["heat/hot water", "heating", "plumbing", "paint", "plaster", "door/window",
                 "electric", "flooring", "appliance", "general construction", "water leak",
                 "unsanitary condition", "mold", "elevator", "safety", "outside building"]),
    ("infrastructure", ["street light", "traffic signal", "water system", "sewer", "street sign",
                        "highway condition", "bridge", "root/sewer", "catch basin", "hydrant",
                        "traffic", "broken muni meter", "curb"]),
    ("sanitation", ["missed collection", "sanitation condition", "recycling", "electronics waste",
                    "commercial disposal", "residential disposal", "dumpster complaint",
                    "sweeping", "snow", "request large bulky item collection"]),
]
DOMAINS = ["physical_disorder", "social_disorder", "housing", "infrastructure", "sanitation", "other"]
DOM_ID = {d: i for i, d in enumerate(DOMAINS)}
_FMTS = ["%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
         "%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y"]
_DOM_CACHE: dict[str, str] = {}
_TRACTS_GDF = None


def chunk_size() -> int:
    env = os.environ.get("PAPER6_CHUNK")
    if env:
        return int(env)
    try:
        avail = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        avail = 8 * (1 << 30)
    return 800_000 if avail > 64 * (1 << 30) else 400_000


def paths():
    project = project_root()
    raw = project / "DataPaper6"
    derived = project / "derived"
    tmp = derived / "_tmp"
    geo = derived / "_geo"
    for p in (derived, tmp, geo):
        p.mkdir(parents=True, exist_ok=True)
    return project, raw, derived, tmp, geo


def norm(s) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(s).strip().lower()).strip("_")


def pick(cols_norm, *cands):
    for c in cands:
        if c in cols_norm:
            return cols_norm[c]
    return None


def hash64(series):
    return pd.util.hash_pandas_object(series.fillna("").astype(str), index=False).astype("int64")


def to_dt(s):
    s = s.astype(str)
    for fmt in _FMTS:
        out = pd.to_datetime(s, errors="coerce", format=fmt)
        if out.notna().mean() > 0.5:
            return out
    try:
        return pd.to_datetime(s, errors="coerce", format="mixed")
    except TypeError:
        return pd.to_datetime(s, errors="coerce")


def month_floor(dt_series):
    return dt_series.dt.to_period("M").dt.to_timestamp()


def done(path: Path) -> bool:
    if path.exists():
        print(f"[skip] {path.name} already exists ({path.stat().st_size / 1e6:.1f} MB)", flush=True)
        return True
    return False


def domain_of(ctype: str) -> str:
    t = ctype or ""
    if t in _DOM_CACHE:
        return _DOM_CACHE[t]
    tl = t.lower()
    res = "other"
    for dom, keys in DOMAIN_RULES:
        if any(k in tl for k in keys):
            res = dom
            break
    _DOM_CACHE[t] = res
    return res


def domain_ids(ct: pd.Series) -> np.ndarray:
    u = pd.Index(ct.unique())
    lut = pd.Series([DOM_ID[domain_of(x)] for x in u], index=u, dtype="int8")
    return ct.map(lut).fillna(DOM_ID["other"]).astype("int8").values


def channel_of(x) -> int:
    t = str(x or "").strip().lower()
    if t.startswith("phone"):
        return 0
    if t.startswith("online"):
        return 1
    if t.startswith("mobile"):
        return 2
    if t.startswith("unknown"):
        return 3
    return 4


def channel_ids(s: pd.Series) -> np.ndarray:
    u = pd.Index(s.astype(str).unique())
    lut = pd.Series([channel_of(x) for x in u], index=u, dtype="int8")
    return s.astype(str).map(lut).fillna(4).astype("int8").values


def tracts_gdf(geo: Path):
    global _TRACTS_GDF
    if _TRACTS_GDF is None:
        _TRACTS_GDF = gpd.read_file(geo / "tracts_ny_il.gpkg")[["GEOID", "geometry"]].to_crs("EPSG:4326")
    return _TRACTS_GDF


def coords_to_tract(unique_coords: pd.DataFrame, geo: Path) -> pd.DataFrame:
    f = 10 ** COORD_DP
    pts = gpd.GeoDataFrame(
        unique_coords.copy(),
        geometry=gpd.points_from_xy(unique_coords.lon_k / f, unique_coords.lat_k / f),
        crs="EPSG:4326",
    )
    j = gpd.sjoin(pts, tracts_gdf(geo), how="left", predicate="within")
    return j[["lat_k", "lon_k", "GEOID"]].drop_duplicates(subset=["lat_k", "lon_k"])


def estimate_csv_rows(path: Path) -> int:
    with path.open("rb") as f:
        f.readline()
        sample = f.read(8_000_000)
    n = sample.count(b"\n")
    avg = len(sample) / max(n, 1)
    return max(int(path.stat().st_size / avg), 1)


def probe_nyc_years(path: Path, n_windows: int = 7) -> set[int]:
    size = path.stat().st_size
    years: set[int] = set()
    with path.open("rb") as f:
        for i in range(n_windows):
            pos = int(size * i / max(n_windows - 1, 1))
            f.seek(pos)
            f.readline()
            for _ in range(80):
                line = f.readline()
                if not line:
                    break
                parts = line.split(b",", 2)
                if len(parts) < 2:
                    continue
                y = parts[1][:4]
                if y.isdigit():
                    years.add(int(y))
    return years


def find_nyc311_files(raw: Path) -> list[Path]:
    """Prefer monthly 2020–present extracts; ignore the 2019-only 14 GB dump."""
    monthly = sorted((raw / NYC311_MONTHLY).glob("nyc311_????-??.csv.gz"))
    keep = []
    for f in monthly:
        try:
            y = int(f.stem.split("_")[1][:4])
        except (IndexError, ValueError):
            continue
        if YEAR_MIN <= y <= YEAR_MAX:
            keep.append(f)
    if keep:
        return keep
    big = raw / "311-service-requests-from-2010-to-present-001.csv"
    if big.exists():
        years = probe_nyc_years(big)
        if any(YEAR_MIN <= y <= YEAR_MAX for y in years):
            return [big]
    return []


_SP_TF = None


def stateplane_to_wgs(x: pd.Series, y: pd.Series) -> tuple[pd.Series, pd.Series]:
    """NY Long Island State Plane (ftUS, EPSG:2263) → WGS84."""
    global _SP_TF
    xs = pd.to_numeric(x, errors="coerce")
    ys = pd.to_numeric(y, errors="coerce")
    ok = xs.notna() & ys.notna() & (xs != 0) & (ys != 0)
    lat = pd.Series(np.nan, index=x.index, dtype="float64")
    lon = pd.Series(np.nan, index=x.index, dtype="float64")
    if not ok.any():
        return lat, lon
    if _SP_TF is None:
        from pyproj import Transformer
        _SP_TF = Transformer.from_crs("EPSG:2263", "EPSG:4326", always_xy=True)
    lo, la = _SP_TF.transform(xs[ok].to_numpy(), ys[ok].to_numpy())
    lat.loc[ok] = la
    lon.loc[ok] = lo
    return lat, lon


def resolve_latlon(ch: pd.DataFrame, col: dict) -> tuple[pd.Series, pd.Series]:
    lat = (pd.to_numeric(ch[col["lat"]], errors="coerce")
           if col.get("lat") else pd.Series(np.nan, index=ch.index))
    lon = (pd.to_numeric(ch[col["lon"]], errors="coerce")
           if col.get("lon") else pd.Series(np.nan, index=ch.index))
    need = ~(lat.between(40.4, 41.1) & lon.between(-74.3, -73.6))
    if need.any() and col.get("x") and col.get("y"):
        la2, lo2 = stateplane_to_wgs(ch[col["x"]], ch[col["y"]])
        lat = lat.where(~need, la2)
        lon = lon.where(~need, lo2)
    return lat, lon


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------
def stage1_tiger(geo: Path, derived: Path, st: RunStatus) -> pd.DataFrame:
    st.update(stage="1_tiger", stage_index=1, progress_pct=4, message="Stage 1 — TIGER tract geometry")
    for _fips, fn in TIGER.items():
        dst = geo / fn
        if dst.exists():
            print("[cached]", fn, flush=True)
            continue
        print("downloading", fn, "...", flush=True)
        via = download_tiger_zip(fn, dst)
        print("  ->", round(dst.stat().st_size / 1e6, 1), "MB via", via, flush=True)

    out = derived / "tract_geo.parquet"
    if not done(out):
        parts = []
        for _fips, fn in TIGER.items():
            g = gpd.read_file(f"zip://{(geo / fn).as_posix()}")
            g = g[["GEOID", "STATEFP", "COUNTYFP", "TRACTCE", "ALAND", "AWATER", "INTPTLAT", "INTPTLON", "geometry"]]
            parts.append(g)
        tracts = pd.concat(parts, ignore_index=True)
        tracts = gpd.GeoDataFrame(tracts, geometry="geometry", crs="EPSG:4269").to_crs("EPSG:4326")
        tracts.to_file(geo / "tracts_ny_il.gpkg", driver="GPKG")
        flat = pd.DataFrame({
            "GEOID": tracts["GEOID"].astype(str),
            "STATEFP": tracts["STATEFP"].astype(str),
            "COUNTYFP": tracts["COUNTYFP"].astype(str),
            "ALAND": tracts["ALAND"].astype("float64"),
            "lat": tracts["INTPTLAT"].astype(float),
            "lon": tracts["INTPTLON"].astype(float),
        })
        flat["city"] = np.where(
            flat.STATEFP.eq("36") & flat.COUNTYFP.isin(NYC_COUNTIES), "NYC",
            np.where(flat.STATEFP.eq("17") & flat.COUNTYFP.isin(CHI_COUNTY), "CHI", "other"),
        )
        flat["area_km2"] = flat["ALAND"] / 1e6
        flat.to_parquet(out, index=False)
        print(flat.city.value_counts(), flush=True)
    return pd.read_parquet(out)


def stage2_acs(raw: Path, derived: Path, tract_geo: pd.DataFrame, st: RunStatus) -> None:
    st.update(stage="2_acs", stage_index=2, progress_pct=8, message="Stage 2 — ACS tract covariates")
    out = derived / "acs_tract.parquet"
    if done(out):
        return
    acs_dir = next(p for p in raw.iterdir() if p.is_dir() and p.name.startswith("2020_2024_ACS5"))
    src = next(acs_dir.glob("*_tract.csv"))
    keep_ctx = ["GISJOIN", "STATEA", "COUNTYA", "TRACTA", "STATE", "COUNTY", "TL_GEO_ID"]
    hdr = pd.read_csv(src, nrows=0)
    est = [c for c in hdr.columns if c.startswith(("AUO6E", "AUQ8E", "AUTWE", "AUUCE", "AUUDE", "AUUEE"))]
    a = pd.read_csv(src, usecols=keep_ctx + est, dtype={"STATEA": str, "COUNTYA": str, "TRACTA": str})
    a["GEOID"] = a.STATEA.str.zfill(2) + a.COUNTYA.str.zfill(3) + a.TRACTA.str.zfill(6)
    pop = a["AUO6E001"].astype(float)
    edu_tot = a["AUQ8E001"].astype(float)
    ba_plus = a[[f"AUQ8E{n:03d}" for n in (22, 23, 24, 25)]].astype(float).sum(axis=1)
    lt_hs = a[[f"AUQ8E{n:03d}" for n in range(2, 17)]].astype(float).sum(axis=1)
    lf = a["AUTWE003"].astype(float)
    unemp = a["AUTWE005"].astype(float)
    hu = a["AUUCE001"].astype(float)
    vac = a["AUUDE003"].astype(float)
    own = a["AUUEE002"].astype(float)
    rent = a["AUUEE003"].astype(float)
    acs = pd.DataFrame({
        "GEOID": a["GEOID"], "state": a["STATE"], "county": a["COUNTY"], "pop": pop,
        "pct_ba_plus": np.where(edu_tot > 0, 100 * ba_plus / edu_tot, np.nan),
        "pct_lt_hs": np.where(edu_tot > 0, 100 * lt_hs / edu_tot, np.nan),
        "pct_unemployed": np.where(lf > 0, 100 * unemp / lf, np.nan),
        "housing_units": hu,
        "pct_vacant": np.where(hu > 0, 100 * vac / hu, np.nan),
        "pct_owner_occ": np.where((own + rent) > 0, 100 * own / (own + rent), np.nan),
        "pct_renter_occ": np.where((own + rent) > 0, 100 * rent / (own + rent), np.nan),
    })
    acs = acs.merge(tract_geo[["GEOID", "area_km2", "city", "lat", "lon"]], on="GEOID", how="left")
    acs["pop_density_km2"] = acs["pop"] / acs["area_km2"].replace(0, np.nan)
    acs.to_parquet(out, index=False)
    print(acs.groupby("city").size(), flush=True)


def _skip_nyc311(st: RunStatus, reason: str) -> None:
    print("[skip] NYC 311 —", reason, flush=True)
    print("       Continuing with NYPD, Chicago 311, ACS, and housing.", flush=True)
    st.update(stage="3_nyc311", stage_index=3, progress_pct=55,
              message=f"[skip] NYC 311: {reason}",
              detail={"skipped": "nyc311", "reason": reason})


def _nyc311_colmap(src: Path) -> dict:
    hdr = pd.read_csv(src, nrows=0)
    cn = {norm(c): c for c in hdr.columns}
    return dict(
        uk=pick(cn, "unique_key"), created=pick(cn, "created_date"), closed=pick(cn, "closed_date"),
        agency=pick(cn, "agency"), ctype=pick(cn, "complaint_type"), descr=pick(cn, "descriptor"),
        zipc=pick(cn, "incident_zip"), addr=pick(cn, "incident_address", "street_name"),
        status=pick(cn, "status"), boro=pick(cn, "borough"),
        channel=pick(cn, "open_data_channel_type"), lat=pick(cn, "latitude"), lon=pick(cn, "longitude"),
        bbl=pick(cn, "bbl"), x=pick(cn, "x_coordinate_state_plane"),
        y=pick(cn, "y_coordinate_state_plane"),
    )


def _events_from_chunk(ch: pd.DataFrame, col: dict, td_counter: dict) -> pd.DataFrame:
    created = to_dt(ch[col["created"]])
    m = created.dt.year.between(YEAR_MIN, YEAR_MAX)
    ct_all = ch[col["ctype"]].fillna("").str.strip()
    de_all = (ch[col["descr"]].fillna("").str.strip()
              if col["descr"] else pd.Series([""] * len(ch), index=ch.index))
    vc = pd.DataFrame({"a": ct_all, "b": de_all}).value_counts()
    for k, v in vc.items():
        td_counter[k] = td_counter.get(k, 0) + int(v)

    ch = ch[m]
    created = created[m]
    if len(ch) == 0:
        return ch
    closed = to_dt(ch[col["closed"]]) if col["closed"] else pd.Series(pd.NaT, index=ch.index)
    lat, lon = resolve_latlon(ch, col)
    ok = lat.between(40.4, 41.1) & lon.between(-74.3, -73.6)
    f = 10 ** COORD_DP
    lat_k = np.where(ok, np.round(lat * f), np.nan)
    lon_k = np.where(ok, np.round(lon * f), np.nan)
    ct = ch[col["ctype"]].fillna("").str.strip()
    blank = pd.Series([""] * len(ch), index=ch.index)
    addr_src = (ch[col["bbl"]].fillna("") if col["bbl"] else blank)
    addr_src = addr_src.where(
        addr_src.str.len() > 5,
        (ch[col["addr"]].fillna("").str.upper().str.strip() if col["addr"] else blank),
    )
    addr_src = addr_src.where(
        addr_src.str.len() > 2,
        pd.Series(lat_k, index=ch.index).astype(str) + "_" + pd.Series(lon_k, index=ch.index).astype(str),
    )
    ev = pd.DataFrame({
        "created": created.values, "closed": closed.values,
        "lat_k": lat_k, "lon_k": lon_k, "ctype": ct.values, "domain": domain_ids(ct),
        "agency": (ch[col["agency"]].fillna("").values if col["agency"] else ""),
        "zipc": (pd.to_numeric(ch[col["zipc"]], errors="coerce").values if col["zipc"] else np.nan),
        "channel": (channel_ids(ch[col["channel"]]) if col["channel"] else np.int8(3)),
        "status": (ch[col["status"]].fillna("").values if col["status"] else ""),
        "boro": (ch[col["boro"]].fillna("").values if col["boro"] else ""),
        "addr_hash": hash64(addr_src).values,
    })
    for c in ("ctype", "agency", "boro", "status"):
        ev[c] = ev[c].astype("category")
    return ev


def stage3_nyc311(raw: Path, derived: Path, tmp: Path, geo: Path, st: RunStatus) -> None:
    out_ev = derived / "nyc311_events.parquet"
    out_td = derived / "nyc311_typedesc.parquet"
    if out_ev.exists() and out_td.exists():
        st.update(stage="3_nyc311", stage_index=3, progress_pct=55,
                  message=f"[skip] NYC 311 already built ({out_ev.stat().st_size / 1e6:.1f} MB)")
        return

    if os.environ.get("PAPER6_SKIP_NYC311", "").strip() in {"1", "true", "yes"}:
        _skip_nyc311(st, "PAPER6_SKIP_NYC311 is set")
        return

    files = find_nyc311_files(raw)
    if not files:
        _skip_nyc311(
            st,
            f"no monthly extracts in {NYC311_MONTHLY} and the 14 GB dump has no {YEAR_MIN}–{YEAR_MAX} dates",
        )
        return

    t0 = time.time()
    part_dir = tmp / "nyc311_parts"
    part_dir.mkdir(exist_ok=True)
    progress_path = tmp / "nyc311_progress.json"
    td_path = tmp / "nyc311_td.pkl"

    progress = {"source": "monthly", "done_files": [], "n_rows": 0, "n_kept": 0}
    td_counter: dict = {}
    if progress_path.exists() and td_path.exists():
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            with td_path.open("rb") as f:
                td_counter = pickle.load(f)
        except Exception:
            progress = {"source": "monthly", "done_files": [], "n_rows": 0, "n_kept": 0}
            td_counter = {}
        print(f"[resume] NYC 311: {len(progress.get('done_files', []))} files done "
              f"(read {progress.get('n_rows', 0):,}, kept {progress.get('n_kept', 0):,})", flush=True)
    else:
        for old in part_dir.glob("*.parquet"):
            old.unlink()

    done_files = set(progress.get("done_files") or [])
    n_rows = int(progress.get("n_rows", 0))
    n_kept = int(progress.get("n_kept", 0))
    n_files = len(files)
    st.update(stage="3_nyc311", stage_index=3, progress_pct=10,
              message=f"Stage 3 — NYC 311 monthly files ({n_files} months, {YEAR_MIN}–{YEAR_MAX})",
              detail={"n_files": n_files, "done_files": len(done_files),
                      "n_rows": n_rows, "n_kept": n_kept})

    for fi, src in enumerate(files):
        tag = src.name
        if tag in done_files and (part_dir / f"{src.stem}.parquet").exists():
            continue
        col = _nyc311_colmap(src)
        if col["created"] is None or col["ctype"] is None:
            raise RuntimeError(f"critical NYC 311 columns missing in {tag}")
        usecols = [v for v in col.values() if v]
        ch = pd.read_csv(src, usecols=usecols, dtype=str, low_memory=False, on_bad_lines="skip")
        n_rows += len(ch)
        ev = _events_from_chunk(ch, col, td_counter)
        if len(ev):
            ev.to_parquet(part_dir / f"{src.stem}.parquet", index=False)
            n_kept += len(ev)
        done_files.add(tag)
        pct = 10 + 40 * (len(done_files) / max(n_files, 1))
        msg = (f"{tag} ({fi + 1}/{n_files}) | read {n_rows / 1e6:6.2f}M | "
               f"kept {n_kept / 1e6:6.2f}M | {time.time() - t0:6.0f}s")
        st.update(progress_pct=pct, message=msg,
                  detail={"file": tag, "n_rows": n_rows, "n_kept": n_kept,
                          "done_files": len(done_files), "n_files": n_files})
        _save_nyc_progress(progress_path, td_path, 0, n_rows, n_kept, td_counter,
                           extra={"source": "monthly", "done_files": sorted(done_files)})
        del ev, ch
        gc.collect()

    print(f"pass 1 complete: {n_rows:,} rows read, {n_kept:,} in window — {time.time() - t0:.1f}s", flush=True)
    td = (pd.Series(td_counter).rename("n").rename_axis(["complaint_type", "descriptor"])
          .reset_index().sort_values("n", ascending=False))
    td["domain"] = [domain_of(x) for x in td.complaint_type]
    td.to_parquet(out_td, index=False)
    print("vocabulary:", td.shape, "| unique complaint_type:", td.complaint_type.nunique(), flush=True)

    st.update(progress_pct=52, message="Stage 3 — census tract lookup for unique coordinates")
    parts = sorted(part_dir.glob("*.parquet"))
    if not parts:
        raise RuntimeError("NYC 311 pass 1 wrote no chunks in the study window.")
    coords = pd.concat([pd.read_parquet(p, columns=["lat_k", "lon_k"]) for p in parts],
                       ignore_index=True).dropna().drop_duplicates()
    coords = coords.astype("int64").reset_index(drop=True)
    print("unique coordinates:", f"{len(coords):,}", flush=True)
    lut = coords_to_tract(coords, geo)
    lut.to_parquet(tmp / "nyc_coord_tract.parquet", index=False)

    st.update(progress_pct=54, message="Stage 3 — writing nyc311_events.parquet")
    import pyarrow as pa
    import pyarrow.parquet as pq
    writer = None
    matched = 0
    total = 0
    for p in parts:
        d = pd.read_parquet(p)
        d["lat_k"] = d["lat_k"].astype("Int64")
        d["lon_k"] = d["lon_k"].astype("Int64")
        d = d.merge(lut.astype({"lat_k": "Int64", "lon_k": "Int64"}), on=["lat_k", "lon_k"], how="left")
        for c in ("ctype", "agency", "boro", "status"):
            d[c] = d[c].astype(str)
        matched += int(d.GEOID.notna().sum())
        total += len(d)
        tbl = pa.Table.from_pandas(d, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_ev, tbl.schema, compression="zstd")
        writer.write_table(tbl)
        del d, tbl
        gc.collect()
    writer.close()
    print(f"events written: {total:,} rows, tract matched {matched / max(total, 1):.1%} — {time.time() - t0:.1f}s",
          flush=True)
    for p in parts:
        p.unlink()
    if progress_path.exists():
        progress_path.unlink()
    if td_path.exists():
        td_path.unlink()


def _save_nyc_progress(progress_path, td_path, next_chunk, n_rows, n_kept, td_counter,
                       extra: dict | None = None) -> None:
    payload = {"next_chunk": next_chunk, "n_rows": n_rows, "n_kept": n_kept,
               "updated": time.strftime("%Y-%m-%d %H:%M:%S")}
    if extra:
        payload.update(extra)
    tmp = progress_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(progress_path)
    with td_path.open("wb") as f:
        pickle.dump(td_counter, f, protocol=pickle.HIGHEST_PROTOCOL)


def stage4_nypd(raw: Path, derived: Path, geo: Path, st: RunStatus) -> None:
    st.update(stage="4_nypd", stage_index=4, progress_pct=62, message="Stage 4 — NYPD complaints")
    out = derived / "nyc_crime_events.parquet"
    if done(out):
        return
    t0 = time.time()
    frames = []
    files = sorted((raw / "NYPD_2020_2024").glob("*.csv"))
    f = 10 ** COORD_DP
    coord_seen: set = set()
    for src in files:
        h = pd.read_csv(src, nrows=3)
        cnn = {norm(c): c for c in h.columns}
        c_dt = pick(cnn, "cmplnt_fr_dt", "rpt_dt", "complaint_date")
        c_cat = pick(cnn, "law_cat_cd", "law_category")
        c_ofn = pick(cnn, "ofns_desc", "offense_description")
        c_lat = pick(cnn, "latitude")
        c_lon = pick(cnn, "longitude")
        cols = [c for c in (c_dt, c_cat, c_ofn, c_lat, c_lon) if c]
        d = pd.read_csv(src, usecols=cols, dtype=str, low_memory=False, on_bad_lines="skip")
        dt = to_dt(d[c_dt])
        lat = pd.to_numeric(d[c_lat], errors="coerce")
        lon = pd.to_numeric(d[c_lon], errors="coerce")
        ok = lat.between(40.4, 41.1) & lon.between(-74.3, -73.6) & dt.dt.year.between(YEAR_MIN, YEAR_MAX)
        d = pd.DataFrame({
            "created": dt[ok].values,
            "lat_k": np.round(lat[ok] * f).values, "lon_k": np.round(lon[ok] * f).values,
            "law_cat": d[c_cat][ok].fillna("").str.upper().values if c_cat else "",
            "ofns": d[c_ofn][ok].fillna("").str.upper().values if c_ofn else "",
        })
        frames.append(d)
        coord_seen.update(map(tuple, d[["lat_k", "lon_k"]].dropna().astype("int64").values))
        print(f"  {src.name}: {len(d):,} usable rows", flush=True)
    cr = pd.concat(frames, ignore_index=True)
    lut = coords_to_tract(pd.DataFrame(sorted(coord_seen), columns=["lat_k", "lon_k"]), geo)
    cr = cr.merge(lut, on=["lat_k", "lon_k"], how="left")
    violent = ("ASSAULT", "ROBBERY", "MURDER", "HOMICIDE", "RAPE", "FELONY ASSAULT", "STRANGULATION",
               "KIDNAPPING", "ARSON", "WEAPONS")
    property_ = ("BURGLARY", "GRAND LARCENY", "PETIT LARCENY", "CRIMINAL MISCHIEF", "THEFT",
                 "POSSESSION OF STOLEN", "FORGERY", "FRAUD")
    cr["violent"] = cr.ofns.str.contains("|".join(violent), na=False).astype("int8")
    cr["property"] = cr.ofns.str.contains("|".join(property_), na=False).astype("int8")
    cr["ofns"] = cr["ofns"].astype("category")
    cr["law_cat"] = cr["law_cat"].astype("category")
    cr.to_parquet(out, index=False)
    print(f"crime events: {len(cr):,}, tract matched {cr.GEOID.notna().mean():.1%} — {time.time() - t0:.1f}s",
          flush=True)


def chi_domain(fname: str) -> str:
    s = fname.lower()
    if "graffiti" in s or "abandoned-vehicles" in s or "vacant" in s or "tree-debris" in s or "rodent" in s:
        return "physical_disorder"
    if "sanitation" in s or "garbage" in s:
        return "sanitation"
    return "infrastructure"


def stage5_chicago(raw: Path, derived: Path, geo: Path, st: RunStatus) -> None:
    st.update(stage="5_chicago", stage_index=5, progress_pct=72, message="Stage 5 — Chicago 311")
    out = derived / "chi311_events.parquet"
    if done(out):
        return
    t0 = time.time()
    frames = []
    coord_seen: set = set()
    f = 10 ** COORD_DP
    for src in sorted((raw / "Chicago_311_Service_Requests").glob("*.csv")):
        h = pd.read_csv(src, nrows=3)
        cnn = {norm(c): c for c in h.columns}
        c_cr = pick(cnn, "creation_date", "date_service_request_was_received")
        c_cm = pick(cnn, "completion_date")
        c_ty = pick(cnn, "type_of_service_request", "service_request_type")
        c_st = pick(cnn, "status")
        c_zp = pick(cnn, "zip_code", "zip_code_1")
        c_ca = pick(cnn, "community_area")
        c_wd = pick(cnn, "ward")
        c_ad = pick(cnn, "street_address", "address_street_name")
        c_la = pick(cnn, "latitude")
        c_lo = pick(cnn, "longitude")
        cols = [c for c in (c_cr, c_cm, c_ty, c_st, c_zp, c_ca, c_wd, c_ad, c_la, c_lo) if c]
        d = pd.read_csv(src, usecols=cols, dtype=str, low_memory=False, on_bad_lines="skip")
        cr = to_dt(d[c_cr])
        cm = to_dt(d[c_cm]) if c_cm else pd.Series(pd.NaT, index=d.index)
        la = pd.to_numeric(d[c_la], errors="coerce") if c_la else pd.Series(np.nan, index=d.index)
        lo = pd.to_numeric(d[c_lo], errors="coerce") if c_lo else pd.Series(np.nan, index=d.index)
        ok = cr.dt.year.between(CHI_YEAR_MIN, CHI_YEAR_MAX)
        okxy = la.between(41.6, 42.1) & lo.between(-87.95, -87.5)
        addr = (d[c_ad].fillna("").str.upper().str.strip() if c_ad else pd.Series("", index=d.index))
        e = pd.DataFrame({
            "created": cr[ok].values, "closed": cm[ok].values,
            "lat_k": np.where(okxy[ok], np.round(la[ok] * f), np.nan),
            "lon_k": np.where(okxy[ok], np.round(lo[ok] * f), np.nan),
            "srtype": (d[c_ty][ok].fillna("").values if c_ty else src.stem),
            "domain": chi_domain(src.name),
            "status": (d[c_st][ok].fillna("").values if c_st else ""),
            "zipc": (pd.to_numeric(d[c_zp][ok], errors="coerce").values if c_zp else np.nan),
            "comm_area": (pd.to_numeric(d[c_ca][ok], errors="coerce").values if c_ca else np.nan),
            "ward": (pd.to_numeric(d[c_wd][ok], errors="coerce").values if c_wd else np.nan),
            "addr_hash": hash64(addr[ok]).values, "src": src.stem,
        })
        frames.append(e)
        coord_seen.update(map(tuple, e[["lat_k", "lon_k"]].dropna().astype("int64").values))
        print(f"  {src.name:<62} {len(e):>9,} rows", flush=True)
    chi = pd.concat(frames, ignore_index=True)
    lut = coords_to_tract(pd.DataFrame(sorted(coord_seen), columns=["lat_k", "lon_k"]), geo)
    chi = chi.merge(lut, on=["lat_k", "lon_k"], how="left")
    for c in ("srtype", "domain", "status", "src"):
        chi[c] = chi[c].astype("category")
    chi.to_parquet(out, index=False)
    print(f"chicago events: {len(chi):,}, tract matched {chi.GEOID.notna().mean():.1%} — {time.time() - t0:.1f}s",
          flush=True)


def stage6_housing(raw: Path, derived: Path, st: RunStatus) -> None:
    st.update(stage="6_housing", stage_index=6, progress_pct=80, message="Stage 6 — Zillow / Redfin")
    out = derived / "zhvi_zip_month.parquet"
    if not done(out):
        src = next((raw / "Zillow&Redfin").glob("Zip_zhvi*.csv"))
        z = pd.read_csv(src, dtype={"RegionName": str}, low_memory=False)
        val_cols = [c for c in z.columns if c[:4].isdigit()]
        keep = z[z.State.isin(["NY", "IL"])]
        zl = keep.melt(id_vars=[c for c in ("RegionName", "State", "City", "CountyName") if c in keep.columns],
                       value_vars=val_cols, var_name="date", value_name="zhvi")
        zl["date"] = pd.to_datetime(zl["date"], errors="coerce")
        zl = zl.dropna(subset=["zhvi", "date"])
        zl["month"] = month_floor(zl["date"])
        zl["zipc"] = pd.to_numeric(zl["RegionName"], errors="coerce")
        zl = zl[["zipc", "State", "City", "CountyName", "month", "zhvi"]]
        zl.to_parquet(out, index=False)
    print(pd.read_parquet(out).shape, flush=True)

    out = derived / "redfin_city_month.parquet"
    if not done(out):
        src = next((raw / "Zillow&Redfin").glob("redfin_*.csv"))
        r = pd.read_csv(src)
        r.columns = [norm(c) for c in r.columns]
        r["month"] = month_floor(pd.to_datetime(r["period_begin"], errors="coerce"))
        r.to_parquet(out, index=False)
    print(pd.read_parquet(out).shape, flush=True)


def build_panel(ev, unit="GEOID", label="nyc"):
    ev = ev.dropna(subset=[unit]).copy()
    ev["month"] = month_floor(ev["created"])
    ev["days_to_close"] = (ev["closed"] - ev["created"]).dt.total_seconds() / 86400
    ev.loc[(ev.days_to_close < 0) | (ev.days_to_close > 365), "days_to_close"] = np.nan
    ev = ev.sort_values(["addr_hash", "created"])
    prev = ev.groupby("addr_hash", observed=True)["created"].shift(1)
    gap = (ev["created"] - prev).dt.total_seconds() / 86400
    ev["is_repeat_12m"] = ((gap <= 365) & gap.notna()).astype("int8")
    prev_d = ev.groupby(["addr_hash", "domain"], observed=True)["created"].shift(1)
    gapd = (ev["created"] - prev_d).dt.total_seconds() / 86400
    ev["is_same_domain_repeat"] = ((gapd <= 365) & gapd.notna()).astype("int8")
    d = ev["days_to_close"]
    ev["_closed3"] = (d <= 3).astype("float32").where(d.notna())
    ev["_open30"] = (d > 30).astype("float32").where(d.notna())
    ev["_unres"] = d.isna().astype("float32")
    g = ev.groupby([unit, "month"], observed=True)
    panel = g.agg(
        n_complaints=("created", "size"), n_addresses=("addr_hash", "nunique"),
        median_days_close=("days_to_close", "median"), mean_days_close=("days_to_close", "mean"),
        share_closed_3d=("_closed3", "mean"), share_open_30d=("_open30", "mean"),
        share_unresolved=("_unres", "mean"), n_repeat_12m=("is_repeat_12m", "sum"),
        n_same_domain_rep=("is_same_domain_repeat", "sum"),
    ).reset_index()
    p90 = g["days_to_close"].quantile(0.90).rename("p90_days_close").reset_index()
    panel = panel.merge(p90, on=[unit, "month"], how="left")
    dom = ev["domain"]
    if pd.api.types.is_numeric_dtype(dom):
        dom = dom.map({i: dname for dname, i in DOM_ID.items()})
    ev["_dom"] = dom.astype(str)
    wide = (ev.groupby([unit, "month", "_dom"], observed=True).size()
            .unstack("_dom", fill_value=0).add_prefix("n_").reset_index())
    panel = panel.merge(wide, on=[unit, "month"], how="left")
    if "channel" in ev.columns:
        ch = (ev.assign(selfserve=ev["channel"].isin([1, 2]).astype("int8"))
              .groupby([unit, "month"], observed=True)["selfserve"].mean()
              .rename("share_selfserve").reset_index())
        panel = panel.merge(ch, on=[unit, "month"], how="left")
    cnt = ev.groupby([unit, "month", "addr_hash"], observed=True).size().rename("k").reset_index()
    hhi = (cnt.assign(sh=lambda d: d.k / d.groupby([unit, "month"])["k"].transform("sum"))
           .assign(sh2=lambda d: d.sh ** 2)
           .groupby([unit, "month"])["sh2"].sum().rename("addr_hhi").reset_index())
    panel = panel.merge(hhi, on=[unit, "month"], how="left")
    panel["repeat_rate"] = panel.n_repeat_12m / panel.n_complaints
    panel["same_domain_rate"] = panel.n_same_domain_rep / panel.n_complaints
    panel["complaints_per_address"] = panel.n_complaints / panel.n_addresses
    for c in [c for c in panel.columns if c.startswith("n_") and c not in
              ("n_complaints", "n_addresses", "n_repeat_12m", "n_same_domain_rep")]:
        panel[c.replace("n_", "share_", 1)] = panel[c] / panel.n_complaints
    panel["city_label"] = label
    return panel


def stage7_panels(derived: Path, st: RunStatus) -> None:
    st.update(stage="7_panels", stage_index=7, progress_pct=88, message="Stage 7 — tract-month panels")
    out = derived / "nyc_tract_month.parquet"
    if not (derived / "nyc311_events.parquet").exists():
        print("[skip] nyc_tract_month.parquet — no NYC 311 events (portal down / file skipped)", flush=True)
    elif not done(out):
        t0 = time.time()
        ev = pd.read_parquet(derived / "nyc311_events.parquet")
        panel = build_panel(ev, unit="GEOID", label="nyc")
        cr = pd.read_parquet(derived / "nyc_crime_events.parquet").dropna(subset=["GEOID"])
        cr["month"] = month_floor(cr["created"])
        cg = cr.groupby(["GEOID", "month"]).agg(
            n_crime=("created", "size"), n_violent=("violent", "sum"), n_property=("property", "sum"),
            n_felony=("law_cat", lambda s: (s == "FELONY").sum()),
        ).reset_index()
        panel = panel.merge(cg, on=["GEOID", "month"], how="left")
        zmap = (ev.dropna(subset=["GEOID", "zipc"]).groupby(["GEOID", "zipc"]).size()
                .rename("k").reset_index().sort_values("k", ascending=False)
                .drop_duplicates("GEOID")[["GEOID", "zipc"]])
        panel = panel.merge(zmap, on="GEOID", how="left")
        zh = pd.read_parquet(derived / "zhvi_zip_month.parquet")[["zipc", "month", "zhvi"]]
        panel = panel.merge(zh, on=["zipc", "month"], how="left")
        panel = panel.merge(pd.read_parquet(derived / "acs_tract.parquet"), on="GEOID", how="left")
        panel = panel[panel.city.eq("NYC")]
        panel.to_parquet(out, index=False)
        print(f"NYC panel: {panel.shape} — {time.time() - t0:.1f}s", flush=True)
        del ev
        gc.collect()

    out = derived / "chi_tract_month.parquet"
    if not done(out):
        chi = pd.read_parquet(derived / "chi311_events.parquet")
        p = build_panel(chi, unit="GEOID", label="chi")
        p = p.merge(pd.read_parquet(derived / "acs_tract.parquet"), on="GEOID", how="left")
        p = p[p.city.eq("CHI")]
        p.to_parquet(out, index=False)
        print("Chicago panel:", p.shape, flush=True)

    out = derived / "nyc_agency_month.parquet"
    if not (derived / "nyc311_events.parquet").exists():
        print("[skip] nyc_agency_month.parquet — no NYC 311 events", flush=True)
    elif not done(out):
        ev = pd.read_parquet(derived / "nyc311_events.parquet",
                             columns=["created", "closed", "agency", "GEOID", "domain"])
        ev["month"] = month_floor(ev["created"])
        ev["days"] = (ev["closed"] - ev["created"]).dt.total_seconds() / 86400
        ev.loc[(ev.days < 0) | (ev.days > 365), "days"] = np.nan
        ev["_c3"] = (ev.days <= 3).astype("float32").where(ev.days.notna())
        ev["_un"] = ev.days.isna().astype("float32")
        ag = ev.groupby(["agency", "month"], observed=True).agg(
            n=("created", "size"), median_days=("days", "median"),
            share_3d=("_c3", "mean"), share_unresolved=("_un", "mean"),
        ).reset_index()
        ag.to_parquet(out, index=False)
        del ev
        gc.collect()

    manifest = {}
    for p in sorted(derived.glob("*.parquet")):
        d = pd.read_parquet(p)
        manifest[p.name] = {"rows": int(len(d)), "cols": list(map(str, d.columns)),
                            "mb": round(p.stat().st_size / 1e6, 2)}
        del d
    skipped = []
    if not (derived / "nyc311_events.parquet").exists():
        skipped.append("nyc311")
    manifest["_meta"] = {
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "window_nyc": [YEAR_MIN, YEAR_MAX],
        "window_chi": [CHI_YEAR_MIN, CHI_YEAR_MAX],
        "skipped": skipped,
        "note": ("NYC 311 skipped — NYC Open Data erm2-nwe9 unreachable; "
                 "Chicago panel + NYPD + ACS + housing were still built") if skipped else "",
    }
    (derived / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    for k, v in manifest.items():
        if k != "_meta":
            print(f"{k:<34} {v['rows']:>10,} rows  {v['mb']:>8.1f} MB", flush=True)


def main() -> int:
    project, raw, derived, tmp, geo = paths()
    st = RunStatus("prep", project)
    device = describe_compute("prep")
    print("python :", sys.version.split()[0], flush=True)
    print("project:", project, flush=True)
    print("raw    :", raw, "exists =", raw.exists(), flush=True)
    print("device :", device["kind"], "—", device["reason"], flush=True)
    st.start(device=device, stage_total=7)
    try:
        if not raw.exists():
            raise FileNotFoundError(f"DataPaper6 not found at {raw}")
        tract_geo = stage1_tiger(geo, derived, st)
        stage2_acs(raw, derived, tract_geo, st)
        stage3_nyc311(raw, derived, tmp, geo, st)
        stage4_nypd(raw, derived, geo, st)
        stage5_chicago(raw, derived, geo, st)
        stage6_housing(raw, derived, st)
        stage7_panels(derived, st)
        note = ""
        if not (derived / "nyc311_events.parquet").exists():
            note = " NYC 311 skipped (Open Data down). Chicago + NYPD + ACS + housing are ready."
        st.done(f"DONE.{note} Outputs in {derived}")
        print("DONE." + note, "Outputs:", derived, flush=True)
        return 0
    except Exception as exc:
        st.fail(exc)
        raise


if __name__ == "__main__":
    sys.exit(main())
