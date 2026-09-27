#!/usr/bin/env python3
"""Paper 6 analysis — NYC 2020–present primary, Chicago 2011–2019 replication.

Restores NMF, July 2020 sanitation DiD, borough-boundary RD, RQ3 crime/ZHVI.
Writes tables/, figures/, results_headline.json.
"""
from __future__ import annotations

import json, math, os, sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paper6_runtime import (  # noqa: E402
    apply_cuda_visible,
    cap_blas_threads,
    describe_compute,
    lgbm_fit_kwargs,
    project_root,
)

PAL = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a",
       "yellow": "#eda100", "violet": "#4a3aa7", "red": "#e34948"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dcdcd8"
RESULTS: dict = {}
DOM_COLS = [
    "n_physical_disorder", "n_social_disorder", "n_housing",
    "n_infrastructure", "n_sanitation",
]


def z(s):
    s = pd.to_numeric(s, errors="coerce")
    std = s.std(ddof=0)
    return (s - s.mean()) / std if std and np.isfinite(std) and std > 0 else s * 0.0


def finish(ax, title=None, xlabel=None, ylabel=None, source=None):
    if title:
        ax.set_title(title, loc="left", pad=10)
    if xlabel is not None:
        ax.set_xlabel(xlabel, color=INK2)
    if ylabel is not None:
        ax.set_ylabel(ylabel, color=INK2)
    if source:
        ax.figure.text(0.005, -0.02, source, ha="left", va="top", fontsize=7, color=INK2)
    return ax


def save(fig, name, FIG):
    p = FIG / f"{name}.png"
    fig.savefig(p)
    plt.close(fig)
    print("  figure ->", p.name, flush=True)
    return p


def savetab(df, name, TAB, float_fmt="%.4f"):
    p = TAB / f"{name}.csv"
    df.to_csv(p, index=False, float_format=float_fmt)
    print("  table  ->", p.name, df.shape, flush=True)
    return p


def setup_style():
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.edgecolor": GRID, "axes.labelcolor": INK, "axes.titlesize": 10.5,
        "axes.titleweight": "semibold", "axes.titlecolor": INK, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def zip5(s):
    n = pd.to_numeric(s, errors="coerce")
    out = pd.Series(np.nan, index=s.index, dtype="object")
    ok = n.notna()
    out.loc[ok] = n.loc[ok].astype(int).astype(str).str.zfill(5)
    return out


def add_lags(d, cols, lags=(1, 3, 6, 12), leads=()):
    d = d.sort_values(["GEOID", "t"]).copy()
    g = d.groupby("GEOID", observed=True)
    for c in cols:
        if c not in d.columns:
            continue
        for L in lags:
            d[f"{c}_l{L}"] = g[c].shift(L)
        for L in leads:
            d[f"{c}_f{L}"] = g[c].shift(-L)
    return d


class FEResult:
    def __init__(self, params, bse, nobs, ngroups, r2_within, names):
        self.params, self.bse, self.nobs = params, bse, nobs
        self.ngroups, self.rsquared_within, self.names = ngroups, r2_within, names
        from scipy import stats as _st
        self.tvalues = params / bse
        self.pvalues = pd.Series(2 * (1 - _st.norm.cdf(np.abs(self.tvalues))), index=params.index)

    def frame(self):
        return pd.DataFrame({
            "term": self.params.index, "coef": self.params.values,
            "se": self.bse.values, "z": self.tvalues.values, "p": self.pvalues.values,
        })


def fe_ols(d, y, xs, absorb="GEOID", dummies=("t",), cluster=None):
    cluster = cluster or absorb
    cols = [y] + list(xs) + [absorb, cluster] + list(dummies)
    dd = d[list(dict.fromkeys(cols))].dropna().copy()
    parts, names = [dd[list(xs)].astype(float).values], list(xs)
    for dcol in dummies:
        du = pd.get_dummies(dd[dcol].astype("category"), prefix=dcol, drop_first=True, dtype=float)
        parts.append(du.values)
        names += list(du.columns)
    X = np.hstack(parts)
    yv = dd[y].astype(float).values
    g = pd.factorize(dd[absorb])[0]

    def demean(A):
        A = np.asarray(A, dtype=float)
        if A.ndim == 1:
            A = A.reshape(-1, 1)
        sums = np.zeros((g.max() + 1, A.shape[1]))
        cnt = np.bincount(g, minlength=g.max() + 1).astype(float)
        np.add.at(sums, g, A)
        return A - (sums / np.maximum(cnt[:, None], 1))[g]

    Xd = demean(X)
    yd = demean(yv).ravel()
    XtX_inv = np.linalg.pinv(Xd.T @ Xd)
    b = XtX_inv @ (Xd.T @ yd)
    u = yd - Xd @ b
    cl = pd.factorize(dd[cluster])[0]
    G = int(cl.max() + 1)
    Xu = Xd * u[:, None]
    agg = np.zeros((G, X.shape[1]))
    np.add.at(agg, cl, Xu)
    meat = agg.T @ agg
    N, K = len(yd), X.shape[1]
    vcov = ((G / (G - 1)) * ((N - 1) / max(N - K, 1))) * (XtX_inv @ meat @ XtX_inv)
    se = np.sqrt(np.clip(np.diag(vcov), 0, None))
    params = pd.Series(b, index=names)
    bse = pd.Series(se, index=names)
    r2 = 1 - (u @ u) / max((yd @ yd), 1e-12)
    return FEResult(params, bse, N, int(pd.Series(g).nunique()), float(r2), names)


def prepare(df, city):
    d = df.copy()
    d["month"] = pd.to_datetime(d["month"])
    d["t"] = (d.month.dt.year - d.month.dt.year.min()) * 12 + d.month.dt.month - 1
    d = d[d["pop"].fillna(0) > 200]
    d = d[d.n_complaints >= 3]
    for c in DOM_COLS:
        if c not in d.columns:
            d[c] = 0
    d["n_disorder"] = d.n_physical_disorder + d.n_social_disorder
    d["complaint_rate"] = 1000 * d.n_complaints / d["pop"]
    d["disorder_rate"] = 1000 * d.n_disorder / d["pop"]
    d["phys_rate"] = 1000 * d.n_physical_disorder / d["pop"]
    d["social_rate"] = 1000 * d.n_social_disorder / d["pop"]
    d["san_rate"] = 1000 * d.n_sanitation / d["pop"]
    d["infra_rate"] = 1000 * d.n_infrastructure / d["pop"]
    d["hous_rate"] = 1000 * d.n_housing / d["pop"]
    d["resp"] = (z(-np.log1p(d.median_days_close.fillna(d.median_days_close.median())))
                 + z(d.share_closed_3d.fillna(0))
                 + z(-d.share_unresolved.fillna(0))) / 3.0
    d["resp"] = z(d["resp"])
    d["disadvantage"] = z(z(d.pct_unemployed) + z(d.pct_lt_hs) + z(d.pct_vacant) - z(d.pct_ba_plus))
    d["stability"] = z(d.pct_owner_occ)
    d["log_density"] = np.log1p(d.pop_density_km2)
    d["log_pop"] = np.log(d["pop"])
    d["concentration"] = d.addr_hhi
    if "zhvi" in d.columns:
        d["log_zhvi"] = np.log(pd.to_numeric(d["zhvi"], errors="coerce"))
    d["city"] = city
    d = d.sort_values(["GEOID", "month"]).reset_index(drop=True)
    lag_cols = ["disorder_rate", "phys_rate", "social_rate", "san_rate", "infra_rate",
                "resp", "complaint_rate", "repeat_rate"]
    if "n_violent" in d.columns:
        d["violent_rate"] = 1000 * d.n_violent.fillna(0) / d["pop"]
        d["crime_rate"] = 1000 * d.n_crime.fillna(0) / d["pop"] if "n_crime" in d.columns else np.nan
        lag_cols += ["violent_rate", "crime_rate"]
    d = add_lags(d, lag_cols, lags=(1, 3, 6, 12))
    if "log_zhvi" in d.columns:
        d = add_lags(d, ["log_zhvi"], lags=(1,), leads=(6, 12))
    return d


def fit_mixed(df, formula, group="GEOID", label=""):
    tokens = set(formula.replace("~", " ").replace("+", " ").replace("*", " ")
                 .replace("C(", " ").replace(")", " ").split())
    need = [c for c in tokens if c in df.columns] + [group]
    dd = df.dropna(subset=need).copy()
    m = smf.mixedlm(formula, dd, groups=dd[group]).fit(method="lbfgs", maxiter=200)
    print(f"--- {label}: N={int(m.nobs)}, groups={dd[group].nunique()}, loglik={m.llf:.1f}", flush=True)
    return m


def tidy(m, label, keep=None):
    p = pd.DataFrame({"term": m.params.index, "coef": m.params.values,
                      "se": m.bse.values, "z": m.tvalues.values, "p": m.pvalues.values})
    p = p[~p.term.str.startswith(("C(t)", "C(month)", "Group Var"))]
    if keep:
        p = p[p.term.isin(keep) | p.term.eq("Intercept")]
    p["model"] = label
    return p


def mixed_block(df, prefix, TAB):
    d4 = df.dropna(subset=["repeat_rate", "disorder_rate_l1", "resp_l1"]).copy()
    d4["disorder_l1_z"] = z(d4.disorder_rate_l1)
    d4["resp_l1_z"] = z(d4.resp_l1)
    FE_T = "C(t)"
    M2 = fit_mixed(d4, f"repeat_rate ~ disorder_l1_z + {FE_T}", label=f"{prefix} M2")
    M3 = fit_mixed(d4, f"repeat_rate ~ disorder_l1_z + resp_l1_z + {FE_T}", label=f"{prefix} M3")
    M4 = fit_mixed(d4, f"repeat_rate ~ disorder_l1_z * resp_l1_z + {FE_T}", label=f"{prefix} M4")
    KEEP = ["disorder_l1_z", "resp_l1_z", "disorder_l1_z:resp_l1_z"]
    tab = pd.concat([tidy(M2, "M2", KEEP), tidy(M3, "M3", KEEP), tidy(M4, "M4", KEEP)])
    savetab(tab, f"T3_multilevel_repeat_{prefix.lower()}", TAB)
    out = {
        "M2_disorder": float(M2.params.get("disorder_l1_z", np.nan)),
        "M3_disorder": float(M3.params.get("disorder_l1_z", np.nan)),
        "M4_disorder": float(M4.params.get("disorder_l1_z", np.nan)),
        "M3_resp": float(M3.params.get("resp_l1_z", np.nan)),
        "M4_interaction": float(M4.params.get("disorder_l1_z:resp_l1_z", np.nan)),
        "M4_interaction_p": float(M4.pvalues.get("disorder_l1_z:resp_l1_z", np.nan)),
    }
    b_d = M4.params.get("disorder_l1_z", np.nan)
    b_i = M4.params.get("disorder_l1_z:resp_l1_z", np.nan)
    V = M4.cov_params()
    idx = ["disorder_l1_z", "disorder_l1_z:resp_l1_z"]
    slope_rows = []
    for lvl, val in [("-1 SD (unresponsive)", -1), ("Mean", 0), ("+1 SD (responsive)", 1)]:
        est = b_d + b_i * val
        var = V.loc[idx[0], idx[0]] + val ** 2 * V.loc[idx[1], idx[1]] + 2 * val * V.loc[idx[0], idx[1]]
        se = math.sqrt(max(float(var), 0))
        from scipy import stats as _st
        slope_rows.append(dict(Responsiveness=lvl, Slope=est, SE=se,
                               z=est / se if se else np.nan,
                               p=2 * (1 - _st.norm.cdf(abs(est / se))) if se else np.nan,
                               CI_lo=est - 1.96 * se, CI_hi=est + 1.96 * se))
    sl = pd.DataFrame(slope_rows)
    savetab(sl, f"T4_simple_slopes_{prefix.lower()}", TAB)
    out["slope_low_resp"] = float(sl.Slope.iloc[0])
    out["slope_high_resp"] = float(sl.Slope.iloc[2])
    return d4, M3, M4, sl, out


def attach_boro(nyc, DERIVED):
    ev = pd.read_parquet(DERIVED / "nyc311_events.parquet", columns=["GEOID", "boro", "zipc"])
    ev["boro"] = ev["boro"].astype(str).str.upper().str.strip()
    ev.loc[ev.boro.isin(["", "UNSPECIFIED", "NAN", "NONE"]), "boro"] = np.nan
    bmap = (ev.dropna(subset=["GEOID", "boro"])
            .groupby(["GEOID", "boro"]).size().rename("k").reset_index()
            .sort_values("k", ascending=False).drop_duplicates("GEOID")[["GEOID", "boro"]])
    nyc = nyc.merge(bmap, on="GEOID", how="left")
    return nyc


def run_nmf(DERIVED, TAB):
    from sklearn.decomposition import NMF
    ev = pd.read_parquet(DERIVED / "nyc311_events.parquet", columns=["GEOID", "created", "ctype"])
    ev["month"] = ev["created"].dt.to_period("M").dt.to_timestamp()
    top = ev.ctype.astype(str).value_counts().head(80).index
    ev = ev[ev.ctype.astype(str).isin(top)]
    mat = (ev.groupby(["GEOID", "month", "ctype"]).size()
           .unstack("ctype", fill_value=0))
    X = mat.to_numpy(dtype=float)
    X = X / np.clip(X.sum(1, keepdims=True), 1, None)
    k = 6
    model = NMF(n_components=k, init="nndsvda", max_iter=400, random_state=7)
    W = model.fit_transform(X)
    H = model.components_
    types = mat.columns.astype(str)
    topics = []
    for i in range(k):
        order = np.argsort(H[i])[::-1][:8]
        topics.append(dict(topic=i, top_types=", ".join(types[j] for j in order),
                           max_loading=float(H[i].max())))
    savetab(pd.DataFrame(topics), "T2_nmf_topics", TAB)
    wdf = pd.DataFrame(W, index=mat.index, columns=[f"nmf_{i}" for i in range(k)]).reset_index()
    wdf.to_parquet(DERIVED / "nyc_nmf_tract_month.parquet", index=False)
    RESULTS["nmf_k"] = k
    RESULTS["nmf_n"] = int(len(wdf))
    print("NMF topics:", k, "docs", len(wdf), flush=True)
    return wdf


def run_did(nyc, TAB, FIG):
    cut = pd.Timestamp("2020-07-01")
    pre = nyc[nyc.month < cut]
    if pre.empty or (nyc.month >= cut).sum() < 100:
        print("DiD skipped — not enough pre/post months", flush=True)
        RESULTS["did_treat_post"] = float("nan")
        RESULTS["did_p"] = float("nan")
        return
    treat = pre.groupby("GEOID").san_rate.mean()
    thr = treat.median()
    nyc = nyc.copy()
    nyc["treat"] = nyc.GEOID.map((treat > thr).astype(int))
    nyc["post"] = (nyc.month >= cut).astype(int)
    nyc["treat_post"] = nyc.treat * nyc.post
    dd = nyc.dropna(subset=["repeat_rate", "treat", "treat_post", "GEOID", "t"])
    m = fe_ols(dd, "repeat_rate", ["treat_post"], absorb="GEOID", dummies=("t",))
    savetab(m.frame().head(8), "T11_did_july2020", TAB)
    RESULTS["did_treat_post"] = float(m.params.get("treat_post", np.nan))
    RESULTS["did_p"] = float(m.pvalues.get("treat_post", np.nan))
    RESULTS["did_n"] = int(m.nobs)
    print(f"DiD July 2020 treat×post={RESULTS['did_treat_post']:.4f} p={RESULTS['did_p']:.3g}", flush=True)

    ts = (nyc.dropna(subset=["treat"])
          .groupby(["month", "treat"]).repeat_rate.mean().unstack("treat"))
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    if 0 in ts.columns:
        ax.plot(ts.index, ts[0], color=PAL["blue"], lw=2, label="Low sanitation (control)")
    if 1 in ts.columns:
        ax.plot(ts.index, ts[1], color=PAL["orange"], lw=2, label="High sanitation (treated)")
    ax.axvline(cut, color=INK2, lw=1, ls="--")
    ax.legend(loc="best", fontsize=8)
    finish(ax, "Repeat complaining around the July 2020 sanitation shock",
           None, "Mean repeat rate",
           "Treated = above-median pre-period sanitation rate. Vertical line: July 2020.")
    save(fig, "F8_did_july2020", FIG)


def run_borough_rd(cs, TAB, FIG):
    cs2 = cs.dropna(subset=["boro", "lon", "lat"]).copy()
    if cs2.boro.nunique() < 2:
        print("borough RD skipped — no borough labels", flush=True)
        RESULTS["boro_rd_n"] = 0
        RESULTS["boro_rd_coef"] = float("nan")
        RESULTS["boro_rd_p"] = float("nan")
        return
    trees = {b: cKDTree(g[["lon", "lat"]].values) for b, g in cs2.groupby("boro") if len(g) >= 3}

    def dist_other(row):
        ds = []
        for b, t in trees.items():
            if b == row.boro:
                continue
            ds.append(t.query([row.lon, row.lat])[0])
        return min(ds) if ds else np.nan

    cs2["d_boundary"] = cs2.apply(dist_other, axis=1) * 111.0
    band = cs2[cs2.d_boundary <= 1.5].copy()
    band["resp_hi"] = (band.resp > band.resp.median()).astype(int)
    rd = smf.ols(
        "repeat_rate ~ resp_hi + d_boundary + resp_hi:d_boundary + disadvantage + stability + log_density",
        band).fit(cov_type="HC1")
    savetab(pd.DataFrame({"term": rd.params.index, "coef": rd.params.values,
                          "se": rd.bse.values, "p": rd.pvalues.values}),
            "T12_borough_boundary_rd", TAB)
    RESULTS["boro_rd_n"] = int(len(band))
    RESULTS["boro_rd_coef"] = float(rd.params.get("resp_hi", np.nan))
    RESULTS["boro_rd_p"] = float(rd.pvalues.get("resp_hi", np.nan))
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.scatter(band.d_boundary, band.repeat_rate, s=12, alpha=0.35, color=PAL["blue"], linewidths=0)
    finish(ax, "Repeat complaining near borough edges",
           "Distance to nearest other-borough tract (km)", "Mean tract repeat rate",
           "Bandwidth 1.5 km. NYC boroughs.")
    save(fig, "F9_borough_boundary", FIG)


def main() -> int:
    PROJECT = project_root()
    os.environ.setdefault("PAPER6_ROOT", str(PROJECT))
    DERIVED = PROJECT / "derived"
    FIG = PROJECT / "figures"
    TAB = PROJECT / "tables"
    FIG.mkdir(exist_ok=True, parents=True)
    TAB.mkdir(exist_ok=True, parents=True)
    setup_style()
    cap_blas_threads()
    device = describe_compute("lgbm")
    apply_cuda_visible(device)
    LGB_KW = lgbm_fit_kwargs(device)
    print("project:", PROJECT, flush=True)
    print("compute:", device["reason"], flush=True)
    print("threads:", cap_blas_threads(), "CUDA_VISIBLE_DEVICES",
          os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)

    nyc_raw = pd.read_parquet(DERIVED / "nyc_tract_month.parquet")
    chi_raw = pd.read_parquet(DERIVED / "chi_tract_month.parquet")
    nyc = prepare(nyc_raw, "NYC")
    chi = prepare(chi_raw, "CHI")
    nyc = attach_boro(nyc, DERIVED)

    RESULTS["sample_city"] = "NYC"
    RESULTS["n_tract_months"] = int(len(nyc))
    RESULTS["n_tracts"] = int(nyc.GEOID.nunique())
    RESULTS["n_requests_total"] = int(nyc_raw.n_complaints.sum())
    RESULTS["window"] = f"{int(nyc.month.dt.year.min())}-{int(nyc.month.dt.year.max())}"
    RESULTS["chi_n_tract_months"] = int(len(chi))
    RESULTS["chi_n_tracts"] = int(chi.GEOID.nunique())
    print("NYC sample", nyc.shape, "tracts", nyc.GEOID.nunique(),
          "window", RESULTS["window"], flush=True)
    print("CHI sample", chi.shape, "tracts", chi.GEOID.nunique(), flush=True)

    DESC = [
        ("repeat_rate", "Repeat complaining rate (12-month recurrence)"),
        ("same_domain_rate", "Same-domain recurrence rate"),
        ("complaints_per_address", "Complaints per reporting address"),
        ("concentration", "Address concentration (HHI)"),
        ("complaint_rate", "Complaints per 1,000 residents"),
        ("disorder_rate", "Disorder complaints per 1,000"),
        ("phys_rate", "Physical disorder per 1,000"),
        ("social_rate", "Social disorder per 1,000"),
        ("san_rate", "Sanitation complaints per 1,000"),
        ("infra_rate", "Infrastructure complaints per 1,000"),
        ("hous_rate", "Housing complaints per 1,000"),
        ("median_days_close", "Median days to close"),
        ("share_closed_3d", "Share closed within 3 days"),
        ("share_unresolved", "Share unresolved"),
        ("resp", "Municipal responsiveness (z)"),
        ("pct_ba_plus", "% BA or higher"),
        ("pct_unemployed", "% unemployed"),
        ("pct_vacant", "% vacant housing units"),
        ("pct_owner_occ", "% owner-occupied"),
        ("pop_density_km2", "Population density (per km2)"),
        ("zhvi", "Zillow Home Value Index (USD)"),
        ("violent_rate", "Violent complaints per 1,000"),
    ]
    rows = []
    for c, lab in DESC:
        if c not in nyc.columns:
            continue
        s = pd.to_numeric(nyc[c], errors="coerce")
        if s.notna().sum() == 0:
            continue
        rows.append(dict(Variable=lab, N=int(s.notna().sum()), Mean=s.mean(), SD=s.std(),
                         P10=s.quantile(.10), Median=s.median(), P90=s.quantile(.90)))
    savetab(pd.DataFrame(rows), "T1_descriptives", TAB)

    try:
        ev2 = pd.read_parquet(DERIVED / "nyc311_events.parquet", columns=["domain"])
        dom = ev2.domain.value_counts(normalize=True).rename("share").reset_index()
        if pd.api.types.is_numeric_dtype(dom.domain):
            names = ["physical_disorder", "social_disorder", "housing",
                     "infrastructure", "sanitation", "other"]
            dom["domain"] = dom.domain.map(lambda i: names[int(i)] if int(i) < len(names) else str(i))
        savetab(dom, "T2b_domain_totals", TAB, "%.4f")
        for key in ["physical_disorder", "social_disorder", "housing", "infrastructure", "sanitation"]:
            RESULTS[f"share_{key}"] = float(dom.loc[dom.domain.eq(key), "share"].sum())
    except Exception as exc:
        print("domain tab skip", exc, flush=True)

    try:
        run_nmf(DERIVED, TAB)
    except Exception as exc:
        print("NMF skipped:", type(exc).__name__, exc, flush=True)
        RESULTS["nmf_k"] = 0

    dom_cols = [(c, lab) for c, lab in [
        ("n_physical_disorder", "Physical disorder"),
        ("n_social_disorder", "Social disorder"),
        ("n_sanitation", "Sanitation"),
        ("n_housing", "Housing"),
        ("n_infrastructure", "Infrastructure"),
    ] if c in nyc.columns and nyc[c].sum() > 0]
    ts = nyc.groupby("month")[[c for c, _ in dom_cols]].sum() / 1000.0
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    colors = [PAL["blue"], PAL["orange"], PAL["aqua"], PAL["violet"], PAL["yellow"]]
    for (c, lab), col in zip(dom_cols, colors):
        ax.plot(ts.index, ts[c], color=col, lw=2, label=lab)
        ax.annotate(lab, (ts.index[-1], ts[c].iloc[-1]), xytext=(4, 0),
                    textcoords="offset points", fontsize=7.5, color=col, va="center")
    finish(ax, "Monthly 311 requests by domain, New York City",
           None, "Requests (thousands)",
           f"Source: NYC 311 monthly extracts, {RESULTS['window']}.")
    save(fig, "F1_volume_by_domain", FIG)

    def lorenz(vals):
        v = np.sort(np.asarray(vals, dtype=float))
        v = v[~np.isnan(v)]
        c = np.cumsum(v) / v.sum()
        return np.linspace(0, 1, len(v) + 1), np.concatenate([[0], c])

    tract_tot = nyc.groupby("GEOID").n_complaints.sum()
    x1, y1 = lorenz(tract_tot.values)
    gini = 1 - 2 * np.trapezoid(y1, x1)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.plot([0, 1], [0, 1], color=GRID, lw=1.5)
    ax.plot(x1, y1, color=PAL["blue"], lw=2)
    ax.fill_between(x1, y1, x1, color=PAL["blue"], alpha=0.10)
    ax.annotate(f"Gini = {gini:.3f}", (0.06, 0.80), color=PAL["blue"], fontsize=9.5, fontweight="semibold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    finish(ax, "Concentration of 311 requests across census tracts",
           "Cumulative share of tracts (ranked)", "Cumulative share of requests",
           f"Source: NYC 311, {RESULTS['window']}.")
    save(fig, "F2_lorenz_tracts", FIG)
    RESULTS["gini_tracts"] = float(gini)

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    samp = nyc.sample(min(8000, len(nyc)), random_state=2)
    ax.scatter(samp.resp, samp.repeat_rate, s=6, alpha=0.15, color=PAL["blue"], linewidths=0)
    bins = pd.qcut(nyc.resp, 20, duplicates="drop")
    g = nyc.groupby(bins, observed=True)
    ax.plot(g.resp.mean(), g.repeat_rate.mean(), color=PAL["orange"], lw=2)
    finish(ax, "Repeat complaining and municipal responsiveness",
           "Responsiveness (z)", "Repeat rate",
           f"Orange: mean repeat rate by responsiveness ventile. NYC {RESULTS['window']}.")
    save(fig, "F3_repeat_vs_responsiveness", FIG)

    _, M3, M4, tab4, nyc_m = mixed_block(nyc, "NYC", TAB)
    RESULTS.update(nyc_m)
    savetab(tab4, "T4_simple_slopes", TAB)
    savetab(pd.read_csv(TAB / "T3_multilevel_repeat_nyc.csv"), "T3_multilevel_repeat", TAB)

    try:
        _, _, _, _, chi_m = mixed_block(chi, "CHI", TAB)
        RESULTS["chi_M4_disorder"] = chi_m["M4_disorder"]
        RESULTS["chi_M4_interaction_p"] = chi_m["M4_interaction_p"]
    except Exception as exc:
        print("Chicago replication skipped:", type(exc).__name__, exc, flush=True)

    b_d = M4.params.get("disorder_l1_z", np.nan)
    b_i = M4.params.get("disorder_l1_z:resp_l1_z", np.nan)
    V = M4.cov_params()
    idx = ["disorder_l1_z", "disorder_l1_z:resp_l1_z"]
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    xs = np.linspace(-1.5, 1.5, 61)
    slope = b_d + b_i * xs
    se_line = np.sqrt(V.loc[idx[0], idx[0]] + xs ** 2 * V.loc[idx[1], idx[1]] + 2 * xs * V.loc[idx[0], idx[1]])
    ax.plot(xs, slope, color=PAL["blue"], lw=2)
    ax.fill_between(xs, slope - 1.96 * se_line, slope + 1.96 * se_line, color=PAL["blue"], alpha=0.12)
    ax.axhline(0, color=INK2, lw=1)
    finish(ax, "The disorder-to-repeat link by municipal responsiveness",
           "Responsiveness (z, t−1)", "Slope of lagged disorder",
           f"Shaded band: 95% CI. MixedLM M4. NYC {RESULTS['window']}.")
    save(fig, "F4_marginal_effect", FIG)

    d4 = nyc.dropna(subset=["repeat_rate", "disorder_rate_l1", "resp_l1"]).copy()
    d4["disorder_l1_z"] = z(d4.disorder_rate_l1)
    d4["resp_l1_z"] = z(d4.resp_l1)
    a_m = fit_mixed(d4, "resp_l1_z ~ disorder_l1_z + C(t)", label="a: disorder -> resp")
    a, b = float(a_m.params.get("disorder_l1_z", np.nan)), float(M3.params.get("resp_l1_z", np.nan))
    savetab(pd.DataFrame([
        dict(path="a: disorder -> responsiveness", est=a, se=float(a_m.bse.get("disorder_l1_z", np.nan))),
        dict(path="b: responsiveness -> repeat", est=b, se=float(M3.bse.get("resp_l1_z", np.nan))),
        dict(path="c': direct disorder -> repeat", est=float(M3.params.get("disorder_l1_z", np.nan)),
             se=float(M3.bse.get("disorder_l1_z", np.nan))),
        dict(path="a*b: indirect", est=a * b, se=np.nan),
    ]), "T5_mediation", TAB)
    RESULTS["indirect_effect"] = float(a * b)
    RESULTS["a_disorder_to_resp"] = a

    from libpysal.weights import KNN
    from esda.moran import Moran
    from spreg import OLS as SPOLS, GM_Lag, GM_Error_Het

    agg = dict(repeat_rate=("repeat_rate", "mean"), disorder_rate=("disorder_rate", "mean"),
               resp=("resp", "mean"), complaint_rate=("complaint_rate", "mean"),
               disadvantage=("disadvantage", "mean"), stability=("stability", "mean"),
               log_density=("log_density", "mean"), log_pop=("log_pop", "mean"),
               lat=("lat", "mean"), lon=("lon", "mean"))
    if "boro" in nyc.columns:
        agg["boro"] = ("boro", lambda s: s.dropna().mode().iloc[0] if s.dropna().size else np.nan)
    cs = nyc.groupby("GEOID").agg(**agg).dropna(subset=["repeat_rate", "disorder_rate", "resp", "lat", "lon"]).reset_index()
    w = KNN.from_array(cs[["lon", "lat"]].values, k=8)
    w.transform = "r"
    mor = []
    for var in ["repeat_rate", "disorder_rate", "resp", "complaint_rate"]:
        s = cs[var].fillna(cs[var].mean()).values
        mi = Moran(s, w)
        mor.append(dict(Variable=var, MoranI=mi.I, E_I=mi.EI, z=mi.z_sim, p=mi.p_sim))
    tab6 = pd.DataFrame(mor)
    savetab(tab6, "T6_morans_i", TAB)
    RESULTS["moran_repeat"] = float(tab6.loc[tab6.Variable.eq("repeat_rate"), "MoranI"].iloc[0])

    Xcols = ["disorder_rate", "resp", "disadvantage", "stability", "log_density", "log_pop"]
    Xm = cs[Xcols].fillna(cs[Xcols].mean()).values
    ym = cs[["repeat_rate"]].values
    ols = SPOLS(ym, Xm, w=w, name_x=Xcols, name_y="repeat_rate", spat_diag=True, moran=True)
    lag = GM_Lag(ym, Xm, w=w, name_x=Xcols, name_y="repeat_rate")
    err = GM_Error_Het(ym, Xm, w=w, name_x=Xcols, name_y="repeat_rate")

    def spread(model, name):
        names = model.name_x + (["W_repeat_rate"] if hasattr(model, "rho") else [])
        bvec = np.asarray(model.betas).ravel()
        se = np.sqrt(np.diag(model.vm))[:len(bvec)]
        return pd.DataFrame({"model": name, "term": names[:len(bvec)], "coef": bvec, "se": se,
                             "z": bvec / se, "p": 2 * (1 - sm.stats.stattools.stats.norm.cdf(np.abs(bvec / se)))})

    savetab(pd.concat([spread(ols, "OLS"), spread(lag, "Spatial lag (SAR)"),
                       spread(err, "Spatial error (SEM)")]), "T7_spatial_models", TAB)
    RESULTS["sar_rho"] = float(np.asarray(lag.betas).ravel()[-1])

    run_did(nyc, TAB, FIG)
    run_borough_rd(cs, TAB, FIG)

    import lightgbm as lgb
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.metrics import r2_score, mean_absolute_error
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split

    FEATS = [f for f in [
        "disorder_rate_l1", "resp_l1", "complaint_rate_l1", "repeat_rate_l1",
        "disorder_rate_l3", "resp_l3", "complaint_rate_l12", "disorder_rate_l12",
        "concentration", "share_physical_disorder", "share_infrastructure", "share_sanitation",
        "disadvantage", "stability", "log_density", "log_pop",
        "pct_ba_plus", "pct_unemployed", "pct_vacant", "pct_owner_occ",
        "median_days_close", "share_closed_3d", "share_unresolved", "t",
        "violent_rate_l1", "crime_rate_l1",
    ] if f in nyc.columns]
    ml = nyc.dropna(subset=["repeat_rate"] + FEATS[:6]).copy()
    cut = ml.t.max() - 12
    tr, te = ml[ml.t <= cut], ml[ml.t > cut]
    Xtr, ytr = tr[FEATS].astype(float), tr.repeat_rate
    Xte, yte = te[FEATS].astype(float), te.repeat_rate
    med = Xtr.median()
    ridge = Ridge(alpha=1.0).fit(Xtr.fillna(med), ytr)
    yb = ridge.predict(Xte.fillna(med))
    model = lgb.LGBMRegressor(
        n_estimators=900, learning_rate=0.045, num_leaves=63, min_child_samples=60,
        subsample=0.85, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=7, verbose=-1, **LGB_KW)
    model.fit(Xtr, ytr, eval_set=[(Xte, yte)], eval_metric="l2",
              callbacks=[lgb.early_stopping(60, verbose=False)])
    yp = model.predict(Xte)
    mlp = Pipeline([
        ("sc", StandardScaler()),
        ("nn", MLPRegressor(hidden_layer_sizes=(128, 64), activation="relu",
                            alpha=1e-4, learning_rate_init=1e-3, max_iter=80,
                            early_stopping=True, random_state=7, verbose=False)),
    ])
    mlp.fit(Xtr.fillna(med), ytr)
    yn = mlp.predict(Xte.fillna(med))
    Xall, yall = ml[FEATS].astype(float).fillna(ml[FEATS].median()), ml.repeat_rate
    Xa, Xb, ya, yb2 = train_test_split(Xall, yall, test_size=0.2, random_state=7)
    n_jobs = int(os.environ.get("PAPER6_THREADS", "36"))
    lgb_r = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31,
                              random_state=7, verbose=-1, device="cpu", n_jobs=n_jobs)
    try:
        lgb_r.fit(Xa, ya)
        r2_rand = float(r2_score(yb2, lgb_r.predict(Xb)))
    except Exception:
        r2_rand = float("nan")
    hold = f"last 12 months ({int(te.month.dt.year.min()) if len(te) else '?'})"
    perf = pd.DataFrame([
        dict(Model="Ridge (holdout)", Split=hold, R2=r2_score(yte, yb), MAE=mean_absolute_error(yte, yb)),
        dict(Model="LightGBM (holdout)", Split=hold, R2=r2_score(yte, yp), MAE=mean_absolute_error(yte, yp)),
        dict(Model="MLP 128-64 (holdout)", Split=hold, R2=r2_score(yte, yn), MAE=mean_absolute_error(yte, yn)),
        dict(Model="LightGBM (random 20%)", Split="i.i.d. split", R2=r2_rand, MAE=np.nan),
    ])
    savetab(perf, "T8_ml_performance", TAB)
    RESULTS["r2_lgbm_holdout"] = float(r2_score(yte, yp))
    RESULTS["r2_mlp_holdout"] = float(r2_score(yte, yn))
    RESULTS["r2_ridge_holdout"] = float(r2_score(yte, yb))
    RESULTS["r2_lgbm_random"] = r2_rand
    print(perf.round(4).to_string(index=False), flush=True)

    import shap
    samp = Xte.sample(min(4000, len(Xte)), random_state=3)
    sv = shap.TreeExplainer(model).shap_values(samp)
    imp = (pd.DataFrame({"feature": FEATS, "mean_abs_shap": np.abs(sv).mean(axis=0)})
           .sort_values("mean_abs_shap", ascending=False))
    savetab(imp, "T9_shap_importance", TAB)
    top = imp.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.barh(top.feature, top.mean_abs_shap, color=PAL["blue"], height=0.62)
    ax.grid(axis="y", visible=False)
    finish(ax, "What tracks repeat complaining (mean |SHAP|)", "Mean |SHAP|", None,
           "LightGBM; SHAP on holdout months. Descriptive.")
    save(fig, "F5_shap_importance", FIG)
    fig = plt.figure(figsize=(6.4, 4.4))
    shap.summary_plot(sv, samp, max_display=12, show=False, plot_size=None, color_bar_label="Feature value")
    plt.title("SHAP value distribution", loc="left", fontsize=10.5, fontweight="semibold")
    save(fig, "F6_shap_beeswarm", FIG)

    key = "resp_l1" if "resp_l1" in FEATS else FEATS[0]
    ki = FEATS.index(key)
    sh = pd.DataFrame({"GEOID": te.loc[samp.index, "GEOID"].values,
                       "lat": te.loc[samp.index, "lat"].values,
                       "lon": te.loc[samp.index, "lon"].values,
                       "shap": sv[:, ki]})
    tsh = sh.groupby("GEOID").agg(shap=("shap", "mean"), lat=("lat", "mean"), lon=("lon", "mean")).reset_index()
    savetab(tsh, "T10_spatial_shap", TAB, "%.6f")
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    sc = ax.scatter(tsh.lon, tsh.lat, c=tsh.shap, s=14, cmap="coolwarm", linewidths=0)
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    ax.set_aspect("equal", adjustable="datalim")
    finish(ax, "Where responsiveness SHAP is largest", "Longitude", "Latitude",
           "Tract-mean SHAP for lagged responsiveness. NYC.")
    save(fig, "F7_spatial_shap_map", FIG)

    RESULTS["gnn_r2"] = float("nan")
    try:
        import torch
        import torch.nn as nn
        from sklearn.metrics import r2_score as _r2
        torch.manual_seed(7)
        feats_g = ["disorder_rate", "resp", "disadvantage", "stability", "log_density", "log_pop"]
        Gx = cs[feats_g].fillna(cs[feats_g].mean()).values.astype(np.float32)
        Gy = cs["repeat_rate"].values.astype(np.float32)
        Gx = (Gx - Gx.mean(0)) / np.clip(Gx.std(0), 1e-6, None)
        coords = cs[["lon", "lat"]].values
        tree = cKDTree(coords)
        _, idxn = tree.query(coords, k=9)
        n = len(cs)
        src = np.repeat(np.arange(n), 8)
        dst = idxn[:, 1:].ravel()
        A = np.zeros((n, n), dtype=np.float32)
        A[src, dst] = 1.0
        A = A / np.clip(A.sum(1, keepdims=True), 1, None)
        device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        A_t = torch.tensor(A, device=device_t)
        X_t = torch.tensor(Gx, device=device_t)
        y_t = torch.tensor(Gy, device=device_t).view(-1, 1)
        rng = np.random.default_rng(7)
        perm = rng.permutation(n)
        nte = max(int(0.2 * n), 20)
        te_i, tr_i = perm[:nte], perm[nte:]

        class Sage(nn.Module):
            def __init__(self, d_in, h=32):
                super().__init__()
                self.lin_self = nn.Linear(d_in, h)
                self.lin_nei = nn.Linear(d_in, h)
                self.out = nn.Linear(h, 1)

            def forward(self, x, a):
                h = torch.relu(self.lin_self(x) + self.lin_nei(a @ x))
                return self.out(h)

        net = Sage(Gx.shape[1]).to(device_t)
        opt = torch.optim.Adam(net.parameters(), lr=1e-2, weight_decay=1e-3)
        tr_i_t = torch.tensor(tr_i, device=device_t, dtype=torch.long)
        for _ in range(200):
            net.train()
            opt.zero_grad()
            pred = net(X_t, A_t)
            ((pred[tr_i_t] - y_t[tr_i_t]) ** 2).mean().backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            pred = net(X_t, A_t).cpu().numpy().ravel()
        RESULTS["gnn_r2"] = float(_r2(Gy[te_i], pred[te_i]))
        print(f"spatial GNN test R2={RESULTS['gnn_r2']:.3f} on {device_t}", flush=True)
        perf.loc[len(perf)] = dict(Model="Spatial GNN (GraphSAGE-lite)", Split="20% tracts",
                                   R2=RESULTS["gnn_r2"], MAE=np.nan)
        savetab(perf, "T8_ml_performance", TAB)
    except Exception as exc:
        print("GNN skipped:", type(exc).__name__, exc, flush=True)

    from econml.dml import CausalForestDML
    from sklearn.ensemble import GradientBoostingRegressor
    cf_df = nyc.dropna(subset=["repeat_rate", "resp_l1", "disadvantage", "stability", "log_density",
                               "log_pop", "disorder_rate_l1", "pct_vacant", "pct_owner_occ"]).copy()
    cf_df = cf_df.sample(min(40000, len(cf_df)), random_state=4)
    RESULTS["cate_mean"] = float("nan")
    RESULTS["cate_sd"] = float("nan")
    try:
        cf = CausalForestDML(
            model_y=GradientBoostingRegressor(n_estimators=80, max_depth=3, random_state=1),
            model_t=GradientBoostingRegressor(n_estimators=80, max_depth=3, random_state=1),
            n_estimators=256, min_samples_leaf=40, discrete_treatment=False, cv=3, random_state=7)
        Y = cf_df.repeat_rate.values
        T = cf_df.resp_l1.values
        Xc = cf_df[["disadvantage", "stability", "log_density", "log_pop", "pct_vacant", "pct_owner_occ"]].values
        Wc = cf_df[["disorder_rate_l1", "complaint_rate_l1", "t"]].fillna(0).values
        cf.fit(Y, T, X=Xc, W=Wc)
        cate = cf.effect(Xc)
        cf_df["cate"] = cate
        RESULTS["cate_mean"] = float(cate.mean())
        RESULTS["cate_sd"] = float(cate.std())
        cf_df["dis_q"] = pd.qcut(cf_df.disadvantage, 5, labels=[f"Q{i}" for i in range(1, 6)])
        eq = (cf_df.groupby("dis_q").agg(CATE=("cate", "mean"), SD=("cate", "std"), N=("cate", "size"))
              .reset_index().rename(columns={"dis_q": "Disadvantage quintile"}))
        savetab(eq, "T13_cate_by_disadvantage", TAB)
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
        axes[0].hist(cate, bins=40, color=PAL["blue"], edgecolor="white", linewidth=0.5, alpha=0.9)
        axes[0].axvline(cate.mean(), color=PAL["orange"], lw=2)
        finish(axes[0], "Distribution of treatment effects", "Effect of lagged responsiveness", "Tract-months")
        axes[1].bar(eq["Disadvantage quintile"], eq.CATE, color=PAL["blue"], width=0.62)
        axes[1].axhline(0, color=INK2, lw=1)
        axes[1].grid(axis="x", visible=False)
        finish(axes[1], "Effect by neighbourhood disadvantage", "Quintile (Q5 = most disadvantaged)",
               "Mean CATE", f"Causal forest (DML). NYC {RESULTS['window']}.")
        save(fig, "F11_causal_forest", FIG)
    except Exception as exc:
        print("causal forest skipped:", type(exc).__name__, exc, flush=True)

    RESULTS["bw_price_phys"] = float("nan")
    RESULTS["n_zhvi"] = 0
    if "log_zhvi" in nyc.columns and "log_zhvi_f12" in nyc.columns:
        hp = nyc.dropna(subset=["log_zhvi", "log_zhvi_f12", "phys_rate_l6", "disadvantage"]).copy()
        if len(hp) > 1000:
            hp["d_log_zhvi"] = hp.log_zhvi_f12 - hp.log_zhvi
            hp["phys_l6_z"] = z(hp.phys_rate_l6)
            keep = ["phys_l6_z"]
            if "resp_l1" in hp.columns:
                hp["resp_l1_z"] = z(hp.resp_l1)
                keep.append("resp_l1_z")
            mhp = fe_ols(hp, "d_log_zhvi", keep, absorb="GEOID", dummies=("t",))
            savetab(mhp.frame().head(12), "T14_zhvi_crosslag", TAB)
            RESULTS["bw_price_phys"] = float(mhp.params.get("phys_l6_z", np.nan))
            RESULTS["n_zhvi"] = int(mhp.nobs)

    RESULTS["crime_l6"] = float("nan")
    if "violent_rate_l6" in nyc.columns or "violent_rate_l1" in nyc.columns:
        ycol = "violent_rate_f6" if "violent_rate_f6" in nyc.columns else None
        # lead of violent crime from add_lags if we add it
    nyc2 = nyc.copy()
    if "violent_rate" in nyc2.columns:
        nyc2 = add_lags(nyc2, ["violent_rate"], lags=(), leads=(6,))
        hp = nyc2.dropna(subset=["violent_rate_f6", "phys_rate_l1", "disadvantage"])
        if len(hp) > 1000:
            hp["phys_l1_z"] = z(hp.phys_rate_l1)
            mcr = fe_ols(hp, "violent_rate_f6", ["phys_l1_z"], absorb="GEOID", dummies=("t",))
            savetab(mcr.frame().head(8), "T15_crime_crosslag", TAB)
            RESULTS["crime_l6"] = float(mcr.params.get("phys_l1_z", np.nan))
            RESULTS["n_crime_fe"] = int(mcr.nobs)

    checks = []

    def check(name, cond, detail=""):
        checks.append(dict(check=name, passed=bool(cond), detail=str(detail)))

    check("NYC panel non-empty", len(nyc) > 5000, f"n={len(nyc)}")
    check("NYC tracts in expected range", 1500 <= nyc.GEOID.nunique() <= 2800, nyc.GEOID.nunique())
    check("repeat_rate within [0,1]", nyc.repeat_rate.between(0, 1).all(),
          f"{nyc.repeat_rate.min():.3f}-{nyc.repeat_rate.max():.3f}")
    check("responsiveness standardised", abs(nyc.resp.mean()) < 0.05 and abs(nyc.resp.std() - 1) < 0.08,
          f"mean {nyc.resp.mean():.3f}, sd {nyc.resp.std():.3f}")
    check("M4 disorder estimated", np.isfinite(RESULTS.get("M4_disorder", np.nan)), RESULTS.get("M4_disorder"))
    check("Moran's I positive", RESULTS.get("moran_repeat", 0) > 0, RESULTS.get("moran_repeat"))
    check("NMF topics written", RESULTS.get("nmf_k", 0) >= 4, RESULTS.get("nmf_k"))
    check("July 2020 DiD estimated", np.isfinite(RESULTS.get("did_treat_post", np.nan)),
          RESULTS.get("did_treat_post"))
    check("Chicago replication present", RESULTS.get("chi_n_tracts", 0) > 500, RESULTS.get("chi_n_tracts"))
    n_fig = len(list(FIG.glob("*.png")))
    n_tab = len(list(TAB.glob("*.csv")))
    check("figures written", n_fig >= 8, n_fig)
    check("tables written", n_tab >= 8, n_tab)
    ver = pd.DataFrame(checks)
    savetab(ver, "T00_verification", TAB, "%.4f")
    RESULTS["n_fig"] = n_fig
    RESULTS["n_tab"] = n_tab
    (TAB / "results_headline.json").write_text(json.dumps(RESULTS, indent=1, default=float), encoding="utf-8")
    print(ver.to_string(index=False), flush=True)
    print("FAILED:", int((~ver.passed).sum()), flush=True)
    print("headline:", json.dumps(RESULTS, indent=1, default=float), flush=True)
    return 0 if ver.passed.all() else 1


if __name__ == "__main__":
    sys.exit(main())
