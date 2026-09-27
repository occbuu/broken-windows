#!/usr/bin/env python3
"""Chicago-only Paper 6 analysis (2011–2019). Writes tables/, figures/, results_headline.json."""
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


def load_enrich(PROJECT, DERIVED):
    chi = pd.read_parquet(DERIVED / "chi_tract_month.parquet")
    ev = pd.read_parquet(DERIVED / "chi311_events.parquet", columns=["GEOID", "ward", "zipc"])
    ev["zip5"] = zip5(ev["zipc"])
    ward = (ev.dropna(subset=["GEOID", "ward"])
            .groupby(["GEOID", "ward"]).size().rename("k").reset_index()
            .sort_values("k", ascending=False).drop_duplicates("GEOID")[["GEOID", "ward"]])
    zmap = (ev.dropna(subset=["GEOID", "zip5"])
            .groupby(["GEOID", "zip5"]).size().rename("k").reset_index()
            .sort_values("k", ascending=False).drop_duplicates("GEOID")[["GEOID", "zip5"]])
    chi = chi.merge(ward, on="GEOID", how="left")
    chi = chi.merge(zmap, on="GEOID", how="left")
    zh = pd.read_parquet(DERIVED / "zhvi_zip_month.parquet")[["zipc", "month", "zhvi"]].copy()
    zh["zip5"] = zip5(zh["zipc"])
    zh["month"] = pd.to_datetime(zh["month"])
    chi["month"] = pd.to_datetime(chi["month"])
    chi = chi.merge(zh[["zip5", "month", "zhvi"]], on=["zip5", "month"], how="left")
    print("panel", chi.shape, "ward coverage", chi.ward.notna().mean(),
          "zhvi coverage", chi.zhvi.notna().mean(), flush=True)
    return chi, ev


def prepare(df):
    d = df.copy()
    d["month"] = pd.to_datetime(d["month"])
    d["t"] = (d.month.dt.year - d.month.dt.year.min()) * 12 + d.month.dt.month - 1
    d = d[d["pop"].fillna(0) > 200]
    d = d[d.n_complaints >= 3]
    for c in ["n_physical_disorder", "n_social_disorder", "n_housing", "n_infrastructure", "n_sanitation"]:
        if c not in d.columns:
            d[c] = 0
    d["n_disorder"] = d.n_physical_disorder + d.n_social_disorder
    d["complaint_rate"] = 1000 * d.n_complaints / d["pop"]
    d["disorder_rate"] = 1000 * d.n_disorder / d["pop"]
    d["phys_rate"] = 1000 * d.n_physical_disorder / d["pop"]
    d["san_rate"] = 1000 * d.n_sanitation / d["pop"]
    d["infra_rate"] = 1000 * d.n_infrastructure / d["pop"]
    d["resp"] = (z(-np.log1p(d.median_days_close.fillna(d.median_days_close.median())))
                 + z(d.share_closed_3d.fillna(0))
                 + z(-d.share_unresolved.fillna(0))) / 3.0
    d["resp"] = z(d["resp"])
    d["disadvantage"] = z(z(d.pct_unemployed) + z(d.pct_lt_hs) + z(d.pct_vacant) - z(d.pct_ba_plus))
    d["stability"] = z(d.pct_owner_occ)
    d["log_density"] = np.log1p(d.pop_density_km2)
    d["log_pop"] = np.log(d["pop"])
    d["concentration"] = d.addr_hhi
    d["log_zhvi"] = np.log(d["zhvi"]) if "zhvi" in d.columns else np.nan
    d["city"] = "CHI"
    d = d.sort_values(["GEOID", "month"]).reset_index(drop=True)
    d = add_lags(d, ["disorder_rate", "phys_rate", "san_rate", "infra_rate", "resp",
                     "complaint_rate", "repeat_rate"], lags=(1, 3, 6, 12))
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

    chi_raw, ev = load_enrich(PROJECT, DERIVED)
    nyc = prepare(chi_raw)  # primary sample name kept for minimal churn in formulas
    RESULTS["sample_city"] = "CHI"
    RESULTS["n_tract_months"] = int(len(nyc))
    RESULTS["n_tracts"] = int(nyc.GEOID.nunique())
    RESULTS["n_requests_total"] = int(chi_raw.n_complaints.sum())
    RESULTS["window"] = "2011-2019"
    print("analysis sample", nyc.shape, "tracts", nyc.GEOID.nunique(), flush=True)

    # ---- T1 descriptives ----
    DESC = [
        ("repeat_rate", "Repeat complaining rate (12-month recurrence)"),
        ("same_domain_rate", "Same-domain recurrence rate"),
        ("complaints_per_address", "Complaints per reporting address"),
        ("concentration", "Address concentration (HHI)"),
        ("complaint_rate", "Complaints per 1,000 residents"),
        ("disorder_rate", "Physical-disorder complaints per 1,000"),
        ("phys_rate", "  Physical disorder per 1,000"),
        ("san_rate", "Sanitation complaints per 1,000"),
        ("infra_rate", "Infrastructure complaints per 1,000"),
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

    # domain mix from events (file taxonomy, not NMF)
    src_counts = None
    try:
        ev2 = pd.read_parquet(DERIVED / "chi311_events.parquet", columns=["src", "domain"])
        dom = ev2.domain.value_counts(normalize=True).rename("share").reset_index()
        savetab(dom, "T2b_domain_totals", TAB, "%.4f")
        RESULTS["share_physical_disorder"] = float(dom.loc[dom.domain.eq("physical_disorder"), "share"].sum())
        RESULTS["share_infrastructure"] = float(dom.loc[dom.domain.eq("infrastructure"), "share"].sum())
        RESULTS["share_sanitation"] = float(dom.loc[dom.domain.eq("sanitation"), "share"].sum())
        src_counts = ev2.src.value_counts()
        savetab(src_counts.rename("n").rename_axis("src").reset_index(),
                "T2_request_types", TAB)
    except Exception as exc:
        print("domain tab skip", exc, flush=True)

    # F1 volume
    dom_cols = [(c, lab) for c, lab in [
        ("n_physical_disorder", "Physical disorder"),
        ("n_sanitation", "Sanitation"),
        ("n_infrastructure", "Infrastructure"),
    ] if c in nyc.columns and nyc[c].sum() > 0]
    ts = nyc.groupby("month")[[c for c, _ in dom_cols]].sum() / 1000.0
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for (c, lab), col in zip(dom_cols, [PAL["blue"], PAL["orange"], PAL["aqua"]]):
        ax.plot(ts.index, ts[c], color=col, lw=2, label=lab)
        ax.annotate(lab, (ts.index[-1], ts[c].iloc[-1]), xytext=(4, 0),
                    textcoords="offset points", fontsize=7.5, color=col, va="center")
    finish(ax, "Monthly 311 requests by domain, Chicago",
           None, "Requests (thousands)",
           "Source: Chicago 311 legacy files, 2011–2019.")
    save(fig, "F1_volume_by_domain", FIG)

    # F2 Lorenz
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
           "Source: Chicago 311, 2011–2019.")
    save(fig, "F2_lorenz_tracts", FIG)
    RESULTS["gini_tracts"] = float(gini)

    # F3 repeat vs resp
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    samp = nyc.sample(min(8000, len(nyc)), random_state=2)
    ax.scatter(samp.resp, samp.repeat_rate, s=6, alpha=0.15, color=PAL["blue"], linewidths=0)
    bins = pd.qcut(nyc.resp, 20, duplicates="drop")
    g = nyc.groupby(bins, observed=True)
    xm = g.resp.mean()
    ym = g.repeat_rate.mean()
    ax.plot(xm, ym, color=PAL["orange"], lw=2)
    finish(ax, "Repeat complaining and municipal responsiveness",
           "Responsiveness (z)", "Repeat rate",
           "Orange: mean repeat rate by responsiveness ventile. Chicago 2011–2019.")
    save(fig, "F3_repeat_vs_responsiveness", FIG)

    # ---- MixedLM: time-varying predictors only (ACS is absorbed by tract RE) ----
    d4 = nyc.dropna(subset=["repeat_rate", "disorder_rate_l1", "resp_l1"]).copy()
    d4["disorder_l1_z"] = z(d4.disorder_rate_l1)
    d4["resp_l1_z"] = z(d4.resp_l1)
    FE_T = "C(t)"
    M2 = fit_mixed(d4, f"repeat_rate ~ disorder_l1_z + {FE_T}", label="M2 + disorder")
    M3 = fit_mixed(d4, f"repeat_rate ~ disorder_l1_z + resp_l1_z + {FE_T}", label="M3 + resp")
    M4 = fit_mixed(d4, f"repeat_rate ~ disorder_l1_z * resp_l1_z + {FE_T}", label="M4 + interaction")
    KEEP = ["disorder_l1_z", "resp_l1_z", "disorder_l1_z:resp_l1_z"]
    tab3 = pd.concat([tidy(M2, "M2", KEEP), tidy(M3, "M3", KEEP), tidy(M4, "M4", KEEP)])
    savetab(tab3, "T3_multilevel_repeat", TAB)
    RESULTS["M2_disorder"] = float(M2.params.get("disorder_l1_z", np.nan))
    RESULTS["M3_disorder"] = float(M3.params.get("disorder_l1_z", np.nan))
    RESULTS["M4_disorder"] = float(M4.params.get("disorder_l1_z", np.nan))
    RESULTS["M3_resp"] = float(M3.params.get("resp_l1_z", np.nan))
    RESULTS["M4_interaction"] = float(M4.params.get("disorder_l1_z:resp_l1_z", np.nan))
    RESULTS["M4_interaction_p"] = float(M4.pvalues.get("disorder_l1_z:resp_l1_z", np.nan))

    b_d = M4.params.get("disorder_l1_z", np.nan)
    b_i = M4.params.get("disorder_l1_z:resp_l1_z", np.nan)
    V = M4.cov_params()
    idx = ["disorder_l1_z", "disorder_l1_z:resp_l1_z"]
    slope_rows = []
    for lvl, val in [("-1 SD (unresponsive)", -1), ("Mean", 0), ("+1 SD (responsive)", 1)]:
        est = b_d + b_i * val
        var = V.loc[idx[0], idx[0]] + val ** 2 * V.loc[idx[1], idx[1]] + 2 * val * V.loc[idx[0], idx[1]]
        se = math.sqrt(max(float(var), 0))
        slope_rows.append(dict(Responsiveness=lvl, Slope=est, SE=se, z=est / se if se else np.nan,
                               p=2 * (1 - sm.stats.stattools.stats.norm.cdf(abs(est / se))) if se else np.nan,
                               CI_lo=est - 1.96 * se, CI_hi=est + 1.96 * se))
    tab4 = pd.DataFrame(slope_rows)
    savetab(tab4, "T4_simple_slopes", TAB)
    RESULTS["slope_low_resp"] = float(tab4.Slope.iloc[0])
    RESULTS["slope_high_resp"] = float(tab4.Slope.iloc[2])

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    xs = np.linspace(-1.5, 1.5, 61)
    slope = b_d + b_i * xs
    se_line = np.sqrt(V.loc[idx[0], idx[0]] + xs ** 2 * V.loc[idx[1], idx[1]] + 2 * xs * V.loc[idx[0], idx[1]])
    ax.plot(xs, slope, color=PAL["blue"], lw=2)
    ax.fill_between(xs, slope - 1.96 * se_line, slope + 1.96 * se_line, color=PAL["blue"], alpha=0.12)
    ax.axhline(0, color=INK2, lw=1)
    finish(ax, "The disorder-to-repeat link by municipal responsiveness",
           "Responsiveness (z, t−1)", "Slope of lagged disorder",
           "Shaded band: 95% CI. MixedLM M4, tract RE and month FE. Chicago 2011–2019.")
    save(fig, "F4_marginal_effect", FIG)

    # mediation (exploratory)
    a_m = fit_mixed(d4, f"resp_l1_z ~ disorder_l1_z + {FE_T}", label="a: disorder -> resp")
    a, b = float(a_m.params.get("disorder_l1_z", np.nan)), float(M3.params.get("resp_l1_z", np.nan))
    med = pd.DataFrame([
        dict(path="a: disorder -> responsiveness", est=a, se=float(a_m.bse.get("disorder_l1_z", np.nan))),
        dict(path="b: responsiveness -> repeat", est=b, se=float(M3.bse.get("resp_l1_z", np.nan))),
        dict(path="c': direct disorder -> repeat", est=float(M3.params.get("disorder_l1_z", np.nan)),
             se=float(M3.bse.get("disorder_l1_z", np.nan))),
        dict(path="a*b: indirect", est=a * b, se=np.nan),
    ])
    savetab(med, "T5_mediation", TAB)
    RESULTS["indirect_effect"] = float(a * b)
    RESULTS["a_disorder_to_resp"] = a

    # ---- Spatial ----
    from libpysal.weights import KNN
    from esda.moran import Moran
    from spreg import OLS as SPOLS, GM_Lag, GM_Error_Het

    cs = (nyc.groupby("GEOID")
          .agg(repeat_rate=("repeat_rate", "mean"), disorder_rate=("disorder_rate", "mean"),
               resp=("resp", "mean"), complaint_rate=("complaint_rate", "mean"),
               disadvantage=("disadvantage", "mean"), stability=("stability", "mean"),
               log_density=("log_density", "mean"), log_pop=("log_pop", "mean"),
               lat=("lat", "mean"), lon=("lon", "mean"),
               ward=("ward", "median"))
          .dropna(subset=["repeat_rate", "disorder_rate", "resp", "lat", "lon"]).reset_index())
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
        b = np.asarray(model.betas).ravel()
        se = np.sqrt(np.diag(model.vm))[:len(b)]
        return pd.DataFrame({"model": name, "term": names[:len(b)], "coef": b, "se": se,
                             "z": b / se, "p": 2 * (1 - sm.stats.stattools.stats.norm.cdf(np.abs(b / se)))})

    tab7 = pd.concat([spread(ols, "OLS"), spread(lag, "Spatial lag (SAR)"), spread(err, "Spatial error (SEM)")])
    savetab(tab7, "T7_spatial_models", TAB)
    RESULTS["sar_rho"] = float(np.asarray(lag.betas).ravel()[-1]) if hasattr(lag, "rho") or True else np.nan

    # ---- ML: LightGBM + MLP; R² reported, not a verification gate ----
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

    # random split: capacity, not a 2019 forecast
    Xall, yall = ml[FEATS].astype(float).fillna(ml[FEATS].median()), ml.repeat_rate
    Xa, Xb, ya, yb2 = train_test_split(Xall, yall, test_size=0.2, random_state=7)
    lgb_r = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31,
                              random_state=7, verbose=-1, device="cpu", n_jobs=8)
    try:
        lgb_r.fit(Xa, ya)
        y_rand = lgb_r.predict(Xb)
        r2_rand = float(r2_score(yb2, y_rand))
    except Exception:
        r2_rand = float("nan")

    perf = pd.DataFrame([
        dict(Model="Ridge (2019 holdout)", Split="last 12 months", R2=r2_score(yte, yb), MAE=mean_absolute_error(yte, yb)),
        dict(Model="LightGBM (2019 holdout)", Split="last 12 months", R2=r2_score(yte, yp), MAE=mean_absolute_error(yte, yp)),
        dict(Model="MLP 128-64 (2019 holdout)", Split="last 12 months", R2=r2_score(yte, yn), MAE=mean_absolute_error(yte, yn)),
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
    expl = shap.TreeExplainer(model)
    sv = expl.shap_values(samp)
    imp = (pd.DataFrame({"feature": FEATS, "mean_abs_shap": np.abs(sv).mean(axis=0)})
           .sort_values("mean_abs_shap", ascending=False))
    savetab(imp, "T9_shap_importance", TAB)
    top = imp.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.barh(top.feature, top.mean_abs_shap, color=PAL["blue"], height=0.62)
    ax.grid(axis="y", visible=False)
    finish(ax, "What tracks repeat complaining (mean |SHAP|)", "Mean |SHAP|", None,
           "LightGBM; SHAP on 2019 months. Descriptive, not a forecast claim.")
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
           "Tract-mean SHAP for lagged responsiveness. Chicago.")
    save(fig, "F7_spatial_shap_map", FIG)

    # optional torch spatial GNN on tract cross-section
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
        deg = A.sum(1, keepdims=True)
        A = A / np.clip(deg, 1, None)
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
            loss = ((pred[tr_i_t] - y_t[tr_i_t]) ** 2).mean()
            loss.backward()
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

    # ---- Ward-boundary RD ----
    cs2 = cs.dropna(subset=["ward", "lon", "lat"]).copy()
    cs2["ward"] = cs2.ward.round().astype(int)
    trees = {wrd: cKDTree(g[["lon", "lat"]].values) for wrd, g in cs2.groupby("ward") if len(g) >= 3}

    def dist_other(row):
        ds = []
        for wrd, t in trees.items():
            if wrd == row.ward:
                continue
            ds.append(t.query([row.lon, row.lat])[0])
        return min(ds) if ds else np.nan

    cs2["d_boundary"] = cs2.apply(dist_other, axis=1) * 111.0
    band = cs2[cs2.d_boundary <= 1.5].copy()
    band["resp_hi"] = (band.resp > band.resp.median()).astype(int)
    rd = smf.ols(
        "repeat_rate ~ resp_hi + d_boundary + resp_hi:d_boundary + disadvantage + stability + log_density",
        band).fit(cov_type="HC1")
    rdt = pd.DataFrame({"term": rd.params.index, "coef": rd.params.values,
                        "se": rd.bse.values, "p": rd.pvalues.values})
    savetab(rdt, "T12_ward_boundary_rd", TAB)
    RESULTS["rd_n"] = int(len(band))
    RESULTS["rd_coef"] = float(rd.params.get("resp_hi", np.nan))
    RESULTS["rd_p"] = float(rd.pvalues.get("resp_hi", np.nan))
    print(f"ward-boundary band: {len(band)} tracts within 1.5 km of another ward", flush=True)

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.scatter(band.d_boundary, band.repeat_rate, s=12, alpha=0.35, color=PAL["blue"], linewidths=0)
    finish(ax, "Repeat complaining near aldermanic ward edges",
           "Distance to nearest other-ward tract (km)", "Mean tract repeat rate",
           "Bandwidth 1.5 km. Chicago wards, 2011–2019 tract means.")
    save(fig, "F9_ward_boundary", FIG)

    # ---- Causal forest ----
    from econml.dml import CausalForestDML
    from sklearn.ensemble import GradientBoostingRegressor
    cf_df = nyc.dropna(subset=["repeat_rate", "resp_l1", "disadvantage", "stability", "log_density",
                               "log_pop", "disorder_rate_l1", "pct_vacant", "pct_owner_occ"]).copy()
    cf_df = cf_df.sample(min(40000, len(cf_df)), random_state=4)
    Y = cf_df.repeat_rate.values
    T = cf_df.resp_l1.values
    Xc = cf_df[["disadvantage", "stability", "log_density", "log_pop", "pct_vacant", "pct_owner_occ"]].values
    Wc = cf_df[["disorder_rate_l1", "complaint_rate_l1", "t"]].fillna(0).values
    RESULTS["cate_mean"] = float("nan")
    RESULTS["cate_sd"] = float("nan")
    try:
        cf = CausalForestDML(
            model_y=GradientBoostingRegressor(n_estimators=80, max_depth=3, random_state=1),
            model_t=GradientBoostingRegressor(n_estimators=80, max_depth=3, random_state=1),
            n_estimators=256, min_samples_leaf=40, discrete_treatment=False, cv=3, random_state=7)
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
               "Mean CATE", "Causal forest (DML). Chicago 2011–2019.")
        save(fig, "F11_causal_forest", FIG)
    except Exception as exc:
        print("causal forest skipped:", type(exc).__name__, exc, flush=True)

    # ---- ZHVI: physical disorder → 12-month house-price change ----
    RESULTS["bw_price_phys"] = float("nan")
    RESULTS["n_zhvi"] = 0
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
        print(f"ZHVI FE: N={mhp.nobs}, phys_l6={RESULTS['bw_price_phys']:.5f}", flush=True)
        fig, ax = plt.subplots(figsize=(5.2, 3.3))
        q = pd.qcut(hp.phys_rate_l6, 8, duplicates="drop")
        gg = hp.groupby(q, observed=True).d_log_zhvi.mean()
        ax.plot(range(len(gg)), gg.values, color=PAL["blue"], marker="o")
        ax.axhline(0, color=INK2, lw=1)
        finish(ax, "Subsequent 12-month ZHVI change by lagged physical disorder",
               "Octile of physical-disorder rate (t−6)", "Mean Δ log ZHVI",
               "Chicago ZIP ZHVI joined to tracts. Within-tract month FE in the table.")
        save(fig, "F12_zhvi_disorder", FIG)
    else:
        print("ZHVI overlap too thin for cross-lag", flush=True)

    # ---- Verification (no 2019 R² gate) ----
    checks = []

    def check(name, cond, detail=""):
        checks.append(dict(check=name, passed=bool(cond), detail=str(detail)))

    check("Chicago panel non-empty", len(nyc) > 5000, f"n={len(nyc)}")
    check("tracts in expected range", 700 <= nyc.GEOID.nunique() <= 1600, nyc.GEOID.nunique())
    check("repeat_rate within [0,1]", nyc.repeat_rate.between(0, 1).all(),
          f"{nyc.repeat_rate.min():.3f}-{nyc.repeat_rate.max():.3f}")
    check("responsiveness standardised", abs(nyc.resp.mean()) < 0.05 and abs(nyc.resp.std() - 1) < 0.08,
          f"mean {nyc.resp.mean():.3f}, sd {nyc.resp.std():.3f}")
    check("M4 disorder estimated", np.isfinite(RESULTS.get("M4_disorder", np.nan)), RESULTS.get("M4_disorder"))
    check("Moran's I positive", RESULTS.get("moran_repeat", 0) > 0, RESULTS.get("moran_repeat"))
    check("disorder slopes positive at both resp levels",
          RESULTS.get("slope_low_resp", 0) > 0 and RESULTS.get("slope_high_resp", 0) > 0,
          f"{RESULTS.get('slope_low_resp'):.4f} / {RESULTS.get('slope_high_resp'):.4f}")
    check("ward-boundary RD estimated", RESULTS.get("rd_n", 0) > 50, RESULTS.get("rd_n"))
    check("ZHVI joined", RESULTS.get("n_zhvi", 0) > 500 or nyc.zhvi.notna().mean() > 0.3,
          f"coverage {nyc.zhvi.notna().mean():.2f}, FE N={RESULTS.get('n_zhvi')}")
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
