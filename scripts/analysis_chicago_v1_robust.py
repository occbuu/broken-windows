#!/usr/bin/env python3
"""Chicago V1 figures + Cities robustness. Writes ONLY to tables/chicago_v1 and
figures/chicago_v1. Does not touch tables/results_headline.json or figures/*.png
from the NYC run.

Usage on the lab node:

    cd /path/to/broken-windows
    export PAPER6_ROOT="$PWD"
    nohup .venv/bin/python -u scripts/analysis_chicago_v1_robust.py \\
      > derived/_runs/chicago_v1_robust.log 2>&1 &
"""
from __future__ import annotations

import json
import math
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analysis_chicago import (  # noqa: E402
    PAL, add_lags, fe_ols, finish, load_enrich, prepare, save, setup_style, z,
)
from paper6_runtime import (  # noqa: E402
    apply_cuda_visible, cap_blas_threads, describe_compute, lgbm_fit_kwargs,
    project_root,
)


def savetab(df, name, tab: Path, float_fmt="%.4f"):
    p = tab / f"{name}.csv"
    df.to_csv(p, index=False, float_format=float_fmt)
    print("  table  ->", p, df.shape, flush=True)
    return p


def slope_row(b_d, b_i, V, idx, lvl, val):
    est = b_d + b_i * val
    var = (V.loc[idx[0], idx[0]] + val ** 2 * V.loc[idx[1], idx[1]]
           + 2 * val * V.loc[idx[0], idx[1]])
    se = math.sqrt(max(float(var), 0))
    from scipy import stats as st
    zstat = est / se if se else np.nan
    p = 2 * (1 - st.norm.cdf(abs(zstat))) if se else np.nan
    return dict(Responsiveness=lvl, Slope=est, SE=se, z=zstat, p=p,
                CI_lo=est - 1.96 * se, CI_hi=est + 1.96 * se)


def fe_term(frame, term):
    r = frame[frame.term.eq(term)]
    if r.empty:
        return {}
    row = r.iloc[0]
    return dict(coef=float(row.coef), se=float(row.se), z=float(row.z), p=float(row.p))


def main() -> int:
    PROJECT = project_root()
    os.environ.setdefault("PAPER6_ROOT", str(PROJECT))
    DERIVED = PROJECT / "derived"
    TAB = PROJECT / "tables" / "chicago_v1"
    FIG = PROJECT / "figures" / "chicago_v1"
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    setup_style()
    cap_blas_threads()
    device = describe_compute("lgbm")
    apply_cuda_visible(device)
    LGB_KW = lgbm_fit_kwargs(device)
    print("project:", PROJECT, flush=True)
    print("outputs:", TAB, FIG, flush=True)
    print("compute:", device["reason"], flush=True)
    assert (DERIVED / "chi_tract_month.parquet").exists(), "missing derived/chi_tract_month.parquet"

    chi_raw, _ev = load_enrich(PROJECT, DERIVED)
    nyc = prepare(chi_raw)
    nyc = add_lags(nyc, ["disorder_rate", "phys_rate", "infra_rate", "san_rate", "resp"],
                   lags=(), leads=(1,))
    RESULTS = {
        "sample_city": "CHI",
        "n_tract_months": int(len(nyc)),
        "n_tracts": int(nyc.GEOID.nunique()),
        "n_requests_total": int(chi_raw.n_complaints.sum()),
        "window": "2011-2019",
        "note": "V1 robustness; isolated from NYC tables/",
    }
    print("analysis sample", nyc.shape, "tracts", nyc.GEOID.nunique(), flush=True)

    # ---- figures (Cities needs these) ----------------------------------------
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

    def lorenz(vals):
        v = np.sort(np.asarray(vals, dtype=float))
        v = v[~np.isnan(v)]
        c = np.cumsum(v) / v.sum()
        return np.linspace(0, 1, len(v) + 1), np.concatenate([[0], c])

    tract_tot = nyc.groupby("GEOID").n_complaints.sum()
    x1, y1 = lorenz(tract_tot.values)
    gini = 1 - 2 * np.trapezoid(y1, x1)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.plot([0, 1], [0, 1], color="#dcdcd8", lw=1.5)
    ax.plot(x1, y1, color=PAL["blue"], lw=2)
    ax.fill_between(x1, y1, x1, color=PAL["blue"], alpha=0.10)
    ax.annotate(f"Gini = {gini:.3f}", (0.06, 0.80), color=PAL["blue"], fontsize=9.5, fontweight="semibold")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    finish(ax, "Concentration of 311 requests across census tracts",
           "Cumulative share of tracts (ranked)", "Cumulative share of requests",
           "Source: Chicago 311, 2011–2019.")
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
           "Orange: mean repeat rate by responsiveness ventile. Chicago 2011–2019.")
    save(fig, "F3_repeat_vs_responsiveness", FIG)

    # ---- preferred FE + robustness ------------------------------------------
    rows = []

    def run_fe(label, df, y, xs):
        fr, nobs, ng, r2w = None, None, None, None
        # fe_ols returns FEResult
        m = fe_ols(df, y, xs, absorb="GEOID", dummies=("t",), cluster="GEOID")
        fr = m.frame()
        fr["model"] = label
        fr["y"] = y
        rows.append(fr)
        hit = fe_term(fr, xs[0] if xs[0] != "inter" else "disorder_l1_z")
        print(f"  {label}: N={m.nobs} R2w={m.rsquared_within:.3f} {xs[0]}={hit.get('coef', float('nan')):.4f}",
              flush=True)
        RESULTS[f"{label}_N"] = int(m.nobs)
        RESULTS[f"{label}_R2w"] = float(m.rsquared_within)
        return m, fr

    d4 = nyc.dropna(subset=["repeat_rate", "disorder_rate_l1", "resp_l1"]).copy()
    d4["disorder_l1_z"] = z(d4.disorder_rate_l1)
    d4["resp_l1_z"] = z(d4.resp_l1)
    d4["inter"] = d4.disorder_l1_z * d4.resp_l1_z
    d4["resp_days_z"] = z(-np.log1p(d4.median_days_close.fillna(d4.median_days_close.median())))
    d4["inter_days"] = d4.disorder_l1_z * d4.resp_days_z

    m2, _ = run_fe("FE2_repeat", d4, "repeat_rate", ["disorder_l1_z"])
    m3, _ = run_fe("FE3_repeat", d4, "repeat_rate", ["disorder_l1_z", "resp_l1_z"])
    m4, fr4 = run_fe("FE4_repeat", d4, "repeat_rate", ["disorder_l1_z", "resp_l1_z", "inter"])
    RESULTS["FE4_disorder"] = fe_term(fr4, "disorder_l1_z")
    RESULTS["FE4_resp"] = fe_term(fr4, "resp_l1_z")
    RESULTS["FE4_inter"] = fe_term(fr4, "inter")

    # simple slopes from FE4 using mixed-style delta method on clustered V
    # reconstruct V from fe_ols internals is awkward; use point estimates + reported SEs from interaction
    b_d = float(fr4.set_index("term").loc["disorder_l1_z", "coef"])
    b_i = float(fr4.set_index("term").loc["inter", "coef"])
    se_d = float(fr4.set_index("term").loc["disorder_l1_z", "se"])
    se_i = float(fr4.set_index("term").loc["inter", "se"])
    # conservative: ignore covariance (overstates SE of slopes a bit if cov negative)
    slope_tab = pd.DataFrame([
        dict(Responsiveness="-1 SD (unresponsive)", Slope=b_d - b_i,
             SE=math.sqrt(se_d ** 2 + se_i ** 2), note="SE ignores cov"),
        dict(Responsiveness="Mean", Slope=b_d, SE=se_d, note="FE4"),
        dict(Responsiveness="+1 SD (responsive)", Slope=b_d + b_i,
             SE=math.sqrt(se_d ** 2 + se_i ** 2), note="SE ignores cov"),
    ])
    savetab(slope_tab, "R_simple_slopes_FE4", TAB)

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    xs = np.linspace(-1.5, 1.5, 61)
    ax.plot(xs, b_d + b_i * xs, color=PAL["blue"], lw=2)
    ax.axhline(0, color="#52514e", lw=1)
    finish(ax, "The disorder-to-repeat link by municipal responsiveness",
           "Responsiveness (z, t−1)", "Slope of lagged disorder",
           "Two-way FE, tract-clustered SEs. Chicago 2011–2019.")
    save(fig, "F4_marginal_effect", FIG)

    dsd = nyc.dropna(subset=["same_domain_rate", "disorder_rate_l1", "resp_l1"]).copy()
    dsd["disorder_l1_z"] = z(dsd.disorder_rate_l1)
    dsd["resp_l1_z"] = z(dsd.resp_l1)
    dsd["inter"] = dsd.disorder_l1_z * dsd.resp_l1_z
    run_fe("FE4_same_domain", dsd, "same_domain_rate", ["disorder_l1_z", "resp_l1_z", "inter"])

    for lab, col in [("FE_phys", "phys_rate_l1"), ("FE_infra", "infra_rate_l1"),
                     ("FE_san", "san_rate_l1")]:
        dd = nyc.dropna(subset=["repeat_rate", col, "resp_l1"]).copy()
        dd["x_z"] = z(dd[col])
        dd["resp_l1_z"] = z(dd.resp_l1)
        dd["inter"] = dd.x_z * dd.resp_l1_z
        run_fe(lab, dd, "repeat_rate", ["x_z", "resp_l1_z", "inter"])

    for lab, col in [("FE_lag1", "disorder_rate_l1"), ("FE_lag3", "disorder_rate_l3"),
                     ("FE_lag6", "disorder_rate_l6")]:
        dd = nyc.dropna(subset=["repeat_rate", col]).copy()
        dd["x_z"] = z(dd[col])
        run_fe(lab, dd, "repeat_rate", ["x_z"])

    # reverse-causality: lead of disorder should not predict current repeat as strongly
    if "disorder_rate_f1" in nyc.columns:
        dd = nyc.dropna(subset=["repeat_rate", "disorder_rate_f1"]).copy()
        dd["x_z"] = z(dd.disorder_rate_f1)
        run_fe("FE_lead1_placebo", dd, "repeat_rate", ["x_z"])

    run_fe("FE4_resp_days", d4, "repeat_rate",
           ["disorder_l1_z", "resp_days_z", "inter_days"])

    # fractional logit: month FE only (tract FE incidental-parameters problem)
    try:
        fl = d4.dropna(subset=["repeat_rate", "disorder_l1_z", "resp_l1_z"]).copy()
        fl["repeat_rate"] = fl.repeat_rate.clip(1e-6, 1 - 1e-6)
        glm = smf.glm("repeat_rate ~ disorder_l1_z * resp_l1_z + C(t)",
                      fl, family=__import__("statsmodels.api", fromlist=["api"]).families.Binomial()).fit()
        gtab = pd.DataFrame({"term": glm.params.index, "coef": glm.params.values,
                             "se": glm.bse.values, "p": glm.pvalues.values})
        gtab = gtab[~gtab.term.str.startswith("C(t)")]
        savetab(gtab, "R_fractional_logit_monthFE", TAB)
        RESULTS["fraclogit_disorder"] = float(glm.params.get("disorder_l1_z", np.nan))
        RESULTS["fraclogit_inter_p"] = float(glm.pvalues.get("disorder_l1_z:resp_l1_z", np.nan))
        print("  fractional logit ok", flush=True)
    except Exception as exc:
        print("  fractional logit skipped:", type(exc).__name__, exc, flush=True)

    savetab(pd.concat(rows, ignore_index=True), "R_fe_robustness", TAB)

    # ---- ward bandwidth ------------------------------------------------------
    cs = (nyc.groupby("GEOID")
          .agg(repeat_rate=("repeat_rate", "mean"), resp=("resp", "mean"),
               disadvantage=("disadvantage", "mean"), stability=("stability", "mean"),
               log_density=("log_density", "mean"),
               lat=("lat", "mean"), lon=("lon", "mean"),
               ward=("ward", "median"))
          .dropna(subset=["repeat_rate", "ward", "lat", "lon"]).reset_index())
    cs["ward"] = cs.ward.round().astype(int)
    trees = {wrd: cKDTree(g[["lon", "lat"]].values) for wrd, g in cs.groupby("ward") if len(g) >= 3}

    def dist_other(row):
        ds = []
        for wrd, t in trees.items():
            if wrd == row.ward:
                continue
            ds.append(t.query([row.lon, row.lat])[0])
        return min(ds) if ds else np.nan

    cs["d_boundary"] = cs.apply(dist_other, axis=1) * 111.0
    band_rows = []
    for km in (1.0, 1.5, 2.0, 3.0):
        band = cs[cs.d_boundary <= km].copy()
        band["resp_hi"] = (band.resp > band.resp.median()).astype(int)
        rd = smf.ols(
            "repeat_rate ~ resp_hi + d_boundary + resp_hi:d_boundary + disadvantage + stability + log_density",
            band).fit(cov_type="HC1")
        band_rows.append(dict(
            km=km, N=int(len(band)),
            resp_hi=float(rd.params.get("resp_hi", np.nan)),
            se=float(rd.bse.get("resp_hi", np.nan)),
            p=float(rd.pvalues.get("resp_hi", np.nan)),
        ))
        print(f"  ward {km} km: N={len(band)} resp_hi={rd.params.get('resp_hi'):.4f} p={rd.pvalues.get('resp_hi'):.3f}",
              flush=True)
    savetab(pd.DataFrame(band_rows), "R_ward_bandwidth", TAB)

    band = cs[cs.d_boundary <= 1.5].copy()
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.scatter(band.d_boundary, band.repeat_rate, s=12, alpha=0.35, color=PAL["blue"], linewidths=0)
    finish(ax, "Repeat complaining near aldermanic ward edges",
           "Distance to nearest other-ward tract (km)", "Mean tract repeat rate",
           "Bandwidth 1.5 km. Chicago wards, 2011–2019 tract means.")
    save(fig, "F9_ward_boundary", FIG)

    # ---- ZHVI plot -----------------------------------------------------------
    hp = nyc.dropna(subset=["log_zhvi", "log_zhvi_f12", "phys_rate_l6"]).copy()
    if len(hp) > 1000:
        hp["d_log_zhvi"] = hp.log_zhvi_f12 - hp.log_zhvi
        fig, ax = plt.subplots(figsize=(5.2, 3.3))
        q = pd.qcut(hp.phys_rate_l6, 8, duplicates="drop")
        gg = hp.groupby(q, observed=True).d_log_zhvi.mean()
        ax.plot(range(len(gg)), gg.values, color=PAL["blue"], marker="o")
        ax.axhline(0, color="#52514e", lw=1)
        finish(ax, "Subsequent 12-month ZHVI change by lagged physical disorder",
               "Octile of physical-disorder rate (t−6)", "Mean Δ log ZHVI",
               "Chicago ZIP ZHVI joined to tracts.")
        save(fig, "F12_zhvi_disorder", FIG)

    # ---- LightGBM + SHAP figures (descriptive) -------------------------------
    try:
        import lightgbm as lgb
        import shap
        from sklearn.metrics import r2_score
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
        model = lgb.LGBMRegressor(
            n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=60,
            subsample=0.85, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=7, verbose=-1, **LGB_KW)
        model.fit(tr[FEATS], tr.repeat_rate, eval_set=[(te[FEATS], te.repeat_rate)],
                  eval_metric="l2", callbacks=[lgb.early_stopping(50, verbose=False)])
        r2h = float(r2_score(te.repeat_rate, model.predict(te[FEATS])))
        RESULTS["r2_lgbm_holdout"] = r2h
        print(f"  LightGBM 2019 holdout R2={r2h:.3f}", flush=True)
        samp = te[FEATS].sample(min(4000, len(te)), random_state=3)
        sv = shap.TreeExplainer(model).shap_values(samp)
        imp = (pd.DataFrame({"feature": FEATS, "mean_abs_shap": np.abs(sv).mean(axis=0)})
               .sort_values("mean_abs_shap", ascending=False))
        savetab(imp, "R_shap_importance", TAB)
        top = imp.head(12).iloc[::-1]
        fig, ax = plt.subplots(figsize=(5.6, 4.0))
        ax.barh(top.feature, top.mean_abs_shap, color=PAL["blue"], height=0.62)
        ax.grid(axis="y", visible=False)
        finish(ax, "What tracks repeat complaining (mean |SHAP|)", "Mean |SHAP|", None,
               "LightGBM; SHAP on 2019 months. Descriptive.")
        save(fig, "F5_shap_importance", FIG)
        if "resp_l1" in FEATS:
            ki = FEATS.index("resp_l1")
            sh = pd.DataFrame({"GEOID": te.loc[samp.index, "GEOID"].values,
                               "lat": te.loc[samp.index, "lat"].values,
                               "lon": te.loc[samp.index, "lon"].values,
                               "shap": sv[:, ki]})
            tsh = sh.groupby("GEOID").agg(shap=("shap", "mean"), lat=("lat", "mean"), lon=("lon", "mean")).reset_index()
            fig, ax = plt.subplots(figsize=(5.2, 5.0))
            sc = ax.scatter(tsh.lon, tsh.lat, c=tsh.shap, s=14, cmap="coolwarm", linewidths=0)
            plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            ax.set_aspect("equal", adjustable="datalim")
            finish(ax, "Where responsiveness SHAP is largest", "Longitude", "Latitude",
                   "Tract-mean SHAP for lagged responsiveness. Chicago.")
            save(fig, "F7_spatial_shap_map", FIG)
    except Exception as exc:
        print("  SHAP skipped:", type(exc).__name__, exc, flush=True)

    n_fig = len(list(FIG.glob("*.png")))
    RESULTS["n_fig"] = n_fig
    (TAB / "results_headline_robust.json").write_text(
        json.dumps(RESULTS, indent=2, default=float), encoding="utf-8")
    print("DONE figures", n_fig, "->", FIG, flush=True)
    print("DONE tables ->", TAB, flush=True)
    print("did NOT write tables/results_headline.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
