"""All numerical experiments reported in the paper (offline, deterministic seeds).

Outputs go to results/.  Runtime: ~20 min on a laptop (4 cores).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sepforecast import models as M  # noqa: E402

RES = ROOT / "results"
N_REPEATS = 10
N_REPEATS_ABLATION = 5
CHRONO_SPLIT = pd.Timestamp("2011-01-01", tz="UTC")
MODELS = {"catboost": M.ALL_FEATURES, "rf": M.ALL_FEATURES, "logreg": M.BASELINE_FEATURES}


def load() -> pd.DataFrame:
    d = pd.read_csv(ROOT / "data/processed/sep_m2_events_clean.csv", low_memory=False)
    for c in ("t_start", "t_peak", "t_end", "spe_onset"):
        d[c] = pd.to_datetime(d[c], utc=True, format="mixed")
    return d


def repeated_cv(d: pd.DataFrame) -> None:
    y = d["y_sep"].to_numpy()
    rows, preds = [], {"event_key": d["event_key"], "y_sep": y}
    alerts = {}
    for kind, feats in MODELS.items():
        for r in range(N_REPEATS):
            out = M.nested_cv(d, y, kind, feats, seed=r)
            base = M.prob_scores(y, out["prob"])
            for s in ("TSS", "HSS"):
                row = {"model": kind, "repeat": r, "threshold_rule": f"max {s} (inner CV)",
                       "threshold_mean": float(np.mean(out["thr"][s])), **base, **M.contingency(y, out["yhat"][s])}
                rows.append(row)
                alerts.setdefault(f"{kind}_{s}", []).append(out["yhat"][s])
            if r == 0:
                preds[f"{kind}_prob"] = out["prob"]
                for s in ("TSS", "HSS"):
                    preds[f"{kind}_alert_{s}"] = out["yhat"][s]
            print(kind, r, round(base["ROC_AUC"], 3), flush=True)
    cv = pd.DataFrame(rows)
    cv.to_csv(RES / "cv_repeats.csv", index=False)
    num = ["ROC_AUC", "PR_AP", "BS", "BSS", "POD", "FAR", "POFD", "TSS", "HSS", "CSI", "TP", "FP", "FN", "threshold_mean"]
    summ = cv.groupby(["model", "threshold_rule"])[num].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.reset_index().to_csv(RES / "cv_summary.csv", index=False)
    p = pd.DataFrame(preds)
    p.to_csv(RES / "oof_predictions_repeat0.csv", index=False)
    freq = pd.DataFrame({k: np.mean(v, axis=0) for k, v in alerts.items()})
    freq.insert(0, "event_key", d["event_key"])
    freq.to_csv(RES / "alert_frequency_over_repeats.csv", index=False)
    ci = []
    for kind in MODELS:
        for s in ("TSS", "HSS"):
            b = M.bootstrap_ci(y, p[f"{kind}_prob"].to_numpy(), p[f"{kind}_alert_{s}"].to_numpy(), seed=1)
            b["model"], b["threshold_rule"] = kind, f"max {s} (inner CV)"
            ci.append(b.reset_index().rename(columns={"index": "metric"}))
    pd.concat(ci).to_csv(RES / "bootstrap_ci_repeat0.csv", index=False)


def chronological(d: pd.DataFrame) -> None:
    tr = (d["t_peak"] < CHRONO_SPLIT).to_numpy()
    y = d["y_sep"].to_numpy()
    rows, preds = [], {"event_key": d.loc[~tr, "event_key"].to_numpy(), "y_sep": y[~tr]}
    for kind, feats in MODELS.items():
        dtr, ytr = d[tr], y[tr]
        probs, inner = [], []
        for seed in range(N_REPEATS):  # seed-averaged forecast for stability
            probs.append(M.Model(kind, feats, seed=seed).fit(dtr, ytr).predict_proba(d[~tr]))
            inner.append(M.nested_cv(dtr, ytr, kind, feats, seed=seed, inner=False)["prob"])
        p, pin = np.mean(probs, axis=0), np.mean(inner, axis=0)
        preds[f"{kind}_prob"] = p
        for s in ("TSS", "HSS"):
            t = M.best_threshold(ytr, pin, s)
            yhat = (p >= t).astype(int)
            ci = M.bootstrap_ci(y[~tr], p, yhat, seed=2)
            row = {"model": kind, "threshold_rule": f"max {s} (training-period CV)", "threshold": t,
                   "n_train": int(tr.sum()), "sep_train": int(ytr.sum()), "n_test": int((~tr).sum()),
                   "sep_test": int(y[~tr].sum()), **M.prob_scores(y[~tr], p, base_rate=ytr.mean()),
                   **M.contingency(y[~tr], yhat)}
            for m in ("POD", "FAR", "TSS", "HSS", "ROC_AUC", "PR_AP"):
                row[f"{m}_lo95"], row[f"{m}_hi95"] = ci.loc[m, "lo95"], ci.loc[m, "hi95"]
            rows.append(row)
            preds[f"{kind}_alert_{s}"] = yhat
    pd.DataFrame(rows).to_csv(RES / "chronological_test.csv", index=False)
    pd.DataFrame(preds).to_csv(RES / "chronological_predictions.csv", index=False)


def ablation(d: pd.DataFrame) -> None:
    y = d["y_sep"].to_numpy()
    variants = {"all features": M.ALL_FEATURES}
    for g, feats in M.FEATURE_GROUPS.items():
        variants[f"without {g}"] = [f for f in M.ALL_FEATURES if f not in feats]
        variants[f"only {g}"] = feats
    variants["logistic baseline set"] = M.BASELINE_FEATURES
    # Sensitivity to the CME association: keep the CME only if it appears within [-60, +180] min of the flare peak.
    strict = d.copy()
    loose = ~strict["cme_delay_min"].between(-60, 180)
    strict.loc[loose, M.FEATURE_GROUPS["cme"]] = np.nan
    variants["all features, strict CME timing"] = M.ALL_FEATURES
    rows = []
    for name, feats in variants.items():
        frame = strict if "strict CME" in name else d
        for r in range(N_REPEATS_ABLATION):
            out = M.nested_cv(frame, y, "catboost", feats, seed=r, inner=False)
            rows.append({"variant": name, "repeat": r, "n_features": len(feats), **M.prob_scores(y, out["prob"])})
        print("ablation", name, flush=True)
    a = pd.DataFrame(rows)
    a.to_csv(RES / "ablation_repeats.csv", index=False)
    s = a.groupby(["variant", "n_features"])[["ROC_AUC", "PR_AP", "BSS"]].agg(["mean", "std"])
    s.columns = [f"{x}_{z}" for x, z in s.columns]
    s.reset_index().to_csv(RES / "ablation_summary.csv", index=False)


def shap_importance(d: pd.DataFrame) -> None:
    y = d["y_sep"].to_numpy()
    x = d[M.ALL_FEATURES].astype(float)
    vals = []
    for seed in range(5):
        m = M.Model("catboost", M.ALL_FEATURES, seed=seed).fit(d, y)
        vals.append(m.est_.get_feature_importance(Pool(x, y), type="ShapValues")[:, :-1])
    sv = np.mean(vals, axis=0)
    imp = pd.DataFrame({"feature": M.ALL_FEATURES, "mean_abs_shap": np.abs(sv).mean(axis=0)})
    imp["group"] = imp["feature"].map({f: g for g, fs in M.FEATURE_GROUPS.items() for f in fs})
    imp.sort_values("mean_abs_shap", ascending=False).to_csv(RES / "shap_importance.csv", index=False)
    out = pd.DataFrame(sv, columns=[f"shap_{f}" for f in M.ALL_FEATURES])
    out.insert(0, "event_key", d["event_key"])
    out.to_csv(RES / "shap_values_full_fit.csv", index=False)


def legacy_comparison(d: pd.DataFrame) -> None:
    """Quantify how each correction changes the skill of the REVISED1 configuration."""
    spec = importlib.util.spec_from_file_location("legacy", ROOT / "provenance/run_recommended_physics_clean_models.py")
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    up = pd.read_csv(ROOT / "data/input/upstream_m2_cme_radio_table.csv", low_memory=False)
    up = legacy.add_recommended_features(up)
    feats = legacy.BASE_FEATURES

    def legacy_cb(seed: int) -> CatBoostClassifier:
        return CatBoostClassifier(loss_function="Logloss", iterations=900, learning_rate=0.025, depth=3,
                                  l2_leaf_reg=30.0, rsm=0.5, random_seed=seed, verbose=False,
                                  allow_writing_files=False, thread_count=4)

    def oof(frame: pd.DataFrame, yv: np.ndarray, seed: int) -> np.ndarray:
        from sklearn.model_selection import StratifiedKFold
        p = np.zeros(len(yv))
        for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(frame, yv)):
            m = legacy_cb(seed + k)
            m.set_params(class_weights=[1.0, (len(a) - yv[a].sum()) / yv[a].sum()])
            m.fit(frame.iloc[a][feats], yv[a])
            p[b] = m.predict_proba(frame.iloc[b][feats])[:, 1]
        return p

    def in_sample_threshold(yv: np.ndarray, p: np.ndarray) -> dict:
        """REVISED1 rule: maximise POD subject to FAR <= 0.67, threshold chosen on the same OOF set."""
        best = None
        for t in np.unique(np.r_[np.linspace(0.01, 0.99, 99), np.linspace(0.001, 0.999, 999)]):
            c = M.contingency(yv, (p >= t).astype(int))
            if c["TP"] > 0 and c["FAR"] <= 0.67 and (best is None or (c["POD"], c["TSS"]) > (best["POD"], best["TSS"])):
                best = {**c, "threshold": t}
        return best

    def nested_threshold(frame: pd.DataFrame, yv: np.ndarray, seed: int) -> dict:
        from sklearn.model_selection import StratifiedKFold
        p = oof(frame, yv, seed)
        yhat = np.zeros(len(yv), dtype=int)
        for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(frame, yv)):
            pin = oof(frame.iloc[a].reset_index(drop=True), yv[a], seed + 1000 + k)
            t = M.best_threshold(yv[a], pin, "TSS")
            yhat[b] = (p[b] >= t).astype(int)
        return {**M.prob_scores(yv, p), **M.contingency(yv, yhat)}

    clean = legacy.add_recommended_features(d.copy())
    configs = {
        "A: upstream table, upstream labels, threshold tuned on OOF (REVISED1 protocol)": (up, up["is_sep_event"].to_numpy(), "in-sample"),
        "B: upstream table, upstream labels, nested threshold": (up, up["is_sep_event"].to_numpy(), "nested"),
        "C: duplicates removed, upstream labels, nested threshold": (clean, clean["upstream_label_max"].to_numpy().astype(int), "nested"),
        "D: duplicates removed, NOAA SPE labels, nested threshold": (clean, clean["y_sep"].to_numpy(), "nested"),
    }
    rows = []
    for name, (frame, yv, mode) in configs.items():
        frame = frame.reset_index(drop=True)
        for r in range(5):
            seed = 42 + r
            if mode == "in-sample":
                p = oof(frame, yv, seed)
                res = {**M.prob_scores(yv, p), **in_sample_threshold(yv, p)}
            else:
                res = nested_threshold(frame, yv, seed)
            rows.append({"configuration": name, "repeat": r, "n_rows": len(frame), "n_sep": int(yv.sum()), **res})
        print("legacy", name, flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(RES / "legacy_comparison_repeats.csv", index=False)
    s = t.groupby(["configuration", "n_rows", "n_sep"])[["ROC_AUC", "PR_AP", "POD", "FAR", "TSS", "HSS"]].agg(["mean", "std"])
    s.columns = [f"{a}_{b}" for a, b in s.columns]
    s.reset_index().to_csv(RES / "legacy_comparison_summary.csv", index=False)


def descriptive(d: pd.DataFrame) -> None:
    pos = d[d["y_sep"] == 1]
    lead_h = (pos["spe_onset"] - pos["t_peak"]).dt.total_seconds() / 3600
    cme_t = pos["t_peak"] + pd.to_timedelta(pos["cme_delay_min"], unit="min")
    info = {
        "lead_time_h_median": float(lead_h.median()),
        "lead_time_h_q25": float(lead_h.quantile(0.25)),
        "lead_time_h_q75": float(lead_h.quantile(0.75)),
        "sep_onset_before_cme_reference_time": int((pos["spe_onset"] < cme_t).sum()),
        "events_per_cycle": d.groupby("cycle")["y_sep"].agg(["size", "sum"]).to_dict(),
        "chrono_train": int((d["t_peak"] < CHRONO_SPLIT).sum()),
        "chrono_test": int((d["t_peak"] >= CHRONO_SPLIT).sum()),
        "cme_delay_min_median_all": float(d["cme_delay_min"].median()),
        "cme_delay_outside_m60_p180": int((~d["cme_delay_min"].between(-60, 180)).sum()),
    }
    num = ["log10_peak_flux_wm2", "log10_sxr_fluence_jm2", "lon_w_deg_clean", "cme_speed_kms", "cme_width_deg",
           "rstn_1415_log_peak", "rstn_8800_log_peak", "type_ii_ground", "srs_area_msh", "flare_history_24h"]
    stats = d.groupby("y_sep")[num].agg(["median", lambda s: s.quantile(0.25), lambda s: s.quantile(0.75), "count"])
    stats.to_csv(RES / "feature_distributions_by_class.csv")
    (RES / "descriptive.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")


def subthreshold(d: pd.DataFrame) -> None:
    """Are persistent false alarms followed by weaker (>=1 pfu) proton enhancements? (SINP MSU, 2010-2022)."""
    sinp = pd.read_csv(ROOT / "data/external/sinp/sinp_sep_catalog_c24_c25.csv")
    flare_t = pd.to_datetime(sinp["flare_t0_utc"], utc=True, errors="coerce")
    start = pd.to_datetime(sinp["start_utc"], utc=True)
    freq = pd.read_csv(RES / "alert_frequency_over_repeats.csv")
    p = d.merge(freq, on="event_key")
    p = p[p["t_peak"] >= start.min() - pd.Timedelta(days=1)]

    def jmax(t: pd.Timestamp) -> float:
        m = flare_t.notna() & ((flare_t - t).abs() <= pd.Timedelta(minutes=90))
        return float(sinp.loc[m, "jmax_pfu"].max()) if m.any() else np.nan

    p = p.assign(sinp_jmax_pfu=[jmax(t) for t in p["t_peak"]])
    neg = p[p["y_sep"] == 0]
    fa = neg["catboost_HSS"] >= 0.5
    info = {
        "period_start": str(start.min()), "flares": len(p), "sep_flares": int(p["y_sep"].sum()),
        "sep_flares_with_sinp_match": int(p.loc[p["y_sep"] == 1, "sinp_jmax_pfu"].notna().sum()),
        "persistent_false_alarms": int(fa.sum()),
        "persistent_false_alarms_with_sinp_event": int(neg.loc[fa, "sinp_jmax_pfu"].notna().sum()),
        "persistent_false_alarms_with_sinp_ge10pfu": int((neg.loc[fa, "sinp_jmax_pfu"] >= 10).sum()),
        "other_negatives": int((~fa).sum()),
        "other_negatives_with_sinp_event": int(neg.loc[~fa, "sinp_jmax_pfu"].notna().sum()),
    }
    neg.loc[fa, ["event_key", "t_peak", "goes_class_final", "lon_w_deg_clean", "cme_speed_kms", "cme_width_deg",
                 "catboost_HSS", "sinp_jmax_pfu"]].to_csv(RES / "persistent_false_alarms_2010_2022.csv", index=False)
    (RES / "subthreshold_check.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def main() -> None:
    RES.mkdir(exist_ok=True)
    d = load()
    steps = sys.argv[1:] or ["descriptive", "cv", "chrono", "ablation", "shap", "legacy", "subthreshold"]
    if "descriptive" in steps:
        descriptive(d)
    if "cv" in steps:
        repeated_cv(d)
    if "chrono" in steps:
        chronological(d)
    if "ablation" in steps:
        ablation(d)
    if "shap" in steps:
        shap_importance(d)
    if "legacy" in steps:
        legacy_comparison(d)
    if "subthreshold" in steps:
        subthreshold(d)


if __name__ == "__main__":
    main()
