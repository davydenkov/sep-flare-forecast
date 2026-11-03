from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold


HERE = Path(__file__).resolve().parent
IN_DIR = HERE / "abubakr_no_synthetic_reproduction"
OUT_DIR = HERE / "recommended_physics_clean_models"
SEED = 42

BASE_FEATURES = [
    "hek_cme_shock_proxy",
    "hek_cme_speed_kms",
    "hek_cme_width_deg",
    "log1p_hek_cme_speed",
    "hek_sxr_fluence_proxy_wm2s",
    "log10_hek_sxr_fluence_proxy_wm2s",
    "goes_class_score",
    "log10_goes_peak_flux_wm2",
    "xs_ratio_10m",
    "goes_ratio_derivative",
    "sxr_peak_flux_derivative_max",
    "connection_factor_w45",
    "lon_w_deg",
    "wind_type2_inverse_delay_clean",
    "wind_type2_log_power_clean",
    "wind_radio_combined_score_clean",
    "rstn_1415_peak",
    "rstn_2695_peak",
    "rstn_410_integral",
    "rstn_4995_peak",
    "rstn_wind_bridge_score",
    "radio_coherence_score",
    "noaa_energy_proxy",
    "noaa_complexity_proxy",
    "flare_history_weight_24h",
]

STRICT_FEATURES = [
    "hek_cme_shock_proxy",
    "hek_cme_speed_kms",
    "hek_cme_width_deg",
    "hek_sxr_fluence_proxy_wm2s",
    "goes_class_score",
    "xs_ratio_10m",
    "goes_ratio_derivative",
    "sxr_peak_flux_derivative_max",
    "connection_factor_w45",
    "wind_type2_inverse_delay_clean",
    "wind_radio_combined_score_clean",
    "rstn_1415_peak",
    "rstn_2695_peak",
    "rstn_410_integral",
    "rstn_wind_bridge_score",
    "radio_coherence_score",
    "noaa_energy_proxy",
    "flare_history_weight_24h",
]


def n(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(0.0, index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0)


def load_dataset() -> pd.DataFrame:
    path = IN_DIR / "hybrid_radio_no_synthetic_dataset.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run AbuBakr no-synthetic reproduction first: {path}")
    df = pd.read_csv(path)
    for col in ("event_starttime", "event_peaktime", "event_endtime"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce", format="mixed")
    df["is_sep_event"] = n(df, "is_sep_event").astype(int)
    return add_recommended_features(df)


def add_recommended_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    wind_noise = n(out, "is_wind_noise_or_none").gt(0) | n(out, "wind_match_quality_score").le(0.05)
    wind_weak = n(out, "is_wind_weak").gt(0) | ((n(out, "wind_match_quality_score") > 0.05) & (n(out, "wind_match_quality_score") < 0.75))
    for col in [
        "wind_type2_inverse_delay",
        "wind_type2_log_power",
        "wind_radio_combined_score",
        "wind_type2_duration_sum_min",
        "wind_type2_bandwidth_max_khz",
        "shock_acceleration_index",
        "wind_shock_survival_index",
    ]:
        out[f"{col}_clean"] = n(out, col).where(~wind_noise, 0.0)

    rstn_integral_cols = [c for c in out.columns if c.startswith("rstn_") and c.endswith("_integral")]
    rstn_peak_cols = [c for c in out.columns if c.startswith("rstn_") and c.endswith("_peak")]
    rstn_available_cols = [c for c in out.columns if c.startswith("rstn_") and c.endswith("_available")]
    out["rstn_total_integral"] = out[rstn_integral_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1) if rstn_integral_cols else 0.0
    out["rstn_peak_max"] = out[rstn_peak_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).max(axis=1) if rstn_peak_cols else 0.0
    out["rstn_available_count"] = out[rstn_available_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1) if rstn_available_cols else 0.0

    # Coherence proxy: WIND type II is trusted only if there is a high-frequency RSTN bridge.
    high_freq_bridge = (
        n(out, "rstn_1415_available").gt(0)
        | n(out, "rstn_2695_available").gt(0)
        | n(out, "rstn_4995_available").gt(0)
        | n(out, "rstn_8800_available").gt(0)
        | n(out, "rstn_15400_available").gt(0)
    )
    wind_evidence = n(out, "wind_type2_present").gt(0) | n(out, "wind_type2_count").gt(0)
    out["rstn_wind_bridge_score"] = (
        np.log1p(out["rstn_total_integral"].clip(lower=0.0))
        * n(out, "wind_match_quality_score").clip(0.0, 1.0)
        * high_freq_bridge.astype(float)
    )
    out["wind_orphan_penalty"] = (wind_evidence & ~high_freq_bridge & wind_weak).astype(float)
    out["radio_coherence_score"] = np.where(
        out["wind_orphan_penalty"].gt(0),
        0.0,
        n(out, "wind_radio_combined_score_clean") + 0.25 * np.log1p(out["rstn_total_integral"].clip(lower=0.0)),
    )

    out["noaa_energy_proxy"] = n(out, "noaa_ar_area_proxy") * np.log1p(n(out, "goes_class_score"))
    out["noaa_complexity_proxy"] = (
        n(out, "noaa_mag_class_score")
        + 2.0 * n(out, "noaa_delta_spot_flag")
        + 0.01 * n(out, "noaa_ar_area_proxy")
    ) * np.log1p(n(out, "goes_class_score"))
    if "flare_history_weight_24h" not in out.columns:
        out["flare_history_weight_24h"] = 0.0

    # Conservative fallback when GOES short channel is unavailable.
    out["xrs_spectral_hardness_available"] = n(out, "xs_ratio_10m").gt(0).astype(float)
    for col in set(BASE_FEATURES + STRICT_FEATURES):
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def metrics_at_threshold(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, float | int]:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    pod = tp / max(tp + fn, 1)
    far = fp / max(tp + fp, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2 * precision * pod / max(precision + pod, 1e-12)
    tss = pod - fp / max(fp + tn, 1)
    hss_num = 2 * (tp * tn - fp * fn)
    hss_den = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    return {
        "threshold": float(threshold),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "POD": float(pod),
        "FAR": float(far),
        "precision": float(precision),
        "F1": float(f1),
        "TSS": float(tss),
        "HSS": float(hss_num / max(hss_den, 1)),
    }


def median_impute(train_x: pd.DataFrame, test_x: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    med = train_x.median(numeric_only=True)
    return train_x.fillna(med).fillna(0.0), test_x.fillna(med).fillna(0.0)


def fit_predict_oof(df: pd.DataFrame, features: list[str], model_name: str) -> tuple[np.ndarray, pd.DataFrame]:
    y = df["is_sep_event"].to_numpy(dtype=int)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    prob = np.zeros(len(df), dtype=float)
    importance_rows = []
    for fold, (tr, te) in enumerate(cv.split(df[features], y), start=1):
        train_x, test_x = df.loc[tr, features], df.loc[te, features]
        if model_name.startswith("rf"):
            train_x, test_x = median_impute(train_x, test_x)
            model = RandomForestClassifier(
                n_estimators=900,
                max_depth=6 if "strict" in model_name else 8,
                min_samples_leaf=6,
                max_features="sqrt",
                class_weight="balanced_subsample",
                random_state=SEED + fold,
                n_jobs=-1,
            )
            model.fit(train_x, y[tr])
            prob[te] = model.predict_proba(test_x)[:, 1]
            importances = model.feature_importances_
        else:
            model = CatBoostClassifier(
                loss_function="Logloss",
                eval_metric="F1",
                iterations=900,
                learning_rate=0.025,
                depth=3,
                l2_leaf_reg=45.0 if "strict" in model_name else 30.0,
                min_data_in_leaf=35,
                rsm=0.5,
                class_weights=[1.0, min((len(tr) - y[tr].sum()) / max(y[tr].sum(), 1), 25.0)],
                random_seed=SEED + fold,
                allow_writing_files=False,
                verbose=False,
            )
            model.fit(train_x, y[tr])
            prob[te] = model.predict_proba(test_x)[:, 1]
            importances = model.get_feature_importance()
        for feature, importance in zip(features, importances):
            importance_rows.append({"model": model_name, "fold": fold, "feature": feature, "importance": float(importance)})
    return prob, pd.DataFrame(importance_rows)


def evaluate_model(df: pd.DataFrame, prob: np.ndarray, model_name: str, features: list[str]) -> pd.DataFrame:
    y = df["is_sep_event"].to_numpy(dtype=int)
    rows = []
    thresholds = np.unique(np.r_[np.linspace(0.01, 0.99, 99), np.linspace(0.001, 0.999, 999)])
    roc_auc = float(roc_auc_score(y, prob))
    pr_ap = float(average_precision_score(y, prob))
    for threshold in thresholds:
        row = metrics_at_threshold(y, prob, float(threshold))
        row.update({"model": model_name, "n_features": len(features), "ROC_AUC": roc_auc, "PR_AP": pr_ap})
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    df = load_dataset()
    configs = {
        "catboost_clean_strict18": STRICT_FEATURES,
        "catboost_clean_phys25": BASE_FEATURES,
        "rf_clean_strict18": STRICT_FEATURES,
        "rf_clean_phys25": BASE_FEATURES,
    }
    all_metrics = []
    all_importances = []
    predictions = df[["hek_event_id", "event_starttime", "event_peaktime", "flare_class", "is_sep_event"]].copy()
    for model_name, features in configs.items():
        prob, imp = fit_predict_oof(df, features, model_name)
        predictions[f"{model_name}_oof_probability"] = prob
        all_importances.append(imp)
        all_metrics.append(evaluate_model(df, prob, model_name, features))

    metrics = pd.concat(all_metrics, ignore_index=True)
    importances = pd.concat(all_importances, ignore_index=True)
    imp_mean = importances.groupby(["model", "feature"], as_index=False)["importance"].mean().sort_values(
        ["model", "importance"], ascending=[True, False]
    )
    metrics.to_csv(OUT_DIR / "recommended_clean_oof_threshold_metrics.csv", index=False)
    predictions.to_csv(OUT_DIR / "recommended_clean_oof_predictions.csv", index=False)
    imp_mean.to_csv(OUT_DIR / "recommended_clean_feature_importance.csv", index=False)

    preferred = []
    for model, group in metrics.groupby("model"):
        for far_limit in (0.30, 0.40, 0.50, 0.60, 0.67):
            feasible = group[(group["TP"] > 0) & (group["FAR"] <= far_limit)]
            if feasible.empty:
                continue
            best = feasible.sort_values(["POD", "TSS", "HSS"], ascending=[False, False, False]).head(1).copy()
            best["selection"] = f"max_POD_at_FAR<={far_limit:.2f}"
            preferred.append(best)
        best_f1 = group[group["TP"] > 0].sort_values(["F1", "TSS"], ascending=[False, False]).head(1).copy()
        best_f1["selection"] = "max_F1"
        preferred.append(best_f1)
    best_table = pd.concat(preferred, ignore_index=True).sort_values(["FAR", "POD"], ascending=[True, False])
    best_table.to_csv(OUT_DIR / "recommended_clean_best_operating_points.csv", index=False)
    summary = {
        "rows": int(len(df)),
        "sep": int(df["is_sep_event"].sum()),
        "best": best_table.head(30).to_dict(orient="records"),
        "note": "No continuous real peak proton flux target is present in the current table, so regression was not trained.",
    }
    (OUT_DIR / "recommended_clean_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(best_table.head(30).to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
