from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
STRUMINSKY = ROOT / "struminsky"
if str(STRUMINSKY) not in sys.path:
    sys.path.insert(0, str(STRUMINSKY))

from cli_scripts.rstn_sagamore_cli import aggregate_minute, fetch_network_day  # noqa: E402


DATASET = HERE / "sep_probability_dataset_cycles23_24_25_c1_real_wind_srs_xrs_hmi.csv"
OUT_DIR = HERE / "abubakr_no_synthetic_reproduction"
RSTN_DAY_DIR = OUT_DIR / "rstn_network_days"
SEED = 42
START = pd.Timestamp("1997-01-01T00:00:00Z")
END = pd.Timestamp("2023-01-01T00:00:00Z")

RSTN_FREQS = {
    "245": "Radio_245MHz",
    "410": "Radio_410MHz",
    "606": "Radio_610MHz",  # Sagamore K7O uses 610 MHz; AbuBakr table reports 606 MHz.
    "1415": "Radio_1415MHz",
    "2695": "Radio_2695MHz",
    "4995": "Radio_4995MHz",
    "8800": "Radio_8800MHz",
    "15400": "Radio_15400MHz",
}

FLARE_CME_FEATURES = [
    "goes_class_score",
    "log10_goes_peak_flux_wm2",
    "flare_peak_flux_wm2",
    "flare_log_peak_flux_wm2",
    "flare_intensity_score",
    "flare_rise_time_min",
    "flare_duration_min",
    "flare_heliolongitude_deg",
    "flare_abs_heliolongitude_deg",
    "flare_heliolongitude_available",
    "flare_heliolatitude_deg",
    "flare_heliolatitude_available",
    "flare_sxr_integral_wm2s",
    "flare_log_sxr_integral_wm2s",
    "flare_sxr_integral_available",
    "hek_sxr_duration_min",
    "hek_sxr_fluence_proxy_wm2s",
    "log10_hek_sxr_fluence_proxy_wm2s",
    "struminsky_sxr_rise_min",
    "struminsky_sxr_total_duration_min",
    "lon_w_deg",
    "abs_lon_w_deg",
    "west_longitude_flag",
    "connection_factor_w45",
    "hek_cme_speed_kms",
    "hek_cme_width_deg",
    "log1p_hek_cme_speed",
    "log1p_hek_cme_width",
    "hek_cme_shock_proxy",
    "cdaw_cme_angular_width_deg",
    "cdaw_cme_linear_speed_kms",
    "cme_initial_speed_2nd_order",
    "cme_2nd_order_acceleration",
    "cme_speed_20rsun",
    "cme_final_speed_2nd_order",
]

SWEEP_FEATURES = FLARE_CME_FEATURES + [
    "type_ii_flag",
    "type_ii_strength",
    "radio_type_ii_log_power",
    "radio_type_iv_present",
    "radio_type_iv_duration_min",
    "wind_type2_present",
    "wind_type2_count",
    "wind_type2_inverse_delay",
    "wind_type2_delay_from_flare_start_min",
    "wind_type2_duration_max_min",
    "wind_type2_duration_sum_min",
    "wind_type2_bandwidth_max_khz",
    "wind_type2_log_power",
    "wind_radio_combined_score",
    "wind_radio_combined_score_clean",
    "rstn_wind_bridge_score",
    "radio_coherence_score",
    "wind_orphan_penalty",
    "rstn_log1p_total_integral",
    "rstn_log1p_peak_max",
    "rstn_available_count",
    "shock_acceleration_index",
    "wind_shock_survival_index",
    "abubakr_product_ii_proxy",
]

FIXED_FEATURES = FLARE_CME_FEATURES + [
    feature
    for freq in RSTN_FREQS
    for feature in (
        f"rstn_{freq}_available",
        f"rstn_{freq}_peak",
        f"rstn_{freq}_rise_min",
        f"rstn_{freq}_duration_min",
        f"rstn_{freq}_integral",
    )
]


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(0.0, index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0)


def load_base_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATASET)
    for col in ("event_starttime", "event_peaktime", "event_endtime"):
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce", format="mixed")
    df["is_sep_event"] = pd.to_numeric(df["is_sep_event"], errors="coerce").fillna(0).astype(int)
    for col in set(FLARE_CME_FEATURES + SWEEP_FEATURES):
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[(df["event_peaktime"] >= START) & (df["event_peaktime"] < END)].copy()


def m2_filter(df: pd.DataFrame) -> pd.Series:
    return _num(df, "goes_class_score").ge(2000.0)


def cme_filter(df: pd.DataFrame) -> pd.Series:
    return _num(df, "hek_cme_flag").gt(0) | _num(df, "hek_cme_speed_available").gt(0) | _num(df, "cdaw_cme_2nd_order_available").gt(0)


def sweep_radio_filter(df: pd.DataFrame) -> pd.Series:
    return (
        _num(df, "type_ii_flag").gt(0)
        | _num(df, "radio_type_ii_log_power").gt(0)
        | _num(df, "wind_type2_present").gt(0)
        | _num(df, "wind_type2_count").gt(0)
        | _num(df, "wind_radio_combined_score").gt(0)
    )


def rstn_day_path(day: pd.Timestamp) -> Path:
    return RSTN_DAY_DIR / f"{day:%Y%m%d}.csv"


def fetch_one_rstn_day(day: pd.Timestamp, *, force: bool = False) -> tuple[str, int, str]:
    day = pd.Timestamp(day).tz_localize(None) if pd.Timestamp(day).tzinfo else pd.Timestamp(day)
    path = rstn_day_path(day)
    missing = path.with_suffix(".missing")
    if path.exists() and not force:
        try:
            return str(day.date()), len(pd.read_csv(path)), "cached"
        except Exception:
            pass
    if missing.exists() and not force:
        return str(day.date()), 0, "missing_cached"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        rows = fetch_network_day(day.to_pydatetime())
        rstn = aggregate_minute(rows)
        if rstn.empty:
            missing.write_text("empty", encoding="utf-8")
            return str(day.date()), 0, "empty"
        rstn["DateTime"] = pd.to_datetime(rstn["DateTime"], utc=True, errors="coerce").dt.tz_convert(None)
        rstn = rstn.dropna(subset=["DateTime"]).sort_values("DateTime")
        rstn.to_csv(path, index=False)
        if missing.exists():
            missing.unlink()
        return str(day.date()), len(rstn), "fetched"
    except Exception as exc:
        missing.write_text(f"error: {exc}", encoding="utf-8")
        return str(day.date()), 0, f"error:{exc}"


def fetch_rstn_for_events(df: pd.DataFrame, *, jobs: int, max_days: int | None, force: bool) -> pd.DataFrame:
    days = pd.to_datetime(df["event_starttime"], utc=True, errors="coerce").dt.floor("D").dropna().drop_duplicates().sort_values()
    if max_days is not None:
        days = days.head(max_days)
    rows = []
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
        futs = {ex.submit(fetch_one_rstn_day, day, force=force): day for day in days}
        for fut in as_completed(futs):
            day, nrows, status = fut.result()
            rows.append({"day": day, "rows": nrows, "status": status})
    out = pd.DataFrame(rows).sort_values("day")
    out.to_csv(OUT_DIR / "rstn_fetch_log.csv", index=False)
    return out


def add_rstn_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for freq in RSTN_FREQS:
        out[f"rstn_{freq}_available"] = 0.0
        out[f"rstn_{freq}_peak"] = np.nan
        out[f"rstn_{freq}_rise_min"] = np.nan
        out[f"rstn_{freq}_duration_min"] = np.nan
        out[f"rstn_{freq}_integral"] = np.nan

    cache: dict[str, pd.DataFrame] = {}
    for idx, row in out.iterrows():
        start = pd.Timestamp(row["event_starttime"])
        end = pd.Timestamp(row["event_endtime"])
        if pd.isna(start):
            continue
        start_naive = start.tz_convert("UTC").tz_localize(None) if start.tzinfo else start
        if pd.isna(end) or end <= start:
            end_naive = start_naive + pd.Timedelta(minutes=30)
        else:
            end_naive = end.tz_convert("UTC").tz_localize(None) if end.tzinfo else end
        day_key = str(start_naive.date())
        if day_key not in cache:
            path = rstn_day_path(pd.Timestamp(start_naive.date()))
            cache[day_key] = pd.read_csv(path, parse_dates=["DateTime"]) if path.exists() else pd.DataFrame()
        rstn = cache[day_key]
        if rstn.empty or "DateTime" not in rstn.columns:
            continue
        t = pd.to_datetime(rstn["DateTime"], utc=True, errors="coerce").dt.tz_convert(None)
        sub = rstn[(t >= start_naive) & (t <= end_naive)]
        if sub.empty:
            continue
        sub_t = t.loc[sub.index]
        for freq, col in RSTN_FREQS.items():
            if col not in sub.columns:
                continue
            values = pd.to_numeric(sub[col], errors="coerce").dropna()
            values = values[values > 0]
            if values.empty:
                continue
            peak_idx = values.idxmax()
            out.at[idx, f"rstn_{freq}_available"] = 1.0
            out.at[idx, f"rstn_{freq}_peak"] = float(values.max())
            out.at[idx, f"rstn_{freq}_rise_min"] = float((sub_t.loc[peak_idx] - start_naive).total_seconds() / 60.0)
            out.at[idx, f"rstn_{freq}_duration_min"] = float(len(values))
            out.at[idx, f"rstn_{freq}_integral"] = float(values.sum())
    return out


def add_proxy_products(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    sxr = pd.to_numeric(out["hek_sxr_fluence_proxy_wm2s"], errors="coerce")
    radio = (
        pd.to_numeric(out["radio_type_ii_log_power"], errors="coerce").fillna(0.0)
        + pd.to_numeric(out["wind_type2_log_power"], errors="coerce").fillna(0.0)
        + pd.to_numeric(out["wind_type2_duration_sum_min"], errors="coerce").fillna(0.0)
    )
    out["abubakr_product_ii_proxy"] = sxr * radio
    return out


def median_impute_from_train(x_train: pd.DataFrame, x_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    med = x_train.median(numeric_only=True)
    x_train = x_train.fillna(med).fillna(0.0)
    x_test = x_test.fillna(med).fillna(0.0)
    return x_train, x_test


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    pod = tp / max(tp + fn, 1)
    far = fp / max(tp + fp, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2 * precision * pod / max(precision + pod, 1e-12)
    tss = pod - fp / max(fp + tn, 1)
    hss_num = 2 * (tp * tn - fp * fn)
    hss_den = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    return {
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


def run_rf_cv(df: pd.DataFrame, features: list[str], dataset_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df.copy().reset_index(drop=True)
    y = work["is_sep_event"].to_numpy(dtype=int)
    n_splits = min(5, int(np.bincount(y).min())) if len(np.unique(y)) == 2 else 0
    if n_splits < 2:
        return pd.DataFrame(), pd.DataFrame()
    outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    param_dist = {
        "n_estimators": [300, 500, 800],
        "max_depth": [4, 6, 8, 10, None],
        "min_samples_leaf": [1, 2, 3, 5, 8],
        "max_features": ["sqrt", "log2", 0.5],
        "class_weight": ["balanced", "balanced_subsample"],
    }
    metric_rows = []
    pred_rows = []
    importance_rows = []
    for fold, (train_idx, test_idx) in enumerate(outer.split(work[features], y), start=1):
        x_train, x_test = median_impute_from_train(work.loc[train_idx, features], work.loc[test_idx, features])
        y_train = y[train_idx]
        y_test = y[test_idx]
        inner = StratifiedKFold(n_splits=min(5, int(np.bincount(y_train).min())), shuffle=True, random_state=SEED + fold)
        search = RandomizedSearchCV(
            RandomForestClassifier(random_state=SEED + fold, n_jobs=-1),
            param_distributions=param_dist,
            n_iter=20,
            scoring="f1",
            cv=inner,
            random_state=SEED + fold,
            n_jobs=-1,
            refit=True,
        )
        search.fit(x_train, y_train)
        model = search.best_estimator_
        y_prob = model.predict_proba(x_test)[:, 1]
        y_pred = model.predict(x_test)
        row = metrics(y_test, y_pred)
        row.update({"dataset": dataset_name, "model": "RF_no_SMOTE", "fold": fold, "n_train": len(train_idx), "n_test": len(test_idx)})
        row.update({f"best_{k}": str(v) for k, v in search.best_params_.items()})
        metric_rows.append(row)
        fold_pred = work.loc[test_idx, ["hek_event_id", "event_starttime", "event_peaktime", "flare_class", "is_sep_event"]].copy()
        fold_pred["dataset"] = dataset_name
        fold_pred["fold"] = fold
        fold_pred["rf_probability"] = y_prob
        fold_pred["rf_prediction"] = y_pred
        pred_rows.append(fold_pred)
        for feature, importance in zip(features, model.feature_importances_):
            importance_rows.append({"dataset": dataset_name, "fold": fold, "feature": feature, "importance": float(importance)})
    metrics_df = pd.DataFrame(metric_rows)
    preds_df = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    pd.DataFrame(importance_rows).groupby(["dataset", "feature"], as_index=False)["importance"].mean().sort_values(
        ["dataset", "importance"], ascending=[True, False]
    ).to_csv(OUT_DIR / f"{dataset_name}_rf_feature_importance.csv", index=False)
    return metrics_df, preds_df


def coverage(df: pd.DataFrame, features: list[str], dataset_name: str) -> pd.DataFrame:
    rows = []
    for feature in features:
        s = pd.to_numeric(df.get(feature, pd.Series(np.nan, index=df.index)), errors="coerce")
        rows.append(
            {
                "dataset": dataset_name,
                "feature": feature,
                "available_rows": int(s.notna().sum()),
                "missing_rows": int(s.isna().sum()),
                "coverage": float(s.notna().mean()) if len(s) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="AbuBakr-style no-synthetic reproduction on real cached/downloaded data.")
    parser.add_argument("--fetch-rstn", action="store_true", help="Download missing multi-station RSTN day files for M2+ events.")
    parser.add_argument("--rstn-jobs", type=int, default=8)
    parser.add_argument("--max-fetch-days", type=int, default=None)
    parser.add_argument("--force-rstn", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = add_proxy_products(load_base_dataset())
    m2 = df[m2_filter(df)].copy()
    if args.fetch_rstn:
        fetch_rstn_for_events(m2, jobs=args.rstn_jobs, max_days=args.max_fetch_days, force=args.force_rstn)
    m2 = add_rstn_features(m2)

    cme = cme_filter(m2)
    sweep_radio = sweep_radio_filter(m2)
    fixed_radio = pd.Series(False, index=m2.index)
    for freq in RSTN_FREQS:
        fixed_radio = fixed_radio | pd.to_numeric(m2[f"rstn_{freq}_available"], errors="coerce").fillna(0).gt(0)

    sweep = m2[cme & sweep_radio].copy()
    fixed = m2[cme & fixed_radio].copy()
    hybrid = m2[cme & (sweep_radio | fixed_radio)].copy()
    datasets = {
        "sweep_frequency_no_synthetic": (sweep, SWEEP_FEATURES),
        "fixed_frequency_no_synthetic": (fixed, FIXED_FEATURES),
        "hybrid_radio_no_synthetic": (hybrid, sorted(set(SWEEP_FEATURES + FIXED_FEATURES))),
    }

    all_metrics = []
    all_preds = []
    all_coverage = []
    summary = {"base_rows_1997_2022": int(len(df)), "m2_rows": int(len(m2)), "m2_sep": int(m2["is_sep_event"].sum())}
    for name, (part, features) in datasets.items():
        part.to_csv(OUT_DIR / f"{name}_dataset.csv", index=False)
        all_coverage.append(coverage(part, features, name))
        summary[name] = {"rows": int(len(part)), "sep": int(part["is_sep_event"].sum()), "non_sep": int(len(part) - part["is_sep_event"].sum())}
        metric_df, pred_df = run_rf_cv(part, features, name)
        if not metric_df.empty:
            metric_df.to_csv(OUT_DIR / f"{name}_rf_cv_metrics.csv", index=False)
            pred_df.to_csv(OUT_DIR / f"{name}_rf_cv_predictions.csv", index=False)
            all_metrics.append(metric_df)
            all_preds.append(pred_df)

    if all_metrics:
        metrics_all = pd.concat(all_metrics, ignore_index=True)
        metrics_all.to_csv(OUT_DIR / "rf_no_synthetic_cv_metrics_all.csv", index=False)
        mean = metrics_all.groupby(["dataset", "model"], as_index=False).agg(
            F1_mean=("F1", "mean"),
            F1_std=("F1", "std"),
            POD_mean=("POD", "mean"),
            POD_std=("POD", "std"),
            FAR_mean=("FAR", "mean"),
            FAR_std=("FAR", "std"),
            TSS_mean=("TSS", "mean"),
            TSS_std=("TSS", "std"),
            HSS_mean=("HSS", "mean"),
            HSS_std=("HSS", "std"),
            TP_sum=("TP", "sum"),
            FP_sum=("FP", "sum"),
            FN_sum=("FN", "sum"),
            TN_sum=("TN", "sum"),
        )
        mean["POD_from_sum"] = mean["TP_sum"] / (mean["TP_sum"] + mean["FN_sum"]).clip(lower=1)
        mean["FAR_from_sum"] = mean["FP_sum"] / (mean["TP_sum"] + mean["FP_sum"]).clip(lower=1)
        mean["precision_from_sum"] = mean["TP_sum"] / (mean["TP_sum"] + mean["FP_sum"]).clip(lower=1)
        mean["F1_from_sum"] = 2 * mean["precision_from_sum"] * mean["POD_from_sum"] / (
            mean["precision_from_sum"] + mean["POD_from_sum"]
        ).clip(lower=1e-12)
        mean.to_csv(OUT_DIR / "rf_no_synthetic_cv_summary.csv", index=False)
        summary["cv_summary"] = mean.to_dict(orient="records")
    if all_preds:
        pd.concat(all_preds, ignore_index=True).to_csv(OUT_DIR / "rf_no_synthetic_cv_predictions_all.csv", index=False)
    if all_coverage:
        pd.concat(all_coverage, ignore_index=True).to_csv(OUT_DIR / "feature_coverage.csv", index=False)

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
