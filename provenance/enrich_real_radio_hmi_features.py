from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
STRUMINSKY_DIR = ROOT / "struminsky"
CLI_DIR = STRUMINSKY_DIR / "cli_scripts"
for path in (ROOT, STRUMINSKY_DIR, CLI_DIR, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data_paths import hmi_sharp_cache_dir  # noqa: E402


INPUT_CSV = HERE / "sep_probability_dataset_cycles23_24_25_c1_struminsky.csv"
OUTPUT_CSV = HERE / "sep_probability_dataset_cycles23_24_25_c1_real_wind_hmi.csv"
STATUS_CSV = HERE / "real_wind_hmi_enrichment_status.csv"
SRS_CACHE_DIR = ROOT / "data" / "noaa_srs_cache"

WIND_TYPE2_CATALOG_CSV = HERE.parent / "wind_waves_type2_catalog.csv"
WIND_TYPE4_CATALOG_CSV = HERE.parent / "wind_stereo_type4_catalog.csv"
if not WIND_TYPE2_CATALOG_CSV.exists():
    WIND_TYPE2_CATALOG_CSV = HERE.parent / "sep_cascade_portable_inference" / "cache" / "wind_waves_type2_catalog.csv"
if not WIND_TYPE4_CATALOG_CSV.exists():
    WIND_TYPE4_CATALOG_CSV = HERE.parent / "sep_cascade_portable_inference" / "cache" / "wind_stereo_type4_catalog.csv"
WIND_FEATURES = (
    "wind_type2_present",
    "wind_type2_count",
    "wind_match_quality_score",
    "is_wind_strong",
    "is_wind_weak",
    "is_wind_noise_or_none",
    "wind_type2_delay_from_flare_start_min",
    "wind_type2_delay_from_xray_peak_min",
    "wind_type2_inverse_delay",
    "wind_type2_duration_max_min",
    "wind_type2_duration_sum_min",
    "wind_type2_freq_start_max_khz",
    "wind_type2_freq_end_min_khz",
    "wind_type2_bandwidth_max_khz",
    "wind_type2_drift_proxy_max",
    "wind_type2_v_shock_proxy",
    "wind_type2_log_power",
    "wind_type4_present",
    "wind_type4_duration_max_min",
    "wind_type4_duration_sum_min",
    "wind_radio_combined_score",
    "shock_acceleration_index",
    "wind_shock_survival_index",
)
WIND_TYPE2_MIN_DELAY_MIN = float(os.environ.get("SEP_WIND_TYPE2_MIN_DELAY_MIN", "0"))
WIND_TYPE2_STRONG_MAX_DELAY_MIN = float(os.environ.get("SEP_WIND_TYPE2_STRONG_MAX_DELAY_MIN", "30"))
WIND_TYPE2_MAX_DELAY_MIN = float(os.environ.get("SEP_WIND_TYPE2_MAX_DELAY_MIN", "120"))
WIND_TYPE4_MAX_DELAY_MIN = float(os.environ.get("SEP_WIND_TYPE4_MAX_DELAY_MIN", "120"))
WIND_DELAY_MISSING_VALUE = -999.0
NOAA_SRS_FEATURES = (
    "noaa_srs_region_number",
    "noaa_srs_lon_w_deg",
    "noaa_srs_match_delta_lon",
    "noaa_ar_area_proxy",
    "noaa_mag_class_score",
    "noaa_delta_spot_flag",
    "noaa_zurich_class_score",
    "flare_history_weight_24h",
    "hmi_magnetic_flux_proxy_wb",
    "hmi_mean_shear_proxy_deg",
    "hmi_r_value_proxy",
)
SXR_DYNAMIC_FEATURES = (
    "sxr_impulse_index",
    "goes_ratio_derivative",
)
HMI_FEATURES = (
    "hmi_magnetic_flux_wb",
    "hmi_mean_bz",
    "hmi_mean_bt",
    "hmi_mean_gamma",
    "hmi_mean_shear",
    "hmi_area_hem",
    "hmi_mean_pot",
    "hmi_tot_pot",
    "hmi_totusjz",
    "hmi_absnjzh",
    "hmi_shrgt45",
    "hmi_r_value",
)


def _num(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _parse_bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def candidate_gate(df: pd.DataFrame) -> pd.Series:
    """Broad real-time prefilter for running expensive enrichments and classifiers."""
    goes = pd.to_numeric(df.get("goes_class_score"), errors="coerce")
    west = pd.to_numeric(df.get("west_longitude_flag"), errors="coerce").fillna(0).gt(0)
    type2 = pd.to_numeric(df.get("type_ii_flag"), errors="coerce").fillna(0).gt(0)
    type4 = pd.to_numeric(df.get("radio_type_iv_present"), errors="coerce").fillna(0).gt(0)
    long_decay = pd.to_numeric(df.get("struminsky_long_decay_flag"), errors="coerce").fillna(0).gt(0)
    cme_speed = pd.to_numeric(df.get("hek_cme_speed_kms"), errors="coerce")
    cme_width = pd.to_numeric(df.get("hek_cme_width_deg"), errors="coerce")
    energetic_cme = (cme_speed.ge(700.0) & cme_width.ge(60.0)) | cme_width.ge(180.0)
    return goes.ge(1000.0) | type2 | type4 | energetic_cme | (west & goes.ge(500.0)) | (long_decay & goes.ge(500.0))


def _load_wind_type2_catalog() -> pd.DataFrame:
    if not WIND_TYPE2_CATALOG_CSV.exists():
        return pd.DataFrame(columns=["start_utc", "end_utc", "freq_start_khz", "freq_end_khz", "duration_min"])
    cat = pd.read_csv(WIND_TYPE2_CATALOG_CSV)
    cat["start_utc"] = pd.to_datetime(cat["start_utc"], utc=True, errors="coerce", format="mixed")
    cat["end_utc"] = pd.to_datetime(cat["end_utc"], utc=True, errors="coerce", format="mixed")
    return cat.dropna(subset=["start_utc", "end_utc"]).reset_index(drop=True)


def _load_wind_type4_catalog() -> pd.DataFrame:
    if not WIND_TYPE4_CATALOG_CSV.exists():
        return pd.DataFrame(columns=["start_utc", "end_utc", "duration_min"])
    cat = pd.read_csv(WIND_TYPE4_CATALOG_CSV)
    cat["start_utc"] = pd.to_datetime(cat["start_utc"], utc=True, errors="coerce", format="mixed")
    cat["end_utc"] = pd.to_datetime(cat["end_utc"], utc=True, errors="coerce", format="mixed")
    return cat.dropna(subset=["start_utc", "end_utc"]).reset_index(drop=True)


def _mag_class_score(value: object) -> float:
    s = str(value or "").strip().lower().replace(" ", "")
    if not s or s == "nan":
        return np.nan
    score = 0.0
    if "alpha" in s:
        score = max(score, 1.0)
    if "beta" in s:
        score = max(score, 2.0)
    if "gamma" in s:
        score = max(score, 3.0)
    if "delta" in s:
        score = max(score, 5.0 if "gamma" in s else 4.0)
    return score if score > 0.0 else np.nan


def _zurich_class_score(value: object) -> float:
    s = str(value or "").strip().upper()
    if not s:
        return np.nan
    return {
        "A": 1.0,
        "B": 2.0,
        "C": 3.0,
        "D": 4.0,
        "E": 5.0,
        "F": 6.0,
        "H": 2.5,
    }.get(s[0], np.nan)


def _srs_cache_path(day: pd.Timestamp) -> Path:
    return SRS_CACHE_DIR / f"{day:%Y}" / f"{day:%Y%m%d}SRS.txt"


def _srs_url(day: pd.Timestamp) -> str:
    return (
        "https://www.ngdc.noaa.gov/stp/space-weather/swpc-products/daily_reports/"
        f"solar_region_summaries/{day:%Y}/{day:%m}/{day:%Y%m%d}SRS.txt"
    )


def _fetch_srs_text(day: pd.Timestamp) -> str | None:
    path = _srs_cache_path(day)
    missing = path.with_suffix(".missing")
    if path.exists():
        return path.read_text(encoding="utf-8", errors="ignore")
    if missing.exists():
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        response = requests.get(_srs_url(day), timeout=float(os.environ.get("SEP_REALDATA_SRS_TIMEOUT", "6")))
    except Exception as exc:
        missing.write_text(f"error: {exc}", encoding="utf-8")
        return None
    if response.status_code != 200 or "Regions with Sunspots" not in response.text:
        missing.write_text(f"status: {response.status_code}", encoding="utf-8")
        return None
    path.write_text(response.text, encoding="utf-8")
    return response.text


def _parse_srs_regions(text: str | None, day: pd.Timestamp) -> pd.DataFrame:
    if not text:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    in_regions = False
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("I.  Regions with Sunspots") or line.startswith("I. Regions with Sunspots"):
            in_regions = True
            continue
        if in_regions and (line.startswith("IA.") or line.startswith("II.")):
            break
        if not in_regions or not line or line.startswith("Nmbr"):
            continue
        parts = line.split()
        if len(parts) < 8 or not parts[0].isdigit():
            continue
        loc = parts[1].upper()
        if len(loc) < 5 or loc[0] not in "NS" or loc[3] not in "EW":
            continue
        try:
            lat = float(loc[1:3]) * (1.0 if loc[0] == "N" else -1.0)
            lon = float(loc[4:]) * (1.0 if loc[3] == "W" else -1.0)
            area = float(parts[3])
        except ValueError:
            continue
        mag_type = " ".join(parts[7:])
        rows.append(
            {
                "srs_day": day.normalize(),
                "noaa_srs_region_number": float(parts[0]),
                "noaa_srs_lat_deg": lat,
                "noaa_srs_lon_w_deg": lon,
                "noaa_ar_area_proxy": area,
                "noaa_zurich_class_score": _zurich_class_score(parts[4]),
                "noaa_mag_class_score": _mag_class_score(mag_type),
                "noaa_delta_spot_flag": float("delta" in mag_type.lower()),
            }
        )
    return pd.DataFrame(rows)


def _load_or_fetch_srs_for_days(days: list[pd.Timestamp]) -> pd.DataFrame:
    frames = []
    jobs = int(os.environ.get("SEP_REALDATA_SRS_JOBS", "16"))
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        futures = {pool.submit(_fetch_srs_text, pd.Timestamp(day)): pd.Timestamp(day) for day in days}
        for future in as_completed(futures):
            day = futures[future]
            try:
                text = future.result()
            except Exception:
                text = None
            parsed = _parse_srs_regions(text, day)
            if not parsed.empty:
                frames.append(parsed)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _attach_srs_features(df: pd.DataFrame, srs: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in NOAA_SRS_FEATURES:
        out[col] = np.nan
    if srs.empty:
        return out
    by_day = {pd.Timestamp(day): part.copy() for day, part in srs.groupby("srs_day")}
    for row in out.itertuples():
        peak = pd.Timestamp(row.event_peaktime)
        if pd.isna(peak):
            continue
        day = peak.normalize()
        lon = _num(getattr(row, "lon_w_deg", np.nan))
        if not np.isfinite(lon):
            continue
        candidates = by_day.get(day)
        if candidates is None or candidates.empty:
            continue
        delta = (pd.to_numeric(candidates["noaa_srs_lon_w_deg"], errors="coerce") - lon).abs()
        idx = delta.idxmin()
        if not np.isfinite(delta.loc[idx]) or float(delta.loc[idx]) > 25.0:
            continue
        for col in (
            "noaa_srs_region_number",
            "noaa_srs_lon_w_deg",
            "noaa_ar_area_proxy",
            "noaa_mag_class_score",
            "noaa_delta_spot_flag",
            "noaa_zurich_class_score",
        ):
            out.at[int(row.Index), col] = _num(candidates.loc[idx, col])
        out.at[int(row.Index), "noaa_srs_match_delta_lon"] = float(delta.loc[idx])
    area = pd.to_numeric(out["noaa_ar_area_proxy"], errors="coerce")
    mag = pd.to_numeric(out["noaa_mag_class_score"], errors="coerce")
    delta = pd.to_numeric(out["noaa_delta_spot_flag"], errors="coerce")
    # Explicit proxy columns: real NOAA SRS text, empirical scale, not HMI FITS.
    out["hmi_magnetic_flux_proxy_wb"] = area * 1e14
    out["hmi_mean_shear_proxy_deg"] = (10.0 + 8.0 * mag + 12.0 * delta).clip(0.0, 90.0)
    out["hmi_r_value_proxy"] = area.fillna(0.0) * (1.0 + mag.fillna(0.0)) * (1.0 + 2.0 * delta.fillna(0.0))
    return out


def _flare_history_weight_24h(df: pd.DataFrame) -> pd.Series:
    t = pd.to_datetime(df["event_peaktime"], utc=True, errors="coerce", format="mixed")
    lon = pd.to_numeric(df.get("lon_w_deg"), errors="coerce")
    score = pd.to_numeric(df.get("goes_class_score"), errors="coerce").fillna(0.0)
    work = pd.DataFrame({"idx": np.arange(len(df)), "t": t, "lon": lon, "score": score}).dropna(subset=["t"]).sort_values("t")
    out = pd.Series(np.nan, index=df.index, dtype=float)
    rows = list(work.itertuples(index=False))
    left = 0
    for right, cur in enumerate(rows):
        while left < right and (cur.t - rows[left].t) > pd.Timedelta(hours=24):
            left += 1
        total = 0.0
        for prev in rows[left:right]:
            if np.isfinite(cur.lon) and np.isfinite(prev.lon) and abs(float(cur.lon) - float(prev.lon)) > 25.0:
                continue
            total += max(float(prev.score), 0.0)
        out.iloc[int(cur.idx)] = total
    return out


def _event_wind_features(
    type2: pd.DataFrame,
    type4: pd.DataFrame,
    start_time: pd.Timestamp,
    peak: pd.Timestamp,
    cme_speed_kms: float,
) -> dict[str, float | str]:
    flare_start = pd.Timestamp(start_time).tz_convert("UTC")
    xray_peak = pd.Timestamp(peak).tz_convert("UTC")
    out = {name: 0.0 for name in WIND_FEATURES}
    out["is_wind_noise_or_none"] = 1.0
    out["wind_type2_delay_from_flare_start_min"] = WIND_DELAY_MISSING_VALUE
    out["wind_type2_delay_from_xray_peak_min"] = WIND_DELAY_MISSING_VALUE
    out["wind_match_quality"] = "none"
    if not type2.empty:
        start = pd.to_datetime(type2["start_utc"], utc=True)
        dt_start_min = (start - flare_start).dt.total_seconds() / 60.0
        sub = type2[
            dt_start_min.ge(WIND_TYPE2_MIN_DELAY_MIN)
            & dt_start_min.le(WIND_TYPE2_MAX_DELAY_MIN)
        ].copy()
        if not sub.empty:
            sub_start = pd.to_datetime(sub["start_utc"], utc=True)
            sub_dt_start = (sub_start - flare_start).dt.total_seconds() / 60.0
            sub_dt_peak = (sub_start - xray_peak).dt.total_seconds() / 60.0
            best_idx = sub_dt_start.idxmin()
            best_delay = float(sub_dt_start.loc[best_idx])
            quality_score = 2.0 if best_delay <= WIND_TYPE2_STRONG_MAX_DELAY_MIN else 1.0
            quality = "strong" if quality_score == 2.0 else "weak"
            f0 = pd.to_numeric(sub["freq_start_khz"], errors="coerce").to_numpy(dtype=float)
            f1 = pd.to_numeric(sub["freq_end_khz"], errors="coerce").to_numpy(dtype=float)
            dur = pd.to_numeric(sub["duration_min"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            bandwidth = np.maximum(f0 - f1, 0.0)
            drift = bandwidth / np.maximum(dur, 1.0)
            v_shock = np.maximum(0.0, (1.0 / np.sqrt(np.maximum(f1, 1.0))) - (1.0 / np.sqrt(np.maximum(f0, 1.0)))) / np.maximum(dur, 1.0)
            strength = np.log10(1.0 + np.maximum(f0, 0.0) / np.maximum(f1, 1.0) + np.maximum(dur, 0.0) / 30.0)
            out.update(
                {
                    "wind_type2_present": 1.0,
                    "wind_type2_count": float(len(sub)),
                    "wind_match_quality_score": quality_score,
                    "is_wind_strong": float(quality == "strong"),
                    "is_wind_weak": float(quality == "weak"),
                    "is_wind_noise_or_none": 0.0,
                    "wind_type2_delay_from_flare_start_min": best_delay,
                    "wind_type2_delay_from_xray_peak_min": float(np.nanmin(sub_dt_peak.to_numpy(dtype=float))),
                    "wind_type2_inverse_delay": float(1.0 / (best_delay + 1e-5)) if best_delay > 0.0 else 1.0,
                    "wind_type2_duration_max_min": float(np.nanmax(dur)),
                    "wind_type2_duration_sum_min": float(np.nansum(dur)),
                    "wind_type2_freq_start_max_khz": float(np.nanmax(f0)),
                    "wind_type2_freq_end_min_khz": float(np.nanmin(f1)),
                    "wind_type2_bandwidth_max_khz": float(np.nanmax(bandwidth)),
                    "wind_type2_drift_proxy_max": float(np.nanmax(drift)),
                    "wind_type2_v_shock_proxy": float(np.nanmax(v_shock)),
                    "wind_type2_log_power": float(np.nanmax(strength)),
                    "wind_shock_survival_index": float(np.nanmax(bandwidth / np.maximum(dur, 1e-5))),
                }
            )
            out["wind_match_quality"] = quality
    if not type4.empty:
        start4 = pd.to_datetime(type4["start_utc"], utc=True)
        dt4_min = (start4 - flare_start).dt.total_seconds() / 60.0
        sub4 = type4[dt4_min.ge(0.0) & dt4_min.le(WIND_TYPE4_MAX_DELAY_MIN)].copy()
        if not sub4.empty:
            dur4 = pd.to_numeric(sub4["duration_min"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            out.update(
                {
                    "wind_type4_present": 1.0,
                    "wind_type4_duration_max_min": float(np.nanmax(dur4)),
                    "wind_type4_duration_sum_min": float(np.nansum(dur4)),
                }
            )
    out["wind_radio_combined_score"] = float(
        out["wind_type2_log_power"] + np.log10(1.0 + out["wind_type4_duration_max_min"] / 10.0)
    )
    speed = _num(cme_speed_kms)
    delay = _num(out["wind_type2_delay_from_flare_start_min"])
    out["shock_acceleration_index"] = float(speed / delay) if np.isfinite(speed) and delay > 0.0 else 0.0
    return out


def _read_hmi_cached_headers() -> pd.DataFrame:
    try:
        from astropy.io import fits
    except ImportError:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for path in Path(hmi_sharp_cache_dir).rglob("*.fits*"):
        try:
            with fits.open(path, memmap=False) as hdul:
                hdr = hdul[1].header if len(hdul) > 1 else hdul[0].header
            dt_raw = hdr.get("DATE-OBS") or hdr.get("DATE_OBS") or hdr.get("T_REC") or hdr.get("T_OBS")
            if not dt_raw:
                continue
            dt_s = str(dt_raw).replace("_TAI", "").replace(".", "-", 2).replace("_", "T")
            dt = pd.to_datetime(dt_s, utc=True, errors="coerce")
            if pd.isna(dt):
                continue
            rows.append(
                {
                    "DateTime": dt,
                    "AR_Number": _num(hdr.get("NOAA_AR")),
                    "HARPNUM": _num(hdr.get("HARPNUM")),
                    "hmi_magnetic_flux_wb": _num(hdr.get("USFLUX")) * 1e-8,
                    "hmi_mean_bz": _num(hdr.get("MEANGBZ")),
                    "hmi_mean_bt": _num(hdr.get("MEANGBT")),
                    "hmi_mean_gamma": _num(hdr.get("MEANGAM")),
                    "hmi_mean_shear": _num(hdr.get("MEANSHR")),
                    "hmi_area_hem": _num(hdr.get("AREA_ACR")),
                    "hmi_mean_pot": _num(hdr.get("MEANPOT")),
                    "hmi_tot_pot": _num(hdr.get("TOTPOT")),
                    "hmi_totusjz": _num(hdr.get("TOTUSJZ")),
                    "hmi_absnjzh": _num(hdr.get("ABSNJZH")),
                    "hmi_shrgt45": _num(hdr.get("SHRGT45")),
                    "hmi_r_value": _num(hdr.get("R_VALUE")),
                }
            )
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("DateTime").reset_index(drop=True)


def _nearest_hmi_features(hmi: pd.DataFrame, peak: pd.Timestamp, max_hours: float = 12.0) -> dict[str, float]:
    if hmi.empty or pd.isna(peak):
        return {}
    t = peak.tz_convert("UTC")
    sub = hmi[(hmi["DateTime"] >= t - pd.Timedelta(hours=max_hours)) & (hmi["DateTime"] <= t + pd.Timedelta(hours=1))]
    if sub.empty:
        return {}
    idx = (sub["DateTime"] - t).abs().idxmin()
    row = sub.loc[idx]
    return {col: _num(row.get(col)) for col in HMI_FEATURES if col in row.index}


def enrich_dataset(input_csv: Path = INPUT_CSV, output_csv: Path = OUTPUT_CSV) -> None:
    df = pd.read_csv(input_csv)
    for col in ("event_starttime", "event_peaktime", "event_endtime"):
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce", format="mixed")
    df["candidate_gate_realdata"] = candidate_gate(df).astype(int)

    # Real WIND/WAVES radio catalog features. These are event catalogs, not synthetic
    # radio intensities, so rows without catalog overlap remain zero/present=0.
    cand = df[df["candidate_gate_realdata"].eq(1)].copy()
    type2 = _load_wind_type2_catalog()
    type4 = _load_wind_type4_catalog()
    for col in WIND_FEATURES:
        df[col] = 0.0
    df["wind_match_quality"] = "none"
    for row in cand.itertuples():
        feats = _event_wind_features(
            type2,
            type4,
            pd.Timestamp(row.event_starttime),
            pd.Timestamp(row.event_peaktime),
            _num(getattr(row, "hek_cme_speed_kms", np.nan)),
        )
        for key, value in feats.items():
            df.at[int(row.Index), key] = value

    srs_days = sorted(pd.Timestamp(t).normalize() for t in cand["event_peaktime"].dropna().unique())
    srs = _load_or_fetch_srs_for_days(srs_days)
    df = _attach_srs_features(df, srs)
    df["flare_history_weight_24h"] = _flare_history_weight_24h(df)

    # HMI is expensive; use real cached FITS if present. Optional network loading
    # can be performed with hmi_sharp_cli before this script or by setting cache externally.
    hmi = _read_hmi_cached_headers()
    for col in HMI_FEATURES:
        df[col] = np.nan
    if not hmi.empty:
        for row in cand.itertuples():
            feats = _nearest_hmi_features(hmi, pd.Timestamp(row.event_peaktime))
            for key, value in feats.items():
                df.at[int(row.Index), key] = value

    df["xray_hard_to_soft_ratio"] = np.nan
    if "Flux_Short" in df.columns and "Flux_Long" in df.columns:
        short = pd.to_numeric(df["Flux_Short"], errors="coerce")
        long = pd.to_numeric(df["Flux_Long"], errors="coerce")
        df["xray_hard_to_soft_ratio"] = short / long.replace(0, np.nan)
    for col in SXR_DYNAMIC_FEATURES:
        df[col] = np.nan
    peak_flux = pd.to_numeric(df["goes_class_score"], errors="coerce") * 1e-8
    rise_min = pd.to_numeric(df.get("struminsky_sxr_rise_min"), errors="coerce")
    df["sxr_impulse_index"] = peak_flux / rise_min.replace(0.0, np.nan)
    # Real two-channel derivative is only available when short/long time-series
    # columns have been merged upstream; otherwise keep NaN.

    df["ar_magnetic_helicity"] = df["hmi_absnjzh"]
    df["mag_shear_angle_max"] = df["hmi_mean_shear"]
    df["real_wind_hmi_feature_coverage"] = df[
        [c for c in df.columns if c.startswith("wind_") or c in HMI_FEATURES]
    ].notna().sum(axis=1)
    df.to_csv(output_csv, index=False)

    status_rows = []
    status_rows.append({"item": "rows", "value": len(df)})
    status_rows.append({"item": "candidate_gate_rows", "value": int(df["candidate_gate_realdata"].sum())})
    status_rows.append({"item": "wind_type2_catalog_rows", "value": len(type2)})
    status_rows.append({"item": "wind_type4_catalog_rows", "value": len(type4)})
    status_rows.append({"item": "rows_with_wind_type2", "value": int(df["wind_type2_present"].sum())})
    status_rows.append({"item": "rows_with_wind_type4", "value": int(df["wind_type4_present"].sum())})
    status_rows.append({"item": "wind_match_strong_rows", "value": int(df["wind_match_quality"].eq("strong").sum())})
    status_rows.append({"item": "wind_match_weak_rows", "value": int(df["wind_match_quality"].eq("weak").sum())})
    status_rows.append({"item": "wind_type2_min_delay_min", "value": WIND_TYPE2_MIN_DELAY_MIN})
    status_rows.append({"item": "wind_type2_strong_max_delay_min", "value": WIND_TYPE2_STRONG_MAX_DELAY_MIN})
    status_rows.append({"item": "wind_type2_max_delay_min", "value": WIND_TYPE2_MAX_DELAY_MIN})
    status_rows.append({"item": "srs_catalog_rows", "value": len(srs)})
    status_rows.append({"item": "rows_with_srs_proxy", "value": int(df["noaa_ar_area_proxy"].notna().sum())})
    status_rows.append({"item": "hmi_cached_rows", "value": len(hmi)})
    status_rows.append({"item": "rows_with_hmi", "value": int(df[list(HMI_FEATURES)].notna().any(axis=1).sum())})
    pd.DataFrame(status_rows).to_csv(STATUS_CSV, index=False)
    print(f"Saved real-data enrichment: {output_csv}")
    print(pd.DataFrame(status_rows).to_string(index=False))


if __name__ == "__main__":
    enrich_dataset()
