from __future__ import annotations

import hashlib
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
STRUMINSKY_DIR = ROOT / "struminsky"
CLI_DIR = STRUMINSKY_DIR / "cli_scripts"
for path in (ROOT, STRUMINSKY_DIR, CLI_DIR, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data_paths import goes_cache_dir, hmi_sharp_cache_dir  # noqa: E402
from goes_sxr_cli import fetch_goes_xrs_records  # noqa: E402
from goes_proton_ncei_cli import fetch_goes_proton_records  # noqa: E402
from rstn_sagamore_cli import aggregate_minute as aggregate_rstn_minute  # noqa: E402
from rstn_sagamore_cli import fetch_network_day as fetch_rstn_day  # noqa: E402


INPUT_CSV = HERE / "sep_probability_dataset_cycles23_24_25_c1_real_wind_hmi.csv"
PREDICTIONS_CSV = HERE / "catboost_real_wind_hmi_gate" / "catboost_real_wind_hmi_gate_predictions.csv"
OUTPUT_CSV = HERE / "sep_probability_dataset_cycles23_24_25_c1_real_wind_srs_xrs_hmi.csv"
TARGET_LIST_CSV = HERE / "targeted_xrs_hmi_request_list.csv"
STATUS_CSV = HERE / "targeted_xrs_hmi_enrichment_status.csv"
TARGETED_XRS_CACHE = ROOT / "data" / "targeted_goes_xrs_cache"
TARGETED_PROTON_CACHE = ROOT / "data" / "targeted_goes_proton_cache"
TARGETED_RSTN_CACHE = ROOT / "data" / "targeted_rstn_cache"
TARGETED_CME_CACHE = ROOT / "data" / "targeted_hek_cme_cache"
CDAW_CME_ALL_TXT = ROOT / "data" / "CME_all.txt"
TARGETED_XRS_MIN_YEAR = int(os.environ.get("SEP_TARGETED_XRS_MIN_YEAR", "2017"))
TARGETED_XRS_OFFLINE = os.environ.get("SEP_TARGETED_XRS_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"}
TARGETED_PROTON_MIN_YEAR = int(os.environ.get("SEP_TARGETED_PROTON_MIN_YEAR", "2011"))
TARGETED_PROTON_OFFLINE = os.environ.get("SEP_TARGETED_PROTON_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"}
TARGETED_RSTN_MIN_YEAR = int(os.environ.get("SEP_TARGETED_RSTN_MIN_YEAR", "2000"))
TARGETED_RSTN_OFFLINE = os.environ.get("SEP_TARGETED_RSTN_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"}
TARGETED_CME_OFFLINE = os.environ.get("SEP_TARGETED_CME_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"}
TARGETED_MAX_NEW_DAYS = int(os.environ.get("SEP_TARGETED_MAX_NEW_DAYS", "0"))
TARGETED_CONTEXT_DAYS = max(0, int(os.environ.get("SEP_TARGETED_CONTEXT_DAYS", "0")))

XRS_FEATURES = (
    "xray_hard_to_soft_ratio",
    "xs_ratio_10m",
    "goes_ratio_derivative",
    "sxr_peak_flux_derivative_max",
    "thermal_energy_growth_rate",
    "hard_xray_impulse_duration",
    "xrs_short_peak_30m",
    "xrs_long_integral_10m",
    "xrs_short_integral_10m",
)
PROTON_FEATURES = (
    "pre_event_proton_background_10mev",
    "pre_event_proton_background_available",
)
CDAW_CME_FEATURES = (
    "cdaw_cme_angular_width_deg",
    "cdaw_cme_linear_speed_kms",
    "cme_initial_speed_2nd_order",
    "cme_2nd_order_acceleration",
    "cme_speed_20rsun",
    "cme_final_speed_2nd_order",
    "cdaw_cme_2nd_order_available",
)
RSTN_FIXED_FREQ_FEATURES = (
    "radio_fixed_peak_245",
    "radio_fixed_integral_245",
    "radio_fixed_available_245",
    "radio_fixed_peak_410",
    "radio_fixed_integral_410",
    "radio_fixed_available_410",
    "radio_fixed_peak_606",
    "radio_fixed_integral_606",
    "radio_fixed_available_606",
    "radio_fixed_peak_1415",
    "radio_fixed_integral_1415",
    "radio_fixed_available_1415",
    "radio_fixed_peak_2695",
    "radio_fixed_integral_2695",
    "radio_fixed_available_2695",
    "radio_fixed_peak_4995",
    "radio_fixed_integral_4995",
    "radio_fixed_available_4995",
    "radio_fixed_peak_8800",
    "radio_fixed_integral_8800",
    "radio_fixed_available_8800",
    "radio_fixed_peak_15400",
    "radio_fixed_integral_15400",
    "radio_fixed_available_15400",
    "radio_fixed_integral_available",
)
HMI_TARGET_FEATURES = (
    "hmi_totusjh",
    "hmi_savncpp",
    "hmi_meanshr_targeted",
)


def _num(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _catalog_num(value: object) -> float:
    text = str(value).strip().replace("*", "")
    if text in {"", "----", "------", "-------"}:
        return float("nan")
    return _num(text)


def _safe_id(value: object) -> str:
    raw = str(value or "").encode("utf-8", errors="ignore")
    return hashlib.sha1(raw).hexdigest()[:16]


def _cache_path(event_id: str) -> Path:
    return TARGETED_XRS_CACHE / f"{_safe_id(event_id)}.csv"


def _day_cache_path(day: pd.Timestamp) -> Path:
    return TARGETED_XRS_CACHE / "days" / f"{day:%Y%m%d}.csv"


def _proton_day_cache_path(day: pd.Timestamp) -> Path:
    return TARGETED_PROTON_CACHE / "days" / f"{day:%Y%m%d}.csv"


def _rstn_day_cache_path(day: pd.Timestamp) -> Path:
    return TARGETED_RSTN_CACHE / "days" / f"{day:%Y%m%d}.csv"


def _cme_day_cache_path(day: pd.Timestamp) -> Path:
    return TARGETED_CME_CACHE / "days" / f"{day:%Y%m%d}.csv"


_NEW_DAY_FETCHES: dict[str, int] = {"xrs": 0, "proton": 0, "rstn": 0, "cme": 0}


def _may_fetch_new_day(source: str, path: Path, missing: Path) -> bool:
    if path.exists() or missing.exists():
        return True
    count = _NEW_DAY_FETCHES.get(source, 0)
    if TARGETED_MAX_NEW_DAYS > 0 and count >= TARGETED_MAX_NEW_DAYS:
        return False
    _NEW_DAY_FETCHES[source] = count + 1
    return True


def _event_context_days(row: pd.Series) -> list[pd.Timestamp]:
    start = pd.Timestamp(row["event_starttime"])
    if start.tzinfo is not None:
        start = start.tz_convert("UTC").tz_localize(None)
    center = start.normalize()
    return [center + pd.Timedelta(days=delta) for delta in range(-TARGETED_CONTEXT_DAYS, TARGETED_CONTEXT_DAYS + 1)]


def _concat_context_days(row: pd.Series, loader: Any) -> pd.DataFrame:
    parts = [loader(day) for day in _event_context_days(row)]
    parts = [part for part in parts if part is not None and not part.empty]
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    if "DateTime" in out.columns:
        out["DateTime"] = pd.to_datetime(out["DateTime"], utc=True, errors="coerce").dt.tz_convert(None)
        out = out.dropna(subset=["DateTime"])
        out = out.drop_duplicates("DateTime").sort_values("DateTime")
    return out.reset_index(drop=True)


def build_target_list(df: pd.DataFrame) -> pd.DataFrame:
    targets = df[df["is_sep_event"].astype(int).eq(1)].copy()
    if "candidate_gate_realdata" in df.columns:
        targets = pd.concat([targets, df[df["candidate_gate_realdata"].eq(1)]], ignore_index=True)
    if PREDICTIONS_CSV.exists():
        pred = pd.read_csv(PREDICTIONS_CSV)
        prob_col = "catboost_candidate_probability"
        if prob_col in pred.columns:
            risky_ids = set(pred[pd.to_numeric(pred[prob_col], errors="coerce").fillna(0.0).ge(0.30)]["hek_event_id"].astype(str))
            pos_ids = set(pred[pred["is_sep_event"].astype(int).eq(1)]["hek_event_id"].astype(str))
            extra = df[df["hek_event_id"].astype(str).isin(risky_ids | pos_ids) & df["candidate_gate_realdata"].eq(1)]
            targets = pd.concat([targets, extra], ignore_index=True)
    cols = ["hek_event_id", "event_starttime", "event_peaktime", "event_endtime", "flare_class", "is_sep_event", "candidate_gate_realdata"]
    targets = targets.drop_duplicates("hek_event_id").sort_values("event_starttime")
    return targets[[c for c in cols if c in targets.columns]].reset_index(drop=True)


def _load_cached_xrs(event_id: str) -> pd.DataFrame | None:
    path = _cache_path(event_id)
    if path.exists():
        return pd.read_csv(path, parse_dates=["DateTime"])
    missing = path.with_suffix(".missing")
    if missing.exists():
        return pd.DataFrame()
    return None


def _load_or_fetch_day_xrs(day: pd.Timestamp) -> pd.DataFrame:
    day = pd.Timestamp(day).tz_localize(None) if pd.Timestamp(day).tzinfo is not None else pd.Timestamp(day)
    path = _day_cache_path(day)
    missing = path.with_suffix(".missing")
    if path.exists():
        return pd.read_csv(path, parse_dates=["DateTime"])
    if missing.exists():
        return pd.DataFrame()
    if TARGETED_XRS_OFFLINE:
        return pd.DataFrame()
    TARGETED_XRS_CACHE.mkdir(parents=True, exist_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    if int(day.year) < TARGETED_XRS_MIN_YEAR:
        missing.write_text(
            f"pre_min_year:{TARGETED_XRS_MIN_YEAR}",
            encoding="utf-8",
        )
        return pd.DataFrame()
    if not _may_fetch_new_day("xrs", path, missing):
        return pd.DataFrame()
    day_start = datetime(day.year, day.month, day.day)
    try:
        records = fetch_goes_xrs_records(day_start, day_start, source="auto", satellite=None, cache=goes_cache_dir)
    except Exception as exc:
        missing.write_text(f"error: {exc}", encoding="utf-8")
        return pd.DataFrame()
    if not records:
        missing.write_text("empty", encoding="utf-8")
        return pd.DataFrame()
    df = pd.DataFrame(records)
    if df.empty or "DateTime" not in df.columns:
        missing.write_text("no_datetime", encoding="utf-8")
        return pd.DataFrame()
    df["DateTime"] = pd.to_datetime(df["DateTime"], utc=True, errors="coerce").dt.tz_convert(None)
    keep = [c for c in ["DateTime", "Flux_Short", "Flux_Long", "SXR_Temp_MK", "SXR_EM", "Quality_A", "Quality_B"] if c in df.columns]
    df = df[keep].dropna(subset=["DateTime"]).sort_values("DateTime")
    df.to_csv(path, index=False)
    return df


def _fetch_event_xrs(row: pd.Series) -> pd.DataFrame:
    return _concat_context_days(row, _load_or_fetch_day_xrs)


def _load_or_fetch_day_protons(day: pd.Timestamp) -> pd.DataFrame:
    day = pd.Timestamp(day).tz_localize(None) if pd.Timestamp(day).tzinfo is not None else pd.Timestamp(day)
    path = _proton_day_cache_path(day)
    missing = path.with_suffix(".missing")
    if path.exists():
        return pd.read_csv(path, parse_dates=["DateTime"])
    if missing.exists():
        return pd.DataFrame()
    if TARGETED_PROTON_OFFLINE:
        return pd.DataFrame()
    path.parent.mkdir(parents=True, exist_ok=True)
    if int(day.year) < TARGETED_PROTON_MIN_YEAR:
        missing.write_text(f"pre_min_year:{TARGETED_PROTON_MIN_YEAR}", encoding="utf-8")
        return pd.DataFrame()
    if not _may_fetch_new_day("proton", path, missing):
        return pd.DataFrame()
    day_start = datetime(day.year, day.month, day.day)
    satellite = _goes_proton_satellite_for_day(day_start)
    try:
        records = fetch_goes_proton_records(day_start, day_start, satellite=satellite, source="auto", raise_errors=False)
    except Exception as exc:
        missing.write_text(f"error: {exc}", encoding="utf-8")
        return pd.DataFrame()
    if not records:
        missing.write_text("empty", encoding="utf-8")
        return pd.DataFrame()
    df = pd.DataFrame(records)
    if df.empty or "DateTime" not in df.columns:
        missing.write_text("no_datetime", encoding="utf-8")
        return pd.DataFrame()
    df["DateTime"] = pd.to_datetime(df["DateTime"], utc=True, errors="coerce").dt.tz_convert(None)
    df = df.dropna(subset=["DateTime"]).sort_values("DateTime")
    df.to_csv(path, index=False)
    return df


def _goes_proton_satellite_for_day(day: datetime) -> int:
    """Pick an archive-supported GOES proton satellite for the event era."""
    if day.year >= 2023:
        return 18
    if day.year >= 2017:
        return 16
    if day.year >= 2011:
        return 15
    if day.year >= 2000:
        return 11
    if day.year >= 1998:
        return 10
    return 8


def _load_or_fetch_day_rstn(day: pd.Timestamp) -> pd.DataFrame:
    day = pd.Timestamp(day).tz_localize(None) if pd.Timestamp(day).tzinfo is not None else pd.Timestamp(day)
    path = _rstn_day_cache_path(day)
    missing = path.with_suffix(".missing")
    if path.exists():
        return pd.read_csv(path, parse_dates=["DateTime"])
    if missing.exists():
        return pd.DataFrame()
    if TARGETED_RSTN_OFFLINE:
        return pd.DataFrame()
    path.parent.mkdir(parents=True, exist_ok=True)
    if int(day.year) < TARGETED_RSTN_MIN_YEAR:
        missing.write_text(f"pre_min_year:{TARGETED_RSTN_MIN_YEAR}", encoding="utf-8")
        return pd.DataFrame()
    if not _may_fetch_new_day("rstn", path, missing):
        return pd.DataFrame()
    try:
        rows = fetch_rstn_day(datetime(day.year, day.month, day.day), None)
        df = aggregate_rstn_minute(rows)
    except Exception as exc:
        missing.write_text(f"error: {exc}", encoding="utf-8")
        return pd.DataFrame()
    if df.empty or "DateTime" not in df.columns:
        missing.write_text("empty", encoding="utf-8")
        return pd.DataFrame()
    df["DateTime"] = pd.to_datetime(df["DateTime"], utc=True, errors="coerce").dt.tz_convert(None)
    df = df.dropna(subset=["DateTime"]).sort_values("DateTime")
    df.to_csv(path, index=False)
    return df


def _load_or_fetch_day_cmes(day: pd.Timestamp) -> pd.DataFrame:
    day = pd.Timestamp(day).tz_localize(None) if pd.Timestamp(day).tzinfo is not None else pd.Timestamp(day)
    path = _cme_day_cache_path(day)
    missing = path.with_suffix(".missing")
    if path.exists():
        df = pd.read_csv(path)
        for col in ("event_starttime", "event_peaktime", "event_endtime"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=True, errors="coerce").dt.tz_convert(None)
        return df
    if missing.exists():
        return pd.DataFrame()
    if TARGETED_CME_OFFLINE:
        return pd.DataFrame()
    if not _may_fetch_new_day("cme", path, missing):
        return pd.DataFrame()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import ssl
        from sunpy.net import attrs as a
        from sunpy.net import hek

        ssl._create_default_https_context = ssl._create_unverified_context
        client = hek.HEKClient()
        start = datetime(day.year, day.month, day.day)
        end = start + pd.Timedelta(days=1)
        events = client.search(a.Time(start, end), hek.attrs.EventType("CE"))
    except Exception as exc:
        missing.write_text(f"error: {exc}", encoding="utf-8")
        return pd.DataFrame()
    rows = [dict(ev) for ev in events]
    if not rows:
        missing.write_text("empty", encoding="utf-8")
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for col in ("event_starttime", "event_peaktime", "event_endtime"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce").dt.tz_convert(None)
    df.to_csv(path, index=False)
    return df


def _fetch_event_cmes(row: pd.Series) -> pd.DataFrame:
    parts = [_concat_context_days(row, _load_or_fetch_day_cmes), _cdaw_catalog_cmes_for_event(row)]
    parts = [part for part in parts if part is not None and not part.empty]
    return pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()


_CDAW_CME_ALL: pd.DataFrame | None = None


def _read_cdaw_cme_all() -> pd.DataFrame:
    global _CDAW_CME_ALL
    if _CDAW_CME_ALL is not None:
        return _CDAW_CME_ALL
    rows: list[dict[str, object]] = []
    if not CDAW_CME_ALL_TXT.exists():
        _CDAW_CME_ALL = pd.DataFrame()
        return _CDAW_CME_ALL
    for raw in CDAW_CME_ALL_TXT.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("=") or line.startswith("Date") or line.startswith("PA"):
            continue
        parts = line.split(maxsplit=12)
        if len(parts) < 12 or "/" not in parts[0] or ":" not in parts[1]:
            continue
        event_time = pd.to_datetime(f"{parts[0]} {parts[1]}", utc=True, errors="coerce")
        if pd.isna(event_time):
            continue
        rows.append(
            {
                "event_starttime": event_time.tz_convert(None),
                "event_peaktime": event_time.tz_convert(None),
                "cdaw_cme_source": "SOHO/LASCO CDAW",
                "cdaw_cme_pa_deg": _catalog_num(parts[2]),
                "cdaw_cme_angular_width_deg": _catalog_num(parts[3]),
                "cdaw_cme_linear_speed_kms": _catalog_num(parts[4]),
                "cme_initial_speed_2nd_order": _catalog_num(parts[5]),
                "cme_final_speed_2nd_order": _catalog_num(parts[6]),
                "cme_speed_20rsun": _catalog_num(parts[7]),
                "cme_2nd_order_acceleration": _catalog_num(parts[8]),
                "cdaw_cme_mpa_deg": _catalog_num(parts[11]),
            }
        )
    _CDAW_CME_ALL = pd.DataFrame(rows).sort_values("event_starttime").reset_index(drop=True) if rows else pd.DataFrame()
    return _CDAW_CME_ALL


def _cdaw_catalog_cmes_for_event(row: pd.Series) -> pd.DataFrame:
    catalog = _read_cdaw_cme_all()
    if catalog.empty:
        return catalog
    t0 = pd.Timestamp(row["event_starttime"])
    if t0.tzinfo is not None:
        t0 = t0.tz_convert("UTC").tz_localize(None)
    times = pd.to_datetime(catalog["event_starttime"], utc=True, errors="coerce").dt.tz_convert(None)
    return catalog[(times >= t0 - pd.Timedelta(hours=2)) & (times <= t0 + pd.Timedelta(hours=12))].copy()


def _fetch_event_protons(row: pd.Series) -> pd.DataFrame:
    return _concat_context_days(row, _load_or_fetch_day_protons)


def _fetch_event_rstn(row: pd.Series) -> pd.DataFrame:
    return _concat_context_days(row, _load_or_fetch_day_rstn)


def _parse_lower_energy_mev(col: str) -> float:
    import re

    match = re.search(r"_(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)MeV", col)
    if match:
        return float(match.group(1))
    return float("nan")


def _proton_10mev_series(protons: pd.DataFrame) -> pd.Series:
    if protons.empty:
        return pd.Series(dtype=float)
    direct = [
        c
        for c in protons.columns
        if c != "DateTime" and ("10" in c or ">10" in c or ">=10" in c) and ("proton" in c.lower() or "int" in c.lower())
    ]
    if direct:
        return pd.to_numeric(protons[direct[0]], errors="coerce")
    energetic = [c for c in protons.columns if np.isfinite(_parse_lower_energy_mev(str(c))) and _parse_lower_energy_mev(str(c)) >= 10.0]
    if energetic:
        return protons[energetic].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
    sem = [c for c in ("SEM_P4", "SEM_P5", "SEM_P6", "SEM_P7") if c in protons.columns]
    if sem:
        return protons[sem].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
    return pd.Series(dtype=float)


def _proton_features_for_event(protons: pd.DataFrame, row: pd.Series) -> dict[str, float]:
    out = {
        "pre_event_proton_background_10mev": 0.1,
        "pre_event_proton_background_available": 0.0,
    }
    if protons.empty or "DateTime" not in protons.columns:
        return out
    t0 = pd.Timestamp(row["event_starttime"])
    if t0.tzinfo is not None:
        t0 = t0.tz_convert("UTC").tz_localize(None)
    t = pd.to_datetime(protons["DateTime"], utc=True, errors="coerce").dt.tz_convert(None)
    series = _proton_10mev_series(protons)
    if series.empty:
        return out
    sub = series[(t >= t0 - pd.Timedelta(minutes=5)) & (t < t0)]
    value = _num(sub.median())
    if np.isfinite(value):
        out["pre_event_proton_background_10mev"] = value
        out["pre_event_proton_background_available"] = 1.0
    return out


def _rstn_features_for_event(rstn: pd.DataFrame, row: pd.Series) -> dict[str, float]:
    freq_cols = (
        ("245", "Radio_245MHz"),
        ("410", "Radio_410MHz"),
        ("606", "Radio_610MHz"),
        ("1415", "Radio_1415MHz"),
        ("2695", "Radio_2695MHz"),
        ("4995", "Radio_4995MHz"),
        ("8800", "Radio_8800MHz"),
        ("15400", "Radio_15400MHz"),
    )
    out = {"radio_fixed_integral_available": 0.0}
    for mhz, _ in freq_cols:
        out[f"radio_fixed_peak_{mhz}"] = 0.0
        out[f"radio_fixed_integral_{mhz}"] = 0.0
        out[f"radio_fixed_available_{mhz}"] = 0.0
    if rstn.empty or "DateTime" not in rstn.columns:
        return out
    t0 = pd.Timestamp(row["event_starttime"])
    if t0.tzinfo is not None:
        t0 = t0.tz_convert("UTC").tz_localize(None)
    t = pd.to_datetime(rstn["DateTime"], utc=True, errors="coerce").dt.tz_convert(None)
    sub = rstn[(t >= t0) & (t <= t0 + pd.Timedelta(minutes=10))]
    if sub.empty:
        return out
    for mhz, col in freq_cols:
        if col in sub.columns:
            series = pd.to_numeric(sub[col], errors="coerce")
            integral = _num(series.sum())
            peak = _num(series.max())
            if np.isfinite(integral) or np.isfinite(peak):
                out[f"radio_fixed_integral_{mhz}"] = integral if np.isfinite(integral) else 0.0
                out[f"radio_fixed_peak_{mhz}"] = peak if np.isfinite(peak) else 0.0
                out[f"radio_fixed_available_{mhz}"] = 1.0
                out["radio_fixed_integral_available"] = 1.0
    return out


def _first_numeric_from_columns(row: pd.Series, candidates: tuple[str, ...]) -> float:
    lower = {str(c).lower(): c for c in row.index}
    for name in candidates:
        col = lower.get(name.lower())
        if col is None:
            continue
        value = _num(row[col])
        if np.isfinite(value):
            return value
    return float("nan")


def _first_numeric_matching(row: pd.Series, required: tuple[str, ...], forbidden: tuple[str, ...] = ()) -> float:
    for col in row.index:
        name = str(col).lower()
        if all(token in name for token in required) and not any(token in name for token in forbidden):
            value = _num(row[col])
            if np.isfinite(value):
                return value
    return float("nan")


def _cdaw_cme_features_for_event(cmes: pd.DataFrame, row: pd.Series) -> dict[str, float]:
    out = {
        "cdaw_cme_angular_width_deg": 0.0,
        "cdaw_cme_linear_speed_kms": 0.0,
        "cme_initial_speed_2nd_order": 0.0,
        "cme_2nd_order_acceleration": 0.0,
        "cme_speed_20rsun": 0.0,
        "cme_final_speed_2nd_order": 0.0,
        "cdaw_cme_2nd_order_available": 0.0,
    }
    if cmes.empty:
        return out
    t0 = pd.Timestamp(row["event_starttime"])
    if t0.tzinfo is not None:
        t0 = t0.tz_convert("UTC").tz_localize(None)
    time_col = next((c for c in ("event_peaktime", "event_starttime", "event_endtime") if c in cmes.columns), None)
    sub = cmes
    if time_col is not None:
        times = pd.to_datetime(cmes[time_col], utc=True, errors="coerce").dt.tz_convert(None)
        sub = cmes[(times >= t0 - pd.Timedelta(hours=2)) & (times <= t0 + pd.Timedelta(hours=12))].copy()
        if sub.empty:
            sub = cmes.copy()
        times = pd.to_datetime(sub[time_col], utc=True, errors="coerce").dt.tz_convert(None)
        if times.notna().any():
            idx = (times - t0).abs().idxmin()
            best = sub.loc[idx]
        else:
            best = sub.iloc[0]
    else:
        best = cmes.iloc[0]
    angular_width = _first_numeric_from_columns(
        best,
        (
            "cdaw_cme_angular_width_deg",
            "cme_angularwidth",
            "cme_angular_width",
            "hek_cme_width_deg",
            "width",
        ),
    )
    linear_speed = _first_numeric_from_columns(
        best,
        (
            "cdaw_cme_linear_speed_kms",
            "cme_radiallinvel",
            "cme_linear_speed",
            "hek_cme_speed_kms",
            "linear_speed",
        ),
    )
    initial_speed = _first_numeric_from_columns(
        best,
        (
            "cme_initial_speed_2nd_order",
            "second_order_speed_initial_kms",
            "cme_initial_speed",
            "initial_speed",
        ),
    )
    accel = _first_numeric_from_columns(
        best,
        (
            "cme_2nd_order_acceleration",
            "cme_acceleration",
            "cme_accel",
            "acceleration",
            "accel",
        ),
    )
    speed20 = _first_numeric_from_columns(
        best,
        (
            "cme_speed_20rsun",
            "cme_speed_20r",
            "speed_20rsun",
            "speed_20r",
            "v_20rsun",
            "v20",
        ),
    )
    if not np.isfinite(speed20):
        speed20 = _first_numeric_matching(best, ("speed", "20"), forbidden=("width",))
    final_speed = _first_numeric_from_columns(
        best,
        (
            "cme_final_speed_2nd_order",
            "second_order_speed_final_kms",
            "cme_final_speed",
            "cme_radial_velocity",
            "cme_linear_velocity",
            "cme_velocity",
            "velocity",
        ),
    )
    if np.isfinite(angular_width):
        out["cdaw_cme_angular_width_deg"] = angular_width
    if np.isfinite(linear_speed):
        out["cdaw_cme_linear_speed_kms"] = linear_speed
    if np.isfinite(initial_speed):
        out["cme_initial_speed_2nd_order"] = initial_speed
        out["cdaw_cme_2nd_order_available"] = 1.0
    if np.isfinite(accel):
        out["cme_2nd_order_acceleration"] = accel
        out["cdaw_cme_2nd_order_available"] = 1.0
    if np.isfinite(speed20):
        out["cme_speed_20rsun"] = speed20
        out["cdaw_cme_2nd_order_available"] = 1.0
    if np.isfinite(final_speed):
        out["cme_final_speed_2nd_order"] = final_speed
        out["cdaw_cme_2nd_order_available"] = 1.0
    return out


def _xrs_features_for_event(xrs: pd.DataFrame, row: pd.Series) -> dict[str, float]:
    out = {name: np.nan for name in XRS_FEATURES}
    if xrs.empty or "Flux_Short" not in xrs.columns or "Flux_Long" not in xrs.columns:
        return out
    t0 = pd.Timestamp(row["event_starttime"])
    if t0.tzinfo is not None:
        t0 = t0.tz_convert("UTC").tz_localize(None)
    lo = t0 - pd.Timedelta(hours=1)
    hi = t0 + pd.Timedelta(minutes=30)
    seg = xrs[(xrs["DateTime"] >= lo) & (xrs["DateTime"] <= hi)].copy()
    if seg.empty:
        return out
    short = pd.to_numeric(seg["Flux_Short"], errors="coerce")
    long = pd.to_numeric(seg["Flux_Long"], errors="coerce")
    tmin = (seg["DateTime"] - t0).dt.total_seconds() / 60.0
    ratio = short / long.replace(0.0, np.nan)
    near10 = (tmin - 10.0).abs()
    if near10.notna().any():
        idx10 = near10.idxmin()
        out["xs_ratio_10m"] = _num(ratio.loc[idx10])
        out["xray_hard_to_soft_ratio"] = _num(ratio.loc[idx10])
    first10 = seg[(tmin >= 0.0) & (tmin <= 10.0)].copy()
    if not first10.empty:
        s10 = pd.to_numeric(first10["Flux_Short"], errors="coerce")
        l10 = pd.to_numeric(first10["Flux_Long"], errors="coerce")
        r10 = s10 / l10.replace(0.0, np.nan)
        if r10.notna().sum() >= 2:
            t10 = (first10["DateTime"] - t0).dt.total_seconds() / 60.0
            valid = r10.notna() & t10.notna()
            if valid.sum() >= 2:
                out["goes_ratio_derivative"] = _num(np.polyfit(t10[valid].to_numpy(dtype=float), r10[valid].to_numpy(dtype=float), 1)[0])
                ratio_diff = r10[valid].diff()
                time_diff = t10[valid].diff()
                out["sxr_peak_flux_derivative_max"] = _num((ratio_diff / time_diff.replace(0.0, np.nan)).max())
        bg_long = _num(long[tmin < 0.0].median())
        bg_short = _num(short[tmin < 0.0].median())
        if not np.isfinite(bg_long):
            bg_long = _num(l10.quantile(0.1))
        if not np.isfinite(bg_short):
            bg_short = _num(s10.quantile(0.1))
        out["xrs_long_integral_10m"] = _num(np.nansum(np.maximum(l10.to_numpy(dtype=float) - bg_long, 0.0)))
        out["xrs_short_integral_10m"] = _num(np.nansum(np.maximum(s10.to_numpy(dtype=float) - bg_short, 0.0)))
        out["thermal_energy_growth_rate"] = out["xrs_long_integral_10m"] / 10.0
        thresh = bg_short + max(1e-8, 3.0 * np.nanstd(short[tmin < 0.0].to_numpy(dtype=float)))
        out["hard_xray_impulse_duration"] = float(np.nansum(s10.to_numpy(dtype=float) > thresh))
    event_window = seg[(tmin >= 0.0) & (tmin <= 30.0)]
    if not event_window.empty:
        short_event = pd.to_numeric(event_window["Flux_Short"], errors="coerce")
        long_event = pd.to_numeric(event_window["Flux_Long"], errors="coerce")
        out["xrs_short_peak_30m"] = _num(short_event.max())
        tev = (event_window["DateTime"] - t0).dt.total_seconds() / 60.0
        valid_short = short_event.notna() & tev.notna()
        if valid_short.sum() >= 2:
            short_slope = short_event[valid_short].diff() / tev[valid_short].diff().replace(0.0, np.nan)
            out["sxr_peak_flux_derivative_max"] = np.nanmax(
                [
                    _num(out["sxr_peak_flux_derivative_max"]),
                    _num(short_slope.max()),
                ]
            )
        ratio_event = short_event / long_event.replace(0.0, np.nan)
        valid_ratio = ratio_event.notna() & tev.notna()
        if valid_ratio.sum() >= 2 and not np.isfinite(_num(out["goes_ratio_derivative"])):
            out["goes_ratio_derivative"] = _num(
                np.polyfit(tev[valid_ratio].to_numpy(dtype=float), ratio_event[valid_ratio].to_numpy(dtype=float), 1)[0]
            )
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
            dt = pd.to_datetime(str(dt_raw).replace("_TAI", "").replace(".", "-", 2).replace("_", "T"), utc=True, errors="coerce")
            if pd.isna(dt):
                continue
            rows.append(
                {
                    "DateTime": dt,
                    "hmi_totusjh": _num(hdr.get("TOTUSJH")),
                    "hmi_savncpp": _num(hdr.get("SAVNCPP")),
                    "hmi_meanshr_targeted": _num(hdr.get("MEANSHR")),
                }
            )
        except Exception:
            continue
    return pd.DataFrame(rows).sort_values("DateTime") if rows else pd.DataFrame()


def _hmi_features_for_event(hmi: pd.DataFrame, row: pd.Series) -> dict[str, float]:
    out = {name: np.nan for name in HMI_TARGET_FEATURES}
    if hmi.empty:
        return out
    t0 = pd.Timestamp(row["event_starttime"])
    if t0.tzinfo is None:
        t0 = t0.tz_localize("UTC")
    else:
        t0 = t0.tz_convert("UTC")
    sub = hmi[(hmi["DateTime"] >= t0 - pd.Timedelta(hours=1)) & (hmi["DateTime"] <= t0 + pd.Timedelta(minutes=30))]
    if sub.empty:
        return out
    idx = (sub["DateTime"] - t0).abs().idxmin()
    return {key: _num(sub.loc[idx, key]) for key in HMI_TARGET_FEATURES if key in sub.columns}


def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    for col in ("event_starttime", "event_peaktime", "event_endtime"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce", format="mixed")
    targets = build_target_list(df)
    targets.to_csv(TARGET_LIST_CSV, index=False)
    all_target_features = XRS_FEATURES + PROTON_FEATURES + CDAW_CME_FEATURES + RSTN_FIXED_FREQ_FEATURES + HMI_TARGET_FEATURES
    for col in all_target_features:
        if col not in df.columns:
            df[col] = np.nan
    hmi = _read_hmi_cached_headers()
    xrs_ok = 0
    proton_ok = 0
    cdaw_cme_ok = 0
    rstn_ok = 0
    hmi_ok = 0
    for target in targets.itertuples():
        row = pd.Series(target._asdict())
        idx = df.index[df["hek_event_id"].astype(str).eq(str(row["hek_event_id"]))]
        if len(idx) == 0:
            continue
        idx0 = int(idx[0])
        xrs = _fetch_event_xrs(row)
        feats = _xrs_features_for_event(xrs, row)
        if any(np.isfinite(v) for v in feats.values()):
            xrs_ok += 1
        for key, value in feats.items():
            df.at[idx0, key] = value
        pfeats = _proton_features_for_event(_fetch_event_protons(row), row)
        if _num(pfeats.get("pre_event_proton_background_available")) > 0.0:
            proton_ok += 1
        for key, value in pfeats.items():
            df.at[idx0, key] = value
        cfeats = _cdaw_cme_features_for_event(_fetch_event_cmes(row), row)
        if _num(cfeats.get("cdaw_cme_2nd_order_available")) > 0.0:
            cdaw_cme_ok += 1
        for key, value in cfeats.items():
            df.at[idx0, key] = value
        rfeats = _rstn_features_for_event(_fetch_event_rstn(row), row)
        if _num(rfeats.get("radio_fixed_integral_available")) > 0.0:
            rstn_ok += 1
        for key, value in rfeats.items():
            df.at[idx0, key] = value
        hfeats = _hmi_features_for_event(hmi, row)
        if any(np.isfinite(v) for v in hfeats.values()):
            hmi_ok += 1
        for key, value in hfeats.items():
            df.at[idx0, key] = value
    neutral_defaults = {
        "pre_event_proton_background_10mev": 0.1,
        "pre_event_proton_background_available": 0.0,
        "cdaw_cme_angular_width_deg": 0.0,
        "cdaw_cme_linear_speed_kms": 0.0,
        "cme_initial_speed_2nd_order": 0.0,
        "cme_2nd_order_acceleration": 0.0,
        "cme_speed_20rsun": 0.0,
        "cme_final_speed_2nd_order": 0.0,
        "cdaw_cme_2nd_order_available": 0.0,
        "radio_fixed_integral_available": 0.0,
    }
    for mhz in ("245", "410", "606", "1415", "2695", "4995", "8800", "15400"):
        neutral_defaults[f"radio_fixed_peak_{mhz}"] = 0.0
        neutral_defaults[f"radio_fixed_integral_{mhz}"] = 0.0
        neutral_defaults[f"radio_fixed_available_{mhz}"] = 0.0
    for key, value in neutral_defaults.items():
        if key in df.columns:
            df[key] = pd.to_numeric(df[key], errors="coerce").fillna(value)
    df.to_csv(OUTPUT_CSV, index=False)
    status = pd.DataFrame(
        [
            {"item": "rows", "value": len(df)},
            {"item": "target_rows", "value": len(targets)},
            {"item": "xrs_feature_rows", "value": xrs_ok},
            {"item": "proton_feature_rows", "value": proton_ok},
            {"item": "cdaw_cme_feature_rows", "value": cdaw_cme_ok},
            {"item": "rstn_fixed_freq_feature_rows", "value": rstn_ok},
            {"item": "hmi_cached_headers", "value": len(hmi)},
            {"item": "hmi_feature_rows", "value": hmi_ok},
            {"item": "targeted_xrs_min_year", "value": TARGETED_XRS_MIN_YEAR},
            {"item": "targeted_proton_min_year", "value": TARGETED_PROTON_MIN_YEAR},
            {"item": "targeted_rstn_min_year", "value": TARGETED_RSTN_MIN_YEAR},
            {"item": "targeted_xrs_offline", "value": int(TARGETED_XRS_OFFLINE)},
            {"item": "targeted_proton_offline", "value": int(TARGETED_PROTON_OFFLINE)},
            {"item": "targeted_rstn_offline", "value": int(TARGETED_RSTN_OFFLINE)},
            {"item": "targeted_cme_offline", "value": int(TARGETED_CME_OFFLINE)},
            {"item": "targeted_max_new_days", "value": TARGETED_MAX_NEW_DAYS},
            {"item": "targeted_context_days", "value": TARGETED_CONTEXT_DAYS},
            {"item": "new_xrs_day_fetches", "value": _NEW_DAY_FETCHES.get("xrs", 0)},
            {"item": "new_proton_day_fetches", "value": _NEW_DAY_FETCHES.get("proton", 0)},
            {"item": "new_rstn_day_fetches", "value": _NEW_DAY_FETCHES.get("rstn", 0)},
            {"item": "new_cme_day_fetches", "value": _NEW_DAY_FETCHES.get("cme", 0)},
        ]
    )
    status.to_csv(STATUS_CSV, index=False)
    print(f"Saved targeted enrichment: {OUTPUT_CSV}")
    print(status.to_string(index=False))


if __name__ == "__main__":
    main()
