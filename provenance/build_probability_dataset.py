import sys
from pathlib import Path
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import math
import os

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
STRUMINSKY_DIR = ROOT / "struminsky"
for path in (ROOT, STRUMINSKY_DIR, HERE.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import garcia_balch_sep_sunpy as sep_ml
from run_gamma_blocks_chrono import _load_type2_catalog, _goes_class_score, _type2_catalog_feature

DATASET_CSV = HERE / "sep_probability_dataset_cycles24_25_c1_with_onset.csv"
RAW_CHECKPOINT_CSV = HERE / "sep_probability_hek_c1_raw_checkpoint_with_onset.csv"
WINDOW_CHECKPOINT_CSV = HERE / "sep_probability_hek_c1_window_checkpoint_with_onset.csv"
START_TIME = pd.Timestamp("2008-12-01T00:00:00Z")
END_TIME = pd.Timestamp("2026-12-31T23:59:59Z")
MIN_FLARE_SCORE = 100.0  # C1.0
MAX_WORKERS = 5
HEK_TIMEOUT_SECONDS = 90

def fetch_flares_for_window(label: str, lo_ts: pd.Timestamp, hi_ts: pd.Timestamp) -> list[dict]:
    from sunpy.net import hek
    from sunpy.net import attrs as a
    import urllib.request
    import ssl
    import socket
    ssl._create_default_https_context = ssl._create_unverified_context
    socket.setdefaulttimeout(HEK_TIMEOUT_SECONDS)
    
    client = hek.HEKClient()
    lo = max(lo_ts, START_TIME).to_pydatetime()
    hi = min(hi_ts, END_TIME).to_pydatetime()
    if lo >= hi:
        return []
    print(f"Fetching HEK flares for {label}...", flush=True)
    
    try:
        res = client.search(a.Time(lo, hi), a.hek.FL)
    except Exception as e:
        print(f"Error fetching flares for {label}: {e}", flush=True)
        return []

    def get_value(row: object, *names: str) -> object:
        for name in names:
            try:
                value = row.get(name)
                if value is not None:
                    return value
            except Exception:
                pass
            try:
                value = row[name]
                if value is not None:
                    return value
            except Exception:
                pass
        return None

    def to_utc(value: object) -> pd.Timestamp | None:
        if value is None:
            return None
        try:
            if hasattr(value, "to_datetime"):
                return pd.Timestamp(value.to_datetime(), tz="UTC")
            if hasattr(value, "isot"):
                return pd.Timestamp(str(value.isot), tz="UTC")
            ts = pd.Timestamp(value)
            return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        except Exception:
            return None

    rows = []
    for row in res:
        cls = str(get_value(row, "fl_goescls", "FL_GOESCLS") or "").strip()
        score = _goes_class_score(cls)
        if score < MIN_FLARE_SCORE:
            continue
            
        start_time = get_value(row, "event_starttime", "EVENT_STARTTIME")
        peak_time = get_value(row, "event_peaktime", "EVENT_PEAKTIME", "event_starttime", "EVENT_STARTTIME")
        end_time = get_value(row, "event_endtime", "EVENT_ENDTIME")
        st = to_utc(start_time)
        pt = to_utc(peak_time)
        et = to_utc(end_time)
        if pt is None:
            continue

        lon = pd.to_numeric(get_value(row, "event_coord1", "EVENT_COORD1", "hgs_x", "HGS_X", "hgs_lon", "HGS_LON"), errors="coerce")
        lon_value = float(lon) if np.isfinite(lon) else np.nan
            
        rows.append({
            "hek_event_id": get_value(row, "kb_archivid", "KB_ARCHIVID") or "",
            "event_starttime": st.isoformat() if st is not None else "",
            "event_peaktime": pt.isoformat(),
            "event_endtime": et.isoformat() if et is not None else "",
            "flare_onset_utc": st.isoformat() if st is not None else "",
            "flare_peak_utc": pt.isoformat(),
            "flare_class": cls,
            "goes_class_score": score,
            "lon_w_deg": lon_value,
            "lon_available": int(np.isfinite(lon_value)),
            "source_window": label,
        })
    print(f"  -> Found {len(rows)} flares >= C1 in {label}", flush=True)
    return rows

def _write_raw_checkpoint(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if not df.empty:
        df["event_peaktime"] = pd.to_datetime(df["event_peaktime"], utc=True, errors="coerce", format="mixed")
        df = df.sort_values("event_peaktime").drop_duplicates("event_peaktime").reset_index(drop=True)
    df.to_csv(RAW_CHECKPOINT_CSV, index=False)
    print(f"Raw checkpoint saved: {len(df)} rows -> {RAW_CHECKPOINT_CSV}", flush=True)
    return df

def _load_completed_windows_from_checkpoint() -> set[str]:
    if not WINDOW_CHECKPOINT_CSV.exists():
        return set()
    df = pd.read_csv(WINDOW_CHECKPOINT_CSV)
    if df.empty or "source_window" not in df.columns or "status" not in df.columns:
        return set()
    done = df[df["status"].astype(str).eq("ok")]
    return set(pd.Series(done["source_window"], dtype=str).dropna().tolist())

def _write_window_status(label: str, status: str, n_rows: int, note: str = "") -> None:
    rows = []
    if WINDOW_CHECKPOINT_CSV.exists():
        old = pd.read_csv(WINDOW_CHECKPOINT_CSV)
        if not old.empty:
            rows = old.to_dict("records")
    rows = [row for row in rows if str(row.get("source_window")) != str(label)]
    rows.append(
        {
            "source_window": label,
            "status": status,
            "n_rows": int(n_rows),
            "note": str(note),
        }
    )
    pd.DataFrame(rows).sort_values("source_window").to_csv(WINDOW_CHECKPOINT_CSV, index=False)

def _build_final_dataset(df_flares: pd.DataFrame) -> pd.DataFrame:
    if df_flares.empty:
        return df_flares

    df_flares = df_flares.copy()
    if "event_starttime" in df_flares.columns:
        df_flares["event_starttime"] = pd.to_datetime(df_flares["event_starttime"], utc=True, errors="coerce", format="mixed")
    df_flares["event_peaktime"] = pd.to_datetime(df_flares["event_peaktime"], utc=True, errors="coerce", format="mixed")
    if "event_endtime" in df_flares.columns:
        df_flares["event_endtime"] = pd.to_datetime(df_flares["event_endtime"], utc=True, errors="coerce", format="mixed")
    if "flare_onset_utc" not in df_flares.columns:
        df_flares["flare_onset_utc"] = df_flares.get("event_starttime", "")
    if "flare_peak_utc" not in df_flares.columns:
        df_flares["flare_peak_utc"] = df_flares["event_peaktime"]

    # Label with True/False Alarm using NOAA SEP Catalog
    cat = sep_ml.dataframe_from_sep_benchmark_raw_file()
    sep_onsets = pd.to_datetime(cat["event_begin_utc"], utc=True)

    df_flares["is_sep_event"] = 0
    df_flares["associated_sep_onset"] = ""

    for onset in sep_onsets:
        # A flare is associated with the SEP if it occurs within 24 hours before the SEP onset
        lo = onset - pd.Timedelta(hours=24)
        hi = onset

        mask = (df_flares["event_peaktime"] >= lo) & (df_flares["event_peaktime"] <= hi)
        candidates = df_flares[mask]

        if not candidates.empty:
            # Pick the strongest flare
            best_idx = candidates["goes_class_score"].idxmax()
            df_flares.at[best_idx, "is_sep_event"] = 1
            df_flares.at[best_idx, "associated_sep_onset"] = onset.isoformat()

    print(f"Total positive SEP events matched: {df_flares['is_sep_event'].sum()}", flush=True)

    # Calculate physics features needed for VQC Classifier
    lon = pd.to_numeric(df_flares["lon_w_deg"], errors="coerce")
    df_flares["connection_factor_w45"] = np.where(np.isfinite(lon), np.cos(np.deg2rad(lon - 45.0)), 0.0)
    df_flares["log10_goes_peak_flux_wm2"] = np.log10(pd.to_numeric(df_flares["goes_class_score"], errors="coerce") * 1e-8)

    type2_powers = []
    for pt in df_flares["event_peaktime"]:
        t0 = pt - pd.Timedelta(hours=1)
        t1 = pt + pd.Timedelta(hours=4)
        f = _type2_catalog_feature(t0, t1)
        type2_powers.append(f["log_power"])
    df_flares["radio_type_ii_log_power"] = type2_powers
    return df_flares

def build_dataset():
    windows = []
    for month_start in pd.date_range(START_TIME.normalize(), END_TIME.normalize(), freq="MS", tz="UTC"):
        month_end = month_start + pd.offsets.MonthBegin(1) - pd.Timedelta(seconds=1)
        lo = max(month_start, START_TIME)
        hi = min(pd.Timestamp(month_end), END_TIME)
        if lo < hi:
            windows.append((month_start.strftime("%Y-%m"), lo, hi))
    if RAW_CHECKPOINT_CSV.exists():
        checkpoint = pd.read_csv(RAW_CHECKPOINT_CSV)
        flares = checkpoint.to_dict("records")
        completed = set(pd.Series(checkpoint.get("source_window", []), dtype=str).dropna().tolist())
        completed |= _load_completed_windows_from_checkpoint()
        print(f"Loaded raw checkpoint: {len(flares)} rows, {len(completed)} completed windows.", flush=True)
    else:
        flares = []
        completed = set()
    windows = [window for window in windows if window[0] not in completed]
    if not windows:
        print("All monthly windows already fetched in raw checkpoint.", flush=True)
        df_flares = _write_raw_checkpoint(flares)
        final = _build_final_dataset(df_flares)
        final.to_csv(DATASET_CSV, index=False)
        print(f"Saved completed dataset to {DATASET_CSV}", flush=True)
        return
    print(f"Fetching HEK flares in monthly windows with {MAX_WORKERS} worker threads...", flush=True)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(fetch_flares_for_window, label, lo, hi): label for label, lo, hi in windows}
        for future in as_completed(futures):
            label = futures[future]
            try:
                rows = future.result()
                flares.extend(rows)
                _write_window_status(label, "ok", len(rows))
                _write_raw_checkpoint(flares)
            except Exception as exc:
                _write_window_status(label, "failed", 0, str(exc))
                print(f"Window {label} failed: {exc}", flush=True)
        
    df_flares = _write_raw_checkpoint(flares)
    
    if df_flares.empty:
        print("No flares found!")
        return

    print(f"Total >= C1 flares found for solar cycles 24/25: {len(df_flares)}", flush=True)
    final = _build_final_dataset(df_flares)
    final.to_csv(DATASET_CSV, index=False)
    print(f"Saved completed dataset to {DATASET_CSV}", flush=True)

if __name__ == "__main__":
    build_dataset()
