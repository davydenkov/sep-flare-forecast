from __future__ import annotations

import math
import socket
import ssl
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
BASE_DATASET_CSV = HERE / "sep_probability_dataset_cycles24_25_c1_with_onset.csv"
ENRICHED_DATASET_CSV = HERE / "sep_probability_dataset_cycles24_25_c1_enriched.csv"
CME_CHECKPOINT_CSV = HERE / "sep_probability_hek_cme_monthly_checkpoint.csv"

START_TIME = pd.Timestamp("2008-12-01T00:00:00Z")
END_TIME = pd.Timestamp("2026-12-31T23:59:59Z")
MAX_WORKERS = 5
HEK_TIMEOUT_SECONDS = 90


def _to_utc(value: object) -> pd.Timestamp | None:
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


def _get_value(row: object, *names: str) -> object:
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


def _num(value: object) -> float:
    out = pd.to_numeric(value, errors="coerce")
    try:
        out_f = float(out)
    except Exception:
        return math.nan
    return out_f if math.isfinite(out_f) else math.nan


def fetch_cme_for_month(label: str, lo_ts: pd.Timestamp, hi_ts: pd.Timestamp) -> list[dict[str, object]]:
    ssl._create_default_https_context = ssl._create_unverified_context
    socket.setdefaulttimeout(HEK_TIMEOUT_SECONDS)

    from sunpy.net import attrs as a
    from sunpy.net import hek

    client = hek.HEKClient()
    lo = (lo_ts - pd.Timedelta(hours=2)).to_pydatetime()
    hi = (hi_ts + pd.Timedelta(hours=8)).to_pydatetime()
    print(f"Fetching HEK CME for {label}...", flush=True)
    try:
        res = client.search(a.Time(lo, hi), a.hek.CE)
    except Exception as exc:
        print(f"Error fetching HEK CME for {label}: {exc}", flush=True)
        return []

    rows: list[dict[str, object]] = []
    for row in res:
        start = _to_utc(_get_value(row, "event_starttime", "EVENT_STARTTIME"))
        peak = _to_utc(_get_value(row, "event_peaktime", "EVENT_PEAKTIME"))
        end = _to_utc(_get_value(row, "event_endtime", "EVENT_ENDTIME"))
        ref = peak or start or end
        if ref is None:
            continue
        speed = _num(
            _get_value(
                row,
                "cme_radiallinvel",
                "CME_RADIALLINVEL",
                "cme_radiallinvelmax",
                "CME_RADIALLINVELMAX",
                "ce_velocity",
                "CE_VELOCITY",
                "speed",
                "SPEED",
            )
        )
        width = _num(
            _get_value(
                row,
                "cme_angularwidth",
                "CME_ANGULARWIDTH",
                "ce_width",
                "CE_WIDTH",
                "width",
                "WIDTH",
                "angular_width",
            )
        )
        rows.append(
            {
                "cme_source_window": label,
                "cme_event_id": _get_value(row, "kb_archivid", "KB_ARCHIVID") or "",
                "cme_ref_time": ref.isoformat(),
                "cme_starttime": start.isoformat() if start is not None else "",
                "cme_peaktime": peak.isoformat() if peak is not None else "",
                "cme_endtime": end.isoformat() if end is not None else "",
                "hek_cme_speed_kms": speed,
                "hek_cme_width_deg": width,
            }
        )
    print(f"  -> Found {len(rows)} HEK CME rows in {label}", flush=True)
    return rows


def _monthly_windows() -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    out = []
    for month_start in pd.date_range(START_TIME.normalize(), END_TIME.normalize(), freq="MS", tz="UTC"):
        month_end = month_start + pd.offsets.MonthBegin(1) - pd.Timedelta(seconds=1)
        lo = max(month_start, START_TIME)
        hi = min(pd.Timestamp(month_end), END_TIME)
        if lo < hi:
            out.append((month_start.strftime("%Y-%m"), lo, hi))
    return out


def load_or_fetch_cme() -> pd.DataFrame:
    if CME_CHECKPOINT_CSV.exists():
        existing = pd.read_csv(CME_CHECKPOINT_CSV)
        speed = pd.to_numeric(existing.get("hek_cme_speed_kms"), errors="coerce")
        if len(existing) > 0 and int(speed.notna().sum()) == 0:
            print("Existing CME checkpoint has no parsed speed values; rebuilding with HEK cme_radiallinvel fields.", flush=True)
            done = set()
            rows = []
        else:
            cme_id = existing.get("cme_event_id", pd.Series([""] * len(existing))).fillna("").astype(str)
            nonempty = existing[cme_id.ne("")]
            done = set(pd.Series(nonempty.get("cme_source_window", []), dtype=str).dropna().tolist())
            rows = existing.to_dict("records")
            print(f"Loaded CME checkpoint: {len(rows)} rows, {len(done)} completed windows.", flush=True)
    else:
        done = set()
        rows = []

    windows = [win for win in _monthly_windows() if win[0] not in done]
    if windows:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(fetch_cme_for_month, label, lo, hi): label for label, lo, hi in windows}
            for future in as_completed(futures):
                label = futures[future]
                try:
                    rows.extend(future.result())
                except Exception as exc:
                    print(f"CME window {label} failed: {exc}", flush=True)
                # Add a sentinel row for empty months so they are not repeated.
                if not any(str(row.get("cme_source_window")) == label for row in rows):
                    rows.append({"cme_source_window": label, "cme_event_id": "", "cme_ref_time": ""})
                pd.DataFrame(rows).to_csv(CME_CHECKPOINT_CSV, index=False)
                print(f"CME checkpoint saved: {len(rows)} rows", flush=True)

    cme = pd.DataFrame(rows)
    if cme.empty:
        return cme
    cme = cme[cme.get("cme_ref_time", "").astype(str).ne("")].copy()
    if cme.empty:
        return cme
    cme["cme_ref_time"] = pd.to_datetime(cme["cme_ref_time"], utc=True, errors="coerce", format="mixed")
    cme["hek_cme_speed_kms"] = pd.to_numeric(cme.get("hek_cme_speed_kms"), errors="coerce")
    cme["hek_cme_width_deg"] = pd.to_numeric(cme.get("hek_cme_width_deg"), errors="coerce")
    return cme.dropna(subset=["cme_ref_time"]).sort_values("cme_ref_time").reset_index(drop=True)


def _match_cme_features(flares: pd.DataFrame, cme: pd.DataFrame) -> pd.DataFrame:
    out = flares.copy()
    defaults = {
        "hek_cme_flag": 0.0,
        "hek_cme_speed_kms": 0.0,
        "hek_cme_width_deg": 0.0,
        "hek_cme_speed_available": 0.0,
        "hek_cme_width_available": 0.0,
        "hek_cme_time_delta_min": np.nan,
    }
    for key, val in defaults.items():
        out[key] = val
    if cme.empty:
        return out

    for idx, row in out.iterrows():
        peak = row["event_peaktime"]
        lo = peak - pd.Timedelta(hours=2)
        hi = peak + pd.Timedelta(hours=8)
        sub = cme[(cme["cme_ref_time"] >= lo) & (cme["cme_ref_time"] <= hi)].copy()
        if sub.empty:
            continue
        sub["score_speed"] = pd.to_numeric(sub["hek_cme_speed_kms"], errors="coerce").fillna(-1.0)
        sub["abs_dt"] = (sub["cme_ref_time"] - peak).abs().dt.total_seconds() / 60.0
        best = sub.sort_values(["score_speed", "abs_dt"], ascending=[False, True]).iloc[0]
        speed = _num(best.get("hek_cme_speed_kms"))
        width = _num(best.get("hek_cme_width_deg"))
        out.at[idx, "hek_cme_flag"] = 1.0
        out.at[idx, "hek_cme_speed_kms"] = speed if math.isfinite(speed) else 0.0
        out.at[idx, "hek_cme_width_deg"] = width if math.isfinite(width) else 0.0
        out.at[idx, "hek_cme_speed_available"] = float(math.isfinite(speed) and speed > 0)
        out.at[idx, "hek_cme_width_available"] = float(math.isfinite(width) and width > 0)
        out.at[idx, "hek_cme_time_delta_min"] = float((best["cme_ref_time"] - peak).total_seconds() / 60.0)
    return out


def enrich_dataset() -> None:
    if not BASE_DATASET_CSV.exists():
        raise FileNotFoundError(BASE_DATASET_CSV)
    df = pd.read_csv(BASE_DATASET_CSV)
    df["event_starttime"] = pd.to_datetime(df["event_starttime"], utc=True, errors="coerce", format="mixed")
    df["event_peaktime"] = pd.to_datetime(df["event_peaktime"], utc=True, errors="coerce", format="mixed")
    df["event_endtime"] = pd.to_datetime(df["event_endtime"], utc=True, errors="coerce", format="mixed")

    goes_peak_flux = pd.to_numeric(df["goes_class_score"], errors="coerce") * 1.0e-8
    duration_min = (df["event_endtime"] - df["event_starttime"]).dt.total_seconds() / 60.0
    peak_delay_min = (df["event_peaktime"] - df["event_starttime"]).dt.total_seconds() / 60.0
    duration_min = duration_min.where(np.isfinite(duration_min) & (duration_min > 0), peak_delay_min)
    duration_min = duration_min.clip(lower=1.0, upper=24.0 * 60.0)

    lon = pd.to_numeric(df["lon_w_deg"], errors="coerce")
    df["longitude_quality_score"] = ((np.isfinite(lon)) & (lon.abs() <= 90.0)).astype(float)
    df["abs_lon_w_deg"] = lon.abs().fillna(999.0)
    df["west_longitude_flag"] = (lon > 0.0).astype(float).where(np.isfinite(lon), 0.0)
    df["hek_sxr_duration_min"] = duration_min.astype(float)
    df["log10_hek_sxr_duration_min"] = np.log10(np.maximum(duration_min.astype(float), 1.0))
    df["hek_sxr_fluence_proxy_wm2s"] = goes_peak_flux * duration_min.astype(float) * 60.0
    df["log10_hek_sxr_fluence_proxy_wm2s"] = np.log10(np.maximum(df["hek_sxr_fluence_proxy_wm2s"], 1.0e-12))
    df["type_ii_strength"] = pd.to_numeric(df.get("radio_type_ii_log_power"), errors="coerce").fillna(0.0)
    df["type_ii_flag"] = (df["type_ii_strength"] > 0.0).astype(float)

    cme = load_or_fetch_cme()
    df = _match_cme_features(df, cme)
    speed = pd.to_numeric(df["hek_cme_speed_kms"], errors="coerce").fillna(0.0).clip(lower=0.0)
    width = pd.to_numeric(df["hek_cme_width_deg"], errors="coerce").fillna(0.0).clip(lower=0.0)
    df["hek_cme_speed_kms"] = speed
    df["hek_cme_width_deg"] = width
    df["log1p_hek_cme_speed"] = np.log1p(speed)
    df["log1p_hek_cme_width"] = np.log1p(width)
    df["hek_cme_shock_proxy"] = df["log1p_hek_cme_speed"] * (1.0 + df["type_ii_flag"]) * df["hek_cme_flag"]

    df.to_csv(ENRICHED_DATASET_CSV, index=False)
    print(f"Saved enriched dataset: {ENRICHED_DATASET_CSV}")
    print(f"Rows: {len(df)}, positives: {int(pd.to_numeric(df['is_sep_event'], errors='coerce').fillna(0).sum())}")
    print(f"CME matched rows: {int(df['hek_cme_flag'].sum())}")
    print(f"CME speed available rows: {int(df['hek_cme_speed_available'].sum())}")


if __name__ == "__main__":
    enrich_dataset()
