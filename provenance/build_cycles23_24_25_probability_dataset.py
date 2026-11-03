from __future__ import annotations

import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
STRUMINSKY_DIR = ROOT / "struminsky"
for path in (ROOT, STRUMINSKY_DIR, HERE.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import build_probability_dataset as base  # noqa: E402
import enrich_probability_features as enrich  # noqa: E402


CYCLE23_START = pd.Timestamp("1996-05-01T00:00:00Z")
CYCLE23_END = pd.Timestamp("2008-11-30T23:59:59Z")
FULL_END = pd.Timestamp("2026-12-31T23:59:59Z")

CYCLE23_RAW_CSV = HERE / "sep_probability_hek_c1_raw_checkpoint_cycle23.csv"
CYCLE23_WINDOW_CSV = HERE / "sep_probability_hek_c1_window_checkpoint_cycle23.csv"
BASE_24_25_RAW_CSV = HERE / "sep_probability_hek_c1_raw_checkpoint_with_onset.csv"
DATASET_CSV = HERE / "sep_probability_dataset_cycles23_24_25_c1_with_onset.csv"
ENRICHED_CSV = HERE / "sep_probability_dataset_cycles23_24_25_c1_enriched.csv"
STRUMINSKY_CSV = HERE / "sep_probability_dataset_cycles23_24_25_c1_struminsky.csv"
CME_CHECKPOINT_CSV = HERE / "sep_probability_hek_cme_monthly_checkpoint_cycles23_24_25.csv"
OLD_CME_CHECKPOINT_CSV = HERE / "sep_probability_hek_cme_monthly_checkpoint.csv"


def _write_cycle23_raw(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if not df.empty:
        df["event_peaktime"] = pd.to_datetime(df["event_peaktime"], utc=True, errors="coerce", format="mixed")
        df = df.sort_values("event_peaktime").drop_duplicates("event_peaktime").reset_index(drop=True)
    df.to_csv(CYCLE23_RAW_CSV, index=False)
    print(f"Cycle 23 raw checkpoint saved: {len(df)} rows -> {CYCLE23_RAW_CSV}", flush=True)
    return df


def _load_completed_windows() -> set[str]:
    if not CYCLE23_WINDOW_CSV.exists():
        return set()
    df = pd.read_csv(CYCLE23_WINDOW_CSV)
    if df.empty or "source_window" not in df.columns or "status" not in df.columns:
        return set()
    return set(df[df["status"].astype(str).eq("ok")]["source_window"].astype(str).tolist())


def _write_window_status(label: str, status: str, n_rows: int, note: str = "") -> None:
    rows = []
    if CYCLE23_WINDOW_CSV.exists():
        old = pd.read_csv(CYCLE23_WINDOW_CSV)
        if not old.empty:
            rows = old.to_dict("records")
    rows = [row for row in rows if str(row.get("source_window")) != str(label)]
    rows.append({"source_window": label, "status": status, "n_rows": int(n_rows), "note": str(note)})
    pd.DataFrame(rows).sort_values("source_window").to_csv(CYCLE23_WINDOW_CSV, index=False)


def _cycle23_windows() -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    windows = []
    for month_start in pd.date_range(CYCLE23_START.normalize(), CYCLE23_END.normalize(), freq="MS", tz="UTC"):
        month_end = month_start + pd.offsets.MonthBegin(1) - pd.Timedelta(seconds=1)
        lo = max(month_start, CYCLE23_START)
        hi = min(pd.Timestamp(month_end), CYCLE23_END)
        if lo < hi:
            windows.append((month_start.strftime("%Y-%m"), lo, hi))
    return windows


def fetch_cycle23_flares() -> pd.DataFrame:
    base.START_TIME = CYCLE23_START
    base.END_TIME = CYCLE23_END
    if CYCLE23_RAW_CSV.exists():
        checkpoint = pd.read_csv(CYCLE23_RAW_CSV)
        flares = checkpoint.to_dict("records")
        completed = set(pd.Series(checkpoint.get("source_window", []), dtype=str).dropna().tolist())
        completed |= _load_completed_windows()
        print(f"Loaded cycle 23 raw checkpoint: {len(flares)} rows, {len(completed)} completed windows.", flush=True)
    else:
        flares = []
        completed = set()

    windows = [window for window in _cycle23_windows() if window[0] not in completed]
    if windows:
        print(f"Fetching cycle 23 HEK C1+ flares in {len(windows)} monthly windows...", flush=True)
        with ThreadPoolExecutor(max_workers=base.MAX_WORKERS) as pool:
            futures = {pool.submit(base.fetch_flares_for_window, label, lo, hi): label for label, lo, hi in windows}
            for future in as_completed(futures):
                label = futures[future]
                try:
                    rows = future.result()
                    flares.extend(rows)
                    _write_window_status(label, "ok", len(rows))
                    _write_cycle23_raw(flares)
                except Exception as exc:
                    _write_window_status(label, "failed", 0, str(exc))
                    print(f"Cycle 23 window {label} failed: {exc}", flush=True)
    return _write_cycle23_raw(flares)


def build_combined_dataset(cycle23_raw: pd.DataFrame) -> pd.DataFrame:
    if not BASE_24_25_RAW_CSV.exists():
        raise FileNotFoundError(BASE_24_25_RAW_CSV)
    current_raw = pd.read_csv(BASE_24_25_RAW_CSV)
    raw = pd.concat([cycle23_raw, current_raw], ignore_index=True, sort=False)
    raw["event_peaktime"] = pd.to_datetime(raw["event_peaktime"], utc=True, errors="coerce", format="mixed")
    raw = raw.dropna(subset=["event_peaktime"]).sort_values("event_peaktime").drop_duplicates("event_peaktime").reset_index(drop=True)
    final = base._build_final_dataset(raw)
    final.to_csv(DATASET_CSV, index=False)
    print(f"Saved combined cycles 23/24/25 dataset: {DATASET_CSV}", flush=True)
    print(f"Rows: {len(final)}, positives: {int(final['is_sep_event'].sum())}", flush=True)
    return final


def enrich_combined_dataset() -> None:
    if not CME_CHECKPOINT_CSV.exists() and OLD_CME_CHECKPOINT_CSV.exists():
        shutil.copy2(OLD_CME_CHECKPOINT_CSV, CME_CHECKPOINT_CSV)
        print(f"Seeded cycle 23/24/25 CME checkpoint from {OLD_CME_CHECKPOINT_CSV}", flush=True)

    enrich.BASE_DATASET_CSV = DATASET_CSV
    enrich.ENRICHED_DATASET_CSV = ENRICHED_CSV
    enrich.CME_CHECKPOINT_CSV = CME_CHECKPOINT_CSV
    enrich.START_TIME = CYCLE23_START
    enrich.END_TIME = FULL_END
    enrich.enrich_dataset()


def main() -> None:
    cycle23_raw = fetch_cycle23_flares()
    build_combined_dataset(cycle23_raw)
    enrich_combined_dataset()


if __name__ == "__main__":
    main()
