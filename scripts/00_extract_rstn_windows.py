"""Stage A (optional): cut RSTN minute data around every flare of the input table.

The RSTN day files (one CSV per UT day, minute averages over the stations
Sagamore Hill, San Vito, Learmonth and Palehua/Kaena Point) are produced by
``provenance/rstn_sagamore_cli.py`` from the NOAA NCEI archive
https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-features/solar-radio/rstn-1-second/
They occupy ~200 MB and are therefore not stored in the repository; this script
reduces them to ``data/external/rstn_event_windows.csv.gz`` (included), which is
all that the offline analysis needs.

Usage:  python scripts/00_extract_rstn_windows.py /path/to/rstn_network_days
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRE_MIN, POST_MIN, DEFAULT_DURATION_MIN = 60, 30, 90


def main(day_dir: Path) -> None:
    table = pd.read_csv(ROOT / "data/input/upstream_m2_cme_radio_table.csv", low_memory=False)
    start = pd.to_datetime(table["event_starttime"], utc=True, format="mixed").dt.tz_localize(None)
    end = pd.to_datetime(table["event_endtime"], utc=True, format="mixed", errors="coerce").dt.tz_localize(None)
    end = end.where(end > start, start + pd.Timedelta(minutes=DEFAULT_DURATION_MIN))
    cache: dict[str, pd.DataFrame] = {}
    pieces = []
    for event_id, t0, t1 in zip(table["hek_event_id"], start, end):
        lo, hi = t0 - pd.Timedelta(minutes=PRE_MIN), t1 + pd.Timedelta(minutes=POST_MIN)
        frames = []
        for day in pd.date_range(lo.normalize(), hi.normalize(), freq="D"):
            key = f"{day:%Y%m%d}"
            if key not in cache:
                path = day_dir / f"{key}.csv"
                cache[key] = pd.read_csv(path, parse_dates=["DateTime"]) if path.exists() else pd.DataFrame()
            frames.append(cache[key])
        data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if data.empty:
            continue
        sub = data[(data["DateTime"] >= lo) & (data["DateTime"] <= hi)].copy()
        if sub.empty:
            continue
        sub.insert(0, "hek_event_id", event_id)
        pieces.append(sub.drop(columns=["Stations"], errors="ignore"))
    out = pd.concat(pieces, ignore_index=True)
    out.to_csv(ROOT / "data/external/rstn_event_windows.csv.gz", index=False, float_format="%.3f", compression="gzip")
    print(f"events with RSTN data: {out['hek_event_id'].nunique()} of {len(table)}; rows: {len(out)}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
