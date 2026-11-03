#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RSTN Sagamore Hill — загрузка .k7o.gz с NOAA, разбор K7O, агрегация до 1 минуты.
Алгоритм как collect_rstn_data в ephin_parser (без GUI).

Зависимости:
  pip install pandas requests
"""
from __future__ import annotations

import argparse
import gzip
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

_STRUM = Path(__file__).resolve().parent.parent
if str(_STRUM) not in sys.path:
    sys.path.insert(0, str(_STRUM))

from data_paths import rstn_cache_dir  # noqa: E402

BASE_URL = (
    "https://www.ngdc.noaa.gov/stp/space-weather/solar-data/"
    "solar-features/solar-radio/rstn-1-second/"
)

STATIONS = {
    "sagamore-hill": {"icao": ("K7OL",), "legacy_ext": "sag", "dat_prefix": "sg"},
    "san-vito": {"icao": ("LISS",), "legacy_ext": "svt", "dat_prefix": "sv"},
    "learmonth": {"icao": ("APLM",), "legacy_ext": "lea", "dat_prefix": "ap"},
    # NOAA keeps post-2011 Ka'ena Point data in the Palehua path.
    "palehua": {"icao": ("PHFF",), "legacy_ext": "pal", "dat_prefix": "pl"},
}

MONTH_ABBR = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
RADIO_COLUMNS = [
    "Radio_245MHz",
    "Radio_410MHz",
    "Radio_610MHz",
    "Radio_1415MHz",
    "Radio_2695MHz",
    "Radio_4995MHz",
    "Radio_8800MHz",
    "Radio_15400MHz",
]


def _month_index_files(station: str, target_date: datetime) -> list[str]:
    url = f"{BASE_URL}{station}/{target_date.year}/{target_date.month:02d}/"
    response = requests.get(url, timeout=30)
    if response.status_code != 200:
        return []
    return [h for h in re.findall(r'href="([^"]+\.gz)"', response.text, flags=re.I)]


def _candidate_file_names(station: str, target_date: datetime) -> list[str]:
    cfg = STATIONS[station]
    yy = target_date.year % 100
    month_abbr = MONTH_ABBR[target_date.month - 1]
    upper = month_abbr.upper()
    return [
        f"{target_date:%Y%m%d}.{cfg['legacy_ext']}.gz",
        f"{target_date.day:02d}{month_abbr}{yy:02d}.{cfg['icao'][0][:3].lower()}.gz",
        f"{target_date.day:02d}{upper}{yy:02d}.{cfg['icao'][0][:3]}.gz",
        f"{cfg['dat_prefix']}{yy:02d}{target_date.month:02d}{target_date.day:02d}.dat.gz",
    ]


def _select_file_from_index(station: str, target_date: datetime) -> str | None:
    files = _month_index_files(station, target_date)
    if not files:
        return None
    lower_to_file = {f.lower(): f for f in files}
    for candidate in _candidate_file_names(station, target_date):
        hit = lower_to_file.get(candidate.lower())
        if hit:
            return hit
    ymd = target_date.strftime("%Y%m%d")
    ddmonyy = f"{target_date.day:02d}{MONTH_ABBR[target_date.month - 1]}{target_date.year % 100:02d}".lower()
    yymmdd = f"{target_date.year % 100:02d}{target_date.month:02d}{target_date.day:02d}"
    mmdd = f"{target_date.month:02d}{target_date.day:02d}"
    for f in files:
        lf = f.lower()
        if ymd in lf or ddmonyy in lf or yymmdd in lf:
            return f
    # Palehua 2001 has a few files named pl0004DD.dat.gz inside the 2001 directory.
    for f in files:
        lf = f.lower()
        if lf.endswith(".dat.gz") and mmdd in lf:
            return f
    return None


def _decode_response(response: requests.Response) -> str:
    return gzip.decompress(response.content).decode("utf-8", errors="ignore")


def _float_or_none(value: str) -> float | None:
    try:
        x = float(value)
    except Exception:
        return None
    if x in {-1.0, -9999.0}:
        return None
    return x


def _parse_compact_line(parts: list[str], station: str) -> dict | None:
    fp = parts[0]
    if len(fp) < 18:
        return None
    date_str = fp[4:12]
    time_str = fp[12:18]
    try:
        dt = datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
    except Exception:
        return None
    rec = {"DateTime": dt, "Station": station}
    for col, value in zip(RADIO_COLUMNS, parts[1:9]):
        rec[col] = _float_or_none(value)
    return rec


def _parse_epoch_line(parts: list[str], station: str) -> dict | None:
    if len(parts) < 9 or not re.fullmatch(r"\d{8,}", parts[0]):
        return None
    try:
        dt = datetime.utcfromtimestamp(int(float(parts[0])))
    except Exception:
        return None
    rec = {"DateTime": dt, "Station": station}
    for col, value in zip(RADIO_COLUMNS, parts[1:9]):
        rec[col] = _float_or_none(value)
    return rec


def _parse_legacy_pipe_line(line: str, station: str) -> dict | None:
    match = re.match(r"^([A-Z0-9]{4})\s+(\d{2}/\d{2}/\d{4})\s+(\d{2}:\d{2}:\d{2})\s*\|\s*(.*)$", line)
    if not match:
        return None
    try:
        dt = datetime.strptime(match.group(2) + " " + match.group(3), "%m/%d/%Y %H:%M:%S")
    except Exception:
        return None
    values = re.findall(r"[-+]?\d+(?:\.\d+)?(?:E[-+]?\d+)?", match.group(4), flags=re.I)
    if len(values) < 8:
        return None
    rec = {"DateTime": dt, "Station": station}
    for col, value in zip(RADIO_COLUMNS, values[:8]):
        rec[col] = _float_or_none(value)
    return rec


def _parse_rstn_text(text: str, station: str) -> list[dict]:
    day_records = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        rec = None
        if parts and re.match(r"^[A-Z0-9]{4}\d{14}$", parts[0]):
            rec = _parse_compact_line(parts, station)
        elif parts and re.fullmatch(r"\d{8,}", parts[0]):
            rec = _parse_epoch_line(parts, station)
        elif "|" in line:
            rec = _parse_legacy_pipe_line(line, station)
        if rec is not None:
            day_records.append(rec)
    return day_records


def fetch_station_day(target_date: datetime, station: str):
    filename = _select_file_from_index(station, target_date)
    if not filename:
        return []
    url = f"{BASE_URL}{station}/{target_date.year}/{target_date.month:02d}/{filename}"
    try:
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            return []
        return _parse_rstn_text(_decode_response(response), station)
    except Exception:
        return []


def fetch_network_day(target_date: datetime, stations: tuple[str, ...] | None = None):
    stations = stations or tuple(STATIONS)
    rows = []
    for station in stations:
        rows.extend(fetch_station_day(target_date, station))
    return rows


def fetch_day(target_date: datetime, cache_dir: str | None):
    """Backward-compatible Sagamore Hill fetcher."""
    return fetch_station_day(target_date, "sagamore-hill")


def fetch_day_legacy_name_builder(target_date: datetime, cache_dir: str | None):
    year, month, day = target_date.year, target_date.month, target_date.day
    month_abbr = MONTH_ABBR[month - 1]
    filename = f"sagamore-hill/{year}/{month:02d}/{day:02d}{month_abbr}{year % 100:02d}.k7o.gz"
    url = BASE_URL + filename
    day_records = []
    try:
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            return day_records
        text = gzip.decompress(response.content).decode("utf-8", errors="ignore")
        for line in text.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split()
            if not parts or not parts[0].startswith("K7"):
                continue
            try:
                fp = parts[0]
                date_str = fp[4:12]
                time_str = fp[12:18]
                dt = datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
                day_records.append({
                    "DateTime": dt,
                    "Station": fp[:4],
                    "Radio_245MHz": float(parts[1]) if len(parts) > 1 and parts[1] != "-1" else None,
                    "Radio_410MHz": float(parts[2]) if len(parts) > 2 and parts[2] != "-1" else None,
                    "Radio_610MHz": float(parts[3]) if len(parts) > 3 and parts[3] != "-1" else None,
                    "Radio_1415MHz": float(parts[4]) if len(parts) > 4 and parts[4] != "-1" else None,
                    "Radio_2695MHz": float(parts[5]) if len(parts) > 5 and parts[5] != "-1" else None,
                    "Radio_4995MHz": float(parts[6]) if len(parts) > 6 and parts[6] != "-1" else None,
                    "Radio_8800MHz": float(parts[7]) if len(parts) > 7 and parts[7] != "-1" else None,
                    "Radio_15400MHz": float(parts[8]) if len(parts) > 8 and parts[8] != "-1" else None,
                })
            except Exception:
                continue
    except Exception:
        pass
    return day_records


def aggregate_minute(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    num_cols = [c for c in df.columns if c.startswith("Radio_")]
    df["Minute"] = pd.to_datetime(df["DateTime"]).dt.floor("min")
    grouped = df.groupby("Minute", as_index=False)[num_cols].mean().rename(columns={"Minute": "DateTime"})
    stations = df.groupby("Minute")["Station"].agg(lambda x: ",".join(sorted(set(map(str, x)))))
    grouped["Stations"] = grouped["DateTime"].map(stations)
    grouped["Station_Count"] = grouped["DateTime"].map(df.groupby("Minute")["Station"].nunique()).fillna(0).astype(int)
    return grouped


def main():
    ap = argparse.ArgumentParser(description="RSTN 1-second CLI")
    ap.add_argument("-s", "--start", required=True)
    ap.add_argument("-e", "--end", required=True)
    ap.add_argument("-o", "--output", default="rstn_1min.csv")
    ap.add_argument("-j", "--jobs", type=int, default=10, help="потоки загрузки")
    ap.add_argument(
        "--stations",
        default="sagamore-hill",
        help="comma-separated stations or 'all': sagamore-hill,san-vito,learmonth,palehua",
    )
    ap.add_argument(
        "--cache",
        default=rstn_cache_dir,
        help="каталог кэша RSTN (по умолчанию — data/rstn_cache от корня репозитория; резерв)",
    )
    args = ap.parse_args()
    os.makedirs(args.cache, exist_ok=True)
    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")
    if end < start:
        sys.exit("--end must be >= --start")

    days = []
    cur = start
    while cur <= end:
        days.append(cur)
        cur += timedelta(days=1)

    stations = tuple(STATIONS) if args.stations.strip().lower() == "all" else tuple(s.strip() for s in args.stations.split(",") if s.strip())
    all_rows = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        futs = {ex.submit(fetch_network_day, d, stations): d for d in days}
        for fut in as_completed(futs):
            all_rows.extend(fut.result())

    agg = aggregate_minute(all_rows)
    if agg.empty:
        print("Нет данных", file=sys.stderr)
        sys.exit(1)
    agg.to_csv(args.output, index=False)
    print(f"✓ 1-с исходно: {len(all_rows)}, 1-мин строк: {len(agg)} → {args.output}")


if __name__ == "__main__":
    main()
