"""Parsers for the public solar-event catalogues used to clean the event sample.

All parsers read local copies of public NOAA/NASA products stored in
``data/external``; no network access is required here.
"""
from __future__ import annotations

import html
import re
from pathlib import Path

import numpy as np
import pandas as pd

CLASS_BASE = {"A": 1.0, "B": 10.0, "C": 100.0, "M": 1000.0, "X": 10000.0}


def class_to_score(letter: str, magnitude: float) -> float:
    """GOES score G = peak 1-8 A flux / 1e-8 W m^-2 (C1.0 -> 100, M1.0 -> 1000)."""
    return CLASS_BASE.get(letter.upper(), np.nan) * magnitude


def parse_location(loc: str) -> tuple[float, float]:
    """'S20W85' -> (lat=-20, lon=+85); heliographic, longitude positive to the west."""
    m = re.match(r"^([NS])(\d{2})([EW])(\d{2,3})$", str(loc).strip().upper())
    if not m:
        return np.nan, np.nan
    lat = float(m.group(2)) * (1 if m.group(1) == "N" else -1)
    lon = float(m.group(4)) * (1 if m.group(3) == "W" else -1)
    if abs(lon) > 120 or abs(lat) > 60:
        return np.nan, np.nan
    return lat, lon


def _times(day: pd.Timestamp, hhmm: list[str]) -> list[pd.Timestamp | None]:
    out: list[pd.Timestamp | None] = []
    prev = None
    for s in hhmm:
        s = s.strip()
        if not re.fullmatch(r"\d{4}", s):
            out.append(None)
            continue
        t = day + pd.Timedelta(hours=int(s[:2]), minutes=int(s[2:]))
        if prev is not None and t < prev:
            t += pd.Timedelta(days=1)
        out.append(t)
        prev = t
    return out


def read_ngdc_xrs_reports(folder: Path) -> pd.DataFrame:
    """NGDC/NCEI GOES XRS flare reports (fixed-width, 1997-2017)."""
    rows = []
    for path in sorted(folder.glob("goes-xrs-report_*.txt")):
        for line in path.read_text(encoding="latin-1").splitlines():
            if not line.startswith("31777") or len(line) < 63:
                continue
            yy, mm, dd = int(line[5:7]), int(line[7:9]), int(line[9:11])
            year = 1900 + yy if yy >= 75 else 2000 + yy
            try:
                day = pd.Timestamp(year=year, month=mm, day=dd, tz="UTC")
            except ValueError:
                continue
            start, peak, end = _times(day, [line[13:17], line[23:27], line[18:22]])
            letter = line[59].strip()
            digits = line[60:63].strip()
            if letter not in CLASS_BASE or not digits.isdigit():
                continue
            magnitude = int(digits) / 10.0
            lat, lon = parse_location(line[28:34])
            region = re.search(r"E[+-]?\d\d\s+(\d{4,5})", line[63:])
            fluence = re.search(r"(\d\.\dE[+-]?\d{2})", line[63:])
            rows.append(
                {
                    "catalog": "NGDC-XRS",
                    "start": start,
                    "peak": peak,
                    "end": end,
                    "goes_class": f"{letter}{magnitude:.1f}",
                    "goes_score": class_to_score(letter, magnitude),
                    "lat_deg": lat,
                    "lon_w_deg": lon,
                    "noaa_region": float(region.group(1)) if region else np.nan,
                    "sxr_fluence_jm2": float(fluence.group(1)) if fluence else np.nan,
                }
            )
    df = pd.DataFrame(rows).dropna(subset=["peak"])
    return df.drop_duplicates(subset=["peak", "goes_class"]).reset_index(drop=True)


def read_swpc_event_reports(folder: Path) -> pd.DataFrame:
    """SWPC 'Edited Events' daily reports: XRA (GOES class) joined with FLA (location)."""
    rows = []
    for path in sorted(folder.glob("*events.txt")):
        day = pd.Timestamp(path.name[:8], tz="UTC")
        groups: dict[str, dict] = {}
        for line in path.read_text(encoding="latin-1").splitlines():
            if not line or line.startswith((":", "#")):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            event_id = parts[0]
            off = 1 if parts[1] == "+" else 0
            begin, peak, end = parts[1 + off : 4 + off]
            obs, qual, kind = parts[4 + off : 7 + off]
            rest = parts[7 + off :]
            g = groups.setdefault(event_id, {})
            if kind == "XRA" and len(rest) >= 2:
                m = re.match(r"^([ABCMX])(\d+(?:\.\d+)?)$", rest[1])
                if not m:
                    continue
                start_t, peak_t, end_t = _times(day, [begin, peak, end])
                if peak_t is None:
                    continue
                g.update(
                    start=start_t,
                    peak=peak_t,
                    end=end_t,
                    goes_class=rest[1],
                    goes_score=class_to_score(m.group(1), float(m.group(2))),
                    sxr_fluence_jm2=float(rest[2]) if len(rest) > 2 and re.fullmatch(r"\d\.\dE[+-]\d{2}", rest[2]) else np.nan,
                    noaa_region=float(rest[-1]) if rest[-1].isdigit() and len(rest) >= 4 else np.nan,
                )
            elif kind == "FLA" and rest:
                lat, lon = parse_location(rest[0])
                if np.isfinite(lon):
                    g.update(lat_deg=lat, lon_w_deg=lon)
                    if rest[-1].isdigit() and len(rest) >= 3:
                        g.setdefault("noaa_region", float(rest[-1]))
        for g in groups.values():
            if "goes_score" in g and g.get("peak") is not None:
                rows.append({"catalog": "SWPC-events", "lat_deg": np.nan, "lon_w_deg": np.nan, **g})
    cols = ["catalog", "start", "peak", "end", "goes_class", "goes_score", "lat_deg", "lon_w_deg", "noaa_region",
            "sxr_fluence_jm2"]
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=cols)
    return df[cols].drop_duplicates(subset=["peak", "goes_class"]).reset_index(drop=True)


def read_flare_reference(external: Path) -> pd.DataFrame:
    """Reference GOES flare list: NGDC reports up to mid-2017, SWPC event reports afterwards."""
    ngdc = read_ngdc_xrs_reports(external / "ngdc_goes_xrs_reports")
    swpc = read_swpc_event_reports(external / "swpc_events")
    last_ngdc = ngdc["peak"].max()
    swpc = swpc[swpc["peak"] > last_ngdc]
    ref = pd.concat([ngdc, swpc], ignore_index=True).sort_values("peak").reset_index(drop=True)
    ref["ref_id"] = np.arange(len(ref))
    return ref


_MONTHS = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}


def _spe_time(year: int, text: str, ref: pd.Timestamp | None = None) -> pd.Timestamp | None:
    """'Nov 04/0830', 'Jun 23 2014' or (with a reference onset) '21/0314'."""
    text = text.strip()
    m = re.match(r"([A-Z][a-z]{2})\s*(\d{1,2})[/ ](\d{4})", text)
    if m:
        return pd.Timestamp(year=year, month=_MONTHS[m.group(1)], day=int(m.group(2)), hour=int(m.group(3)[:2]),
                            minute=int(m.group(3)[2:]), tz="UTC")
    m = re.match(r"^(\d{1,2})/(\d{4})$", text)
    if m and ref is not None:
        day = int(m.group(1))
        base = ref if day <= ref.day + 3 else ref - pd.DateOffset(months=1)
        return pd.Timestamp(year=base.year, month=base.month, day=day, hour=int(m.group(2)[:2]),
                            minute=int(m.group(2)[2:]), tz="UTC")
    return None


def read_spe_umbra(path: Path) -> pd.DataFrame:
    """NOAA SWPC 'Solar Proton Events Affecting the Earth Environment' (NASA SDAC mirror)."""
    text = path.read_text(encoding="latin-1")
    year = None
    rows = []
    for chunk in re.split(r"<tr>", text, flags=re.I)[1:]:
        cells = [html.unescape(re.sub(r"<[^>]+>", "", c)).strip() for c in re.split(r"<td[^>]*>", chunk, flags=re.I)[1:]]
        ystr = re.search(r"<strong>(\d{4})</strong>", chunk)
        if ystr:
            year = int(ystr.group(1))
            continue
        if year is None or len(cells) < 9 or not cells[0]:
            continue
        onset = _spe_time(year, cells[0])
        if onset is None:
            continue
        tmax = _spe_time(year, cells[1])
        if tmax is not None and tmax < onset:
            tmax += pd.DateOffset(years=1)
        pfu = float(re.sub(r"[^\d.]", "", cells[2]) or "nan")
        flare_t = _spe_time(year, cells[5], ref=onset)
        if flare_t is not None and flare_t > onset + pd.Timedelta(days=5):
            flare_t -= pd.DateOffset(years=1)
        rows.append(
            {
                "spe_onset": onset,
                "spe_max": tmax,
                "peak_pfu_10mev": pfu,
                "flare_max_time": flare_t,
                "flare_importance": cells[6],
                "flare_location": cells[7],
                "noaa_region": cells[8],
                "source": "SWPC SPE list (SDAC mirror)",
            }
        )
    return pd.DataFrame(rows)


def _flare_from_text(onset: pd.Timestamp, flare: str) -> tuple[pd.Timestamp | None, str]:
    """'X2/2B 09/24 1038' -> (time of flare maximum, importance)."""
    flare = re.sub(r"\s+", " ", str(flare)).strip()
    fm = re.match(r"^(\S+)\s+(\d{1,2})/(\d{1,2})\s+(\d{4})$", flare)
    importance = flare.split(" ")[0] if flare else ""
    if not fm:
        return None, importance
    month = int(fm.group(2))
    year = onset.year - 1 if month > onset.month + 1 else onset.year
    t = pd.Timestamp(year=year, month=month, day=int(fm.group(3)), hour=int(fm.group(4)[:2]),
                     minute=int(fm.group(4)[2:]), tz="UTC")
    return t, importance


def read_spe_swpc_page(path: Path) -> pd.DataFrame:
    """Current SWPC web version of the SPE list (HTML table, one row per event)."""
    text = path.read_text(encoding="utf-8", errors="replace")
    stamp = re.compile(r"^(\d{4}) (\d{2})/(\d{2}) (\d{4})$")
    rows = []
    for chunk in re.findall(r"<tr.*?</tr>", text, flags=re.S | re.I):
        cells = [re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", c))).strip()
                 for c in re.findall(r"<td.*?>(.*?)</td>", chunk, flags=re.S | re.I)]
        if len(cells) < 6 or not stamp.match(cells[0]):
            continue
        y, mo, d, hm = stamp.match(cells[0]).groups()
        onset = pd.Timestamp(year=int(y), month=int(mo), day=int(d), hour=int(hm[:2]), minute=int(hm[2:]), tz="UTC")
        m2 = stamp.match(cells[1])
        tmax = (pd.Timestamp(year=int(m2.group(1)), month=int(m2.group(2)), day=int(m2.group(3)),
                             hour=int(m2.group(4)[:2]), minute=int(m2.group(4)[2:]), tz="UTC") if m2 else None)
        try:
            pfu = float(cells[2].replace(",", ""))
        except ValueError:
            continue
        flare_t, importance = _flare_from_text(onset, cells[5])
        rows.append({"spe_onset": onset, "spe_max": tmax, "peak_pfu_10mev": pfu, "flare_max_time": flare_t,
                     "flare_importance": importance, "flare_location": cells[4], "noaa_region": cells[3],
                     "source": "SWPC SPE list (web version)"})
    return pd.DataFrame(rows)


def read_spe_list(external: Path, start: str, end: str) -> pd.DataFrame:
    """Union of the SDAC mirror (1976-2017) and the current SWPC web version of the SPE list.

    Each version omits some events of the other.  Entries describing the same proton event
    share ``spe_id``; an event keeps more than one row only if the two versions name
    different parent flares.
    """
    umbra = read_spe_umbra(external / "noaa_spe" / "umbra_SEP_index.html")
    page = read_spe_swpc_page(external / "noaa_spe" / "swpc_spe_page.html")
    spe = pd.concat([umbra, page], ignore_index=True)
    spe = spe[(spe["spe_onset"] >= pd.Timestamp(start, tz="UTC")) & (spe["spe_onset"] < pd.Timestamp(end, tz="UTC"))]
    spe = spe.sort_values("spe_onset").reset_index(drop=True)
    # Same event: onsets within 3 h, or within 24 h with an identical peak flux or parent flare
    # (the two versions sometimes quote different onset times for one event).
    same = spe["spe_onset"].diff() <= pd.Timedelta(hours=3)
    near = spe["spe_onset"].diff() <= pd.Timedelta(hours=24)
    same |= near & (spe["peak_pfu_10mev"].diff() == 0)
    same |= near & (spe["flare_max_time"].diff().abs() <= pd.Timedelta(minutes=30))
    spe["spe_id"] = (~same).cumsum().astype(int) - 1
    spe["in_umbra"] = spe.groupby("spe_id")["source"].transform(lambda s: s.str.contains("SDAC").any())
    spe["in_web"] = spe.groupby("spe_id")["source"].transform(lambda s: s.str.contains("web").any())
    spe["peak_pfu_10mev"] = spe.groupby("spe_id")["peak_pfu_10mev"].transform("max")
    spe["flare_key"] = spe["flare_max_time"].dt.round("30min").astype(str)
    spe = spe.sort_values(["spe_id", "flare_max_time"], na_position="last")
    spe = spe.drop_duplicates(["spe_id", "flare_key"]).drop(columns="flare_key")
    # drop flare-less duplicates when the same event has an associated flare in the other version
    has_flare = spe.groupby("spe_id")["flare_max_time"].transform(lambda s: s.notna().any())
    spe = spe[spe["flare_max_time"].notna() | ~has_flare].reset_index(drop=True)
    letter = spe["flare_importance"].str.extract(r"^([ABCMX])(\d+(?:\.\d+)?)")
    spe["flare_score_lower_bound"] = [
        class_to_score(a, float(b)) if isinstance(a, str) else np.nan for a, b in zip(letter[0], letter[1])
    ]
    return spe


def read_srs_regions(folder: Path) -> pd.DataFrame:
    """NOAA Solar Region Summary, part I (regions with sunspots)."""
    rows = []
    for path in sorted(folder.glob("*SRS.txt")):
        day = pd.Timestamp(path.name[:8], tz="UTC")
        in_regions = False
        for raw in path.read_text(encoding="latin-1").splitlines():
            line = raw.strip()
            if line.upper().startswith(("I.  REGIONS WITH SUNSPOTS", "I. REGIONS WITH SUNSPOTS")):
                in_regions = True
                continue
            if in_regions and line.upper().startswith(("IA.", "II.")):
                break
            parts = line.split()
            if not in_regions or len(parts) < 8 or not parts[0].isdigit():
                continue
            lat, lon = parse_location(parts[1])
            try:
                area = float(parts[3])
            except ValueError:
                continue
            mag = " ".join(parts[7:]).lower()
            score = max([v for k, v in (("alpha", 1), ("beta", 2), ("gamma", 3)) if k in mag] + [0])
            if "delta" in mag:
                score = 5 if "gamma" in mag else 4
            number = int(parts[0])
            rows.append(
                {
                    "srs_day": day,
                    "srs_region": number,
                    "srs_lon_w_deg": lon,
                    "srs_area_msh": area,
                    "srs_mcintosh": parts[4],
                    "srs_mag_class_score": float(score) if score else np.nan,
                    "srs_delta_flag": float("delta" in mag),
                }
            )
    return pd.DataFrame(rows)
