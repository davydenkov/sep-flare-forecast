"""Quality control and construction of the cleaned event sample.

Input: the upstream flare table (one row per HEK flare record; M2.0+ flares
1997-2022 with an associated CME and either sweep-frequency radio evidence or
RSTN fixed-frequency coverage).  The upstream table contains (i) the same flare
reported by several HEK feature-recognition modules, (ii) longitudes that are
helioprojective arcseconds (Yohkoh records) or zero placeholders, and (iii) SEP
labels assigned by a "strongest flare within 24 h before onset" heuristic.

This module anchors each record to the NOAA GOES flare list, removes
duplicates, takes the flare location from NOAA (event list or SRS region
position rotated to the flare time), re-labels SEP events from the flare
associations published in the NOAA SWPC Solar Proton Event list, and
recomputes the radio and active-region descriptors.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import catalogs

START, END = "1997-01-01", "2023-01-01"
MATCH_TOL_MIN = 20.0          # HEK peak vs NOAA peak
SPE_FLARE_TOL_MIN = 30.0      # SPE-list flare maximum vs NOAA peak
SYNODIC_ROTATION_DEG_PER_DAY = 13.2
RSTN_FREQS = ["245", "410", "610", "1415", "2695", "4995", "8800", "15400"]
RSTN_HIGH = ["1415", "2695", "4995", "8800", "15400"]
SOURCE_PRIORITY = {"FL_SWPC": 0, "FL_SECstandard": 1, "FL_SSWLatestEvents": 2, "FL_YOHKOH": 3, "other": 4}


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def load_upstream(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    for col in ("event_starttime", "event_peaktime", "event_endtime"):
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce", format="mixed")
    df["hek_source"] = df["hek_event_id"].str.extract(r"/(FL_[A-Za-z]+)")[0].fillna("other")
    return df


def match_to_reference(df: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Attach the nearest NOAA flare (|dt| <= 20 min, class within a factor of 2)."""
    out = df.copy()
    ref_t = ref["peak"].astype("int64").to_numpy() / 6e10
    ref_s = ref["goes_score"].to_numpy()
    ids, dts = [], []
    for peak, score in zip(out["event_peaktime"], _num(out, "goes_class_score")):
        dt = ref_t - peak.value / 6e10
        ok = (np.abs(dt) <= MATCH_TOL_MIN) & (np.abs(np.log10(ref_s / score)) <= np.log10(2.0))
        if not ok.any():
            ids.append(np.nan)
            dts.append(np.nan)
            continue
        cost = np.where(ok, np.abs(dt) / MATCH_TOL_MIN + np.abs(np.log10(ref_s / score)), np.inf)
        k = int(np.argmin(cost))
        ids.append(int(ref["ref_id"].iloc[k]))
        dts.append(float(dt[k]))
    out["ref_id"] = ids
    out["ref_match_dt_min"] = dts
    return out


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """One row per physical flare; unmatched records are clustered by time (15 min) and class."""
    work = df.sort_values("event_peaktime").copy()
    key = []
    last_t, last_s, cluster = None, None, -1
    for rid, t, s in zip(work["ref_id"], work["event_peaktime"], _num(work, "goes_class_score")):
        if np.isfinite(rid):
            key.append(f"ref{int(rid)}")
            continue
        if last_t is None or (t - last_t) > pd.Timedelta(minutes=15) or abs(np.log10(s / last_s)) > np.log10(2.0):
            cluster += 1
        key.append(f"hek{cluster}")
        last_t, last_s = t, s
    work["event_key"] = key
    work["n_records"] = work.groupby("event_key")["event_key"].transform("size")
    feature_cols = [c for c in work.columns if c.startswith(("rstn_", "wind_", "hek_cme", "xs_", "goes_ratio", "sxr_"))]
    work["_completeness"] = work[feature_cols].notna().sum(axis=1)
    work["_priority"] = work["hek_source"].map(SOURCE_PRIORITY).fillna(9)
    work["upstream_label_max"] = work.groupby("event_key")["is_sep_event"].transform("max")
    rep = (
        work.sort_values(["event_key", "_completeness", "_priority"], ascending=[True, False, True])
        .drop_duplicates("event_key")
        .drop(columns=["_completeness", "_priority"])
    )
    return rep.sort_values("event_peaktime").reset_index(drop=True)


def attach_reference(df: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    cols = ["ref_id", "catalog", "start", "peak", "end", "goes_class", "goes_score", "lat_deg", "lon_w_deg",
            "noaa_region", "sxr_fluence_jm2"]
    r = ref[cols].rename(columns={c: f"noaa_{c}" for c in cols if c != "ref_id"})
    out = df.merge(r, on="ref_id", how="left")
    # Times and class: NOAA where matched, HEK otherwise.
    out["t_start"] = out["noaa_start"].fillna(out["event_starttime"])
    out["t_peak"] = out["noaa_peak"].fillna(out["event_peaktime"])
    out["t_end"] = out["noaa_end"].fillna(out["event_endtime"])
    out["goes_score"] = out["noaa_goes_score"].fillna(_num(out, "goes_class_score"))
    out["goes_class_final"] = out["noaa_goes_class"].fillna(out["flare_class"])
    return out


def assign_location(df: pd.DataFrame, srs: pd.DataFrame) -> pd.DataFrame:
    """Flare longitude: NOAA flare report > SRS region rotated to t_peak > HEK (degrees only)."""
    out = df.copy()
    out["lon_w_deg_clean"] = out["noaa_lon_w_deg"]
    out["lat_deg_clean"] = out["noaa_lat_deg"]
    out["lon_source"] = np.where(out["noaa_lon_w_deg"].notna(), "NOAA flare report", "")
    srs = srs.copy()
    srs["region4"] = srs["srs_region"] % 10000
    by_day = {d: g for d, g in srs.groupby("srs_day")}
    for col in ("srs_area_msh", "srs_mag_class_score", "srs_delta_flag"):
        out[col] = np.nan
    out["srs_mcintosh"] = ""
    for i, row in out.iterrows():
        reg = row["noaa_noaa_region"]
        if not np.isfinite(reg):
            continue
        day = row["t_peak"].normalize()
        hit = None
        for shift in (0, -1, 1):
            g = by_day.get(day + pd.Timedelta(days=shift))
            if g is None:
                continue
            m = g[g["region4"] == int(reg) % 10000]
            if not m.empty:
                hit = (m.iloc[0], day + pd.Timedelta(days=shift))
                break
        if hit is None:
            continue
        s, sday = hit
        out.at[i, "srs_area_msh"] = s["srs_area_msh"]
        out.at[i, "srs_mag_class_score"] = s["srs_mag_class_score"]
        out.at[i, "srs_delta_flag"] = s["srs_delta_flag"]
        out.at[i, "srs_mcintosh"] = s["srs_mcintosh"]
        if not np.isfinite(out.at[i, "lon_w_deg_clean"]) and np.isfinite(s["srs_lon_w_deg"]):
            days = (row["t_peak"] - sday).total_seconds() / 86400.0
            out.at[i, "lon_w_deg_clean"] = s["srs_lon_w_deg"] + SYNODIC_ROTATION_DEG_PER_DAY * days
            out.at[i, "lon_source"] = "SRS region rotated"
    hek_ok = out["hek_source"].isin(["FL_SSWLatestEvents", "FL_SWPC"]) & _num(out, "lon_w_deg").abs().between(0.5, 90.0)
    fill = out["lon_w_deg_clean"].isna() & hek_ok
    out.loc[fill, "lon_w_deg_clean"] = _num(out, "lon_w_deg")[fill]
    out.loc[fill, "lon_source"] = "HEK (degrees)"
    out.loc[out["lon_w_deg_clean"].isna(), "lon_source"] = "unknown"
    return out


def assign_labels(df: pd.DataFrame, spe: pd.DataFrame, ref: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Label = the flare is the parent flare of a >10 MeV SPE in the NOAA SWPC list."""
    ref_t = ref["peak"].astype("int64").to_numpy() / 6e10
    spe = spe.copy()
    parent, dtm = [], []
    for t, imp in zip(spe["flare_max_time"], spe["flare_importance"]):
        if pd.isna(t):
            parent.append(np.nan)
            dtm.append(np.nan)
            continue
        dt = ref_t - t.value / 6e10
        ok = np.abs(dt) <= SPE_FLARE_TOL_MIN
        letter = str(imp)[:1]
        if letter in "CMX":
            ok &= ref["goes_class"].str[0].to_numpy() == letter
        if not ok.any():
            parent.append(np.nan)
            dtm.append(np.nan)
            continue
        k = int(np.argmin(np.where(ok, np.abs(dt), np.inf)))
        parent.append(int(ref["ref_id"].iloc[k]))
        dtm.append(float(dt[k]))
    spe["parent_ref_id"] = parent
    spe["parent_match_dt_min"] = dtm
    spe = spe.merge(ref[["ref_id", "goes_class", "goes_score"]].rename(columns={"ref_id": "parent_ref_id",
                    "goes_class": "parent_goes_class", "goes_score": "parent_goes_score"}), on="parent_ref_id", how="left")
    agg = (
        spe.dropna(subset=["parent_ref_id"]).groupby("parent_ref_id")
        .agg(spe_onset=("spe_onset", "min"), peak_pfu_10mev=("peak_pfu_10mev", "max"), n_spe=("spe_id", "nunique"))
        .reset_index().rename(columns={"parent_ref_id": "ref_id"})
    )
    out = df.merge(agg, on="ref_id", how="left")
    out["y_sep"] = out["peak_pfu_10mev"].notna().astype(int)
    spe["parent_in_sample"] = spe["parent_ref_id"].isin(out.loc[out["y_sep"] == 1, "ref_id"])
    spe["in_sample"] = spe.groupby("spe_id")["parent_in_sample"].transform("any")
    return out, spe


def rstn_features(df: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    """Background-subtracted RSTN peak excess (sfu) and time integral (sfu min) per frequency."""
    out = df.copy()
    groups = {k: g for k, g in windows.groupby("hek_event_id")}
    rows = []
    for _, row in out.iterrows():
        rec = {"rstn_coverage": 0.0}
        g = groups.get(row["hek_event_id"])
        t0 = row["t_start"].tz_convert(None)
        t1 = row["t_end"].tz_convert(None) if pd.notna(row["t_end"]) and row["t_end"] > row["t_start"] else t0 + pd.Timedelta(minutes=90)
        for f in RSTN_FREQS:
            rec[f"rstn_{f}_peak_sfu"] = np.nan
            rec[f"rstn_{f}_integral_sfu_min"] = np.nan
            rec[f"rstn_{f}_detected"] = np.nan
        if g is not None:
            t = g["DateTime"]
            during = (t >= t0) & (t <= t1 + pd.Timedelta(minutes=10))
            pre = (t >= t0 - pd.Timedelta(minutes=30)) & (t < t0)
            for f in RSTN_FREQS:
                col = f"Radio_{f}MHz"
                s = pd.to_numeric(g[col], errors="coerce").where(lambda x: x > 0)
                sd = s[during].dropna()
                if len(sd) < 3:
                    continue
                sp = s[pre].dropna()
                if len(sp) >= 5:
                    bg = float(sp.median())
                    sig = float(1.4826 * (sp - bg).abs().median())
                else:
                    bg = float(np.nanpercentile(s[during | pre].dropna(), 10))
                    sig = float(1.4826 * (sd - bg).abs().median())
                excess = sd - bg
                peak = float(excess.max())
                rec[f"rstn_{f}_peak_sfu"] = max(peak, 0.0)
                rec[f"rstn_{f}_integral_sfu_min"] = float(excess.clip(lower=0).sum())
                rec[f"rstn_{f}_detected"] = float(peak > max(5.0 * sig, 10.0))
                rec["rstn_coverage"] = 1.0
        rows.append(rec)
    feats = pd.DataFrame(rows, index=out.index)
    out = pd.concat([out, feats], axis=1)
    hi = [f"rstn_{f}_integral_sfu_min" for f in RSTN_HIGH]
    lo = [f"rstn_{f}_integral_sfu_min" for f in ("245", "410", "610")]
    out["rstn_log_integral_high"] = np.log10(1 + out[hi].sum(axis=1, min_count=1))
    out["rstn_log_integral_low"] = np.log10(1 + out[lo].sum(axis=1, min_count=1))
    out["rstn_high_detected"] = out[[f"rstn_{f}_detected" for f in RSTN_HIGH]].max(axis=1)
    return out


def flare_history(df: pd.DataFrame, ref: pd.DataFrame) -> pd.Series:
    """Sum of GOES scores of C1+ flares from the same NOAA region in the preceding 24 h."""
    r = ref[(ref["goes_score"] >= 100) & ref["noaa_region"].notna()].copy()
    r["region4"] = (r["noaa_region"] % 10000).astype(int)
    by_region = {k: g for k, g in r.groupby("region4")}
    vals = []
    for reg, t in zip(df["noaa_noaa_region"], df["t_peak"]):
        if not np.isfinite(reg):
            vals.append(np.nan)
            continue
        g = by_region.get(int(reg) % 10000)
        if g is None:
            vals.append(0.0)
            continue
        m = (g["peak"] < t - pd.Timedelta(minutes=5)) & (g["peak"] >= t - pd.Timedelta(hours=24))
        vals.append(float(g.loc[m, "goes_score"].sum()))
    return pd.Series(vals, index=df.index)


def physical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derived physical descriptors (all from measured or catalogued quantities)."""
    out = df.copy()
    lon = out["lon_w_deg_clean"]
    out["log10_peak_flux_wm2"] = np.log10(out["goes_score"] * 1e-8)
    fluence = out["noaa_sxr_fluence_jm2"]
    proxy = out["goes_score"] * 1e-8 * (out["t_end"] - out["t_start"]).dt.total_seconds()
    out["sxr_fluence_source"] = np.where(fluence.notna(), "NOAA integrated flux", "peak x duration")
    out["log10_sxr_fluence_jm2"] = np.log10(fluence.fillna(proxy))
    out["rise_time_min"] = (out["t_peak"] - out["t_start"]).dt.total_seconds() / 60
    out["duration_min"] = (out["t_end"] - out["t_start"]).dt.total_seconds() / 60
    out["connection_cos45"] = np.cos(np.deg2rad(lon - 45.0))
    out["cme_speed_kms"] = _num(out, "hek_cme_speed_kms").where(lambda v: v > 0)
    out["cme_width_deg"] = _num(out, "hek_cme_width_deg").where(lambda v: v > 0)
    out["cme_shock_proxy"] = np.log1p(out["cme_speed_kms"]) * np.log1p(out["cme_width_deg"])
    out["cme_delay_min"] = _num(out, "hek_cme_time_delta_min")
    area = out.get("srs_area_msh", pd.Series(np.nan, index=out.index))
    out["ar_energy_proxy"] = area * np.log1p(out["goes_score"])
    out["ar_complexity_proxy"] = (out["srs_mag_class_score"] + 2 * out["srs_delta_flag"] + 0.01 * area) * np.log1p(out["goes_score"])

    # WIND/WAVES type II: suppress amplitudes for unmatched (noise) rows; orphan test needs an RSTN bridge.
    q = _num(out, "wind_match_quality_score").fillna(0)
    noise = _num(out, "is_wind_noise_or_none").fillna(1).gt(0) | q.le(0.05)
    weak = _num(out, "is_wind_weak").fillna(0).gt(0)
    for col in ("wind_type2_inverse_delay", "wind_type2_log_power", "wind_radio_combined_score"):
        out[f"{col}_clean"] = _num(out, col).fillna(0).where(~noise, 0.0)
    bridge = out["rstn_high_detected"].fillna(0).gt(0)
    wind_ev = _num(out, "wind_type2_present").fillna(0).gt(0) | _num(out, "wind_type2_count").fillna(0).gt(0)
    total_int = 10 ** out["rstn_log_integral_high"].fillna(0) - 1
    out["wind_quality_unit"] = (q / 2.0).clip(0, 1)
    out["rstn_wind_bridge"] = np.log1p(total_int) * out["wind_quality_unit"] * bridge.astype(float)
    out["wind_orphan"] = (wind_ev & ~bridge & weak).astype(float)
    out["radio_coherence"] = np.where(out["wind_orphan"] > 0, 0.0,
                                      out["wind_radio_combined_score_clean"] + 0.25 * np.log1p(total_int))
    out["type_ii_ground"] = _num(out, "type_ii_flag").fillna(0)
    out["type_ii_log_power"] = _num(out, "radio_type_ii_log_power").fillna(0)
    out["type_iv_present"] = _num(out, "radio_type_iv_present").fillna(0)
    for f in RSTN_FREQS:
        out[f"rstn_{f}_log_peak"] = np.log10(1 + out[f"rstn_{f}_peak_sfu"])
    return out


def build(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    ext = root / "data" / "external"
    ref = catalogs.read_flare_reference(ext)
    spe = catalogs.read_spe_list(ext, START, END)
    srs = catalogs.read_srs_regions(ext / "noaa_srs")
    windows = pd.read_csv(ext / "rstn_event_windows.csv.gz", parse_dates=["DateTime"])
    up = load_upstream(root / "data" / "input" / "upstream_m2_cme_radio_table.csv")
    qc: dict = {"upstream_rows": len(up), "upstream_sep_labels": int(up["is_sep_event"].sum())}
    qc["upstream_lon_abs_gt90"] = int((_num(up, "lon_w_deg").abs() > 90).sum())
    qc["upstream_lon_zero"] = int((_num(up, "lon_w_deg") == 0).sum())

    m = match_to_reference(up, ref)
    qc["records_matched_to_noaa"] = int(m["ref_id"].notna().sum())
    d = deduplicate(m)
    qc["unique_flares"] = len(d)
    qc["duplicate_records_removed"] = len(up) - len(d)
    d = attach_reference(d, ref)
    below = d["goes_score"] < 2000
    qc["removed_below_M2_in_noaa_list"] = int(below.sum())
    d = d[~below].reset_index(drop=True)
    d = assign_location(d, srs)
    d, spe = assign_labels(d, spe, ref)
    d = rstn_features(d, windows)
    d["flare_history_24h"] = flare_history(d, ref)
    d = physical_features(d)
    d["year"] = d["t_peak"].dt.year + (d["t_peak"].dt.dayofyear - 1) / 365.25
    d["cycle"] = np.where(d["t_peak"] < pd.Timestamp("2008-12-01", tz="UTC"), 23,
                          np.where(d["t_peak"] < pd.Timestamp("2019-12-01", tz="UTC"), 24, 25))
    qc.update(
        events=len(d),
        sep_events=int(d["y_sep"].sum()),
        upstream_heuristic_positive_after_dedup=int(d["upstream_label_max"].sum()),
        label_agreement=int(((d["upstream_label_max"] > 0) == (d["y_sep"] > 0)).sum()),
        spe_events_total=int(spe["spe_id"].nunique()),
        spe_events_only_in_sdac_mirror=int(spe.groupby("spe_id")["in_web"].first().eq(False).sum()),
        spe_events_only_in_web_version_to_2017=int(
            (spe.groupby("spe_id")["in_umbra"].first().eq(False)
             & spe.groupby("spe_id")["spe_onset"].first().lt(pd.Timestamp("2017-10-01", tz="UTC"))).sum()),
        spe_events_with_flare=int(spe.groupby("spe_id")["flare_max_time"].apply(lambda s: s.notna().any()).sum()),
        spe_events_parent_in_noaa_list=int(spe.groupby("spe_id")["parent_ref_id"].apply(lambda s: s.notna().any()).sum()),
        spe_events_parent_m2=int(spe.groupby("spe_id")["parent_goes_score"].max().ge(2000).sum()),
        spe_events_parent_in_sample=int(spe.groupby("spe_id")["in_sample"].first().sum()),
        lon_source_counts=d["lon_source"].value_counts().to_dict(),
        srs_matched=int(d["srs_area_msh"].notna().sum()),
        rstn_coverage=int(d["rstn_coverage"].sum()),
        fluence_from_noaa=int((d["sxr_fluence_source"] == "NOAA integrated flux").sum()),
    )
    return d, spe, qc
