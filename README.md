# sep-flare-forecast

Data and code for the paper

> I. I. Davydenkov, V. A. Ozheredov, *Probabilistic forecasting of >10 MeV solar proton
> events from flare, CME and radio observations of solar cycles 23–25: physical
> predictors and verification of tree-ensemble classifiers*, submitted to **Astrophysics**
> (Springer, ISSN 0571-7256).

Every number, table and figure of the paper is regenerated from the files in this
repository by one command, without network access:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash run_all.sh            # ~25 min on a 4-core laptop
```

Results are deterministic for the pinned package versions (all random seeds are fixed).
Reference outputs produced by the authors are committed in `results/` and `figures/`, so a
reader can compare a re-run with them file by file.

## What the pipeline does

| Step | Script | Input | Output |
|---|---|---|---|
| 1 | `scripts/01_build_clean_sample.py` | `data/input/`, `data/external/` | `data/processed/sep_m2_events_clean.csv`, `results/qc_summary.json` |
| 2 | `scripts/02_run_analysis.py` | cleaned sample | `results/*.csv`, `results/*.json` |
| 3 | `scripts/03_make_figures_tables.py` | `results/` | `figures/Fig1–Fig6.{pdf,eps}`, `results/latex_tables/*.tex` |

Step 2 can be run in parts: `python scripts/02_run_analysis.py descriptive cv chrono ablation shap legacy subthreshold`
(`subthreshold` needs the output of `cv`).

### Step 1 – quality control of the event sample

The input table `data/input/upstream_m2_cme_radio_table.csv` (1150 rows) lists HEK flare
records of GOES class ≥ M2.0 (1997–2022) that have an associated SOHO/LASCO CME and either
sweep-frequency radio evidence (ground or WIND/WAVES type II/IV) or RSTN fixed-frequency
coverage. Step 1

1. anchors every record to the NOAA GOES flare list (NGDC GOES XRS reports to mid-2017,
   SWPC *Edited Events* reports afterwards; |Δt_peak| ≤ 20 min, class within a factor 2);
2. removes duplicate records of the same flare produced by different HEK modules
   (1150 → 964 flares);
3. takes the flare position from the NOAA report, or from the NOAA SRS position of the
   source region rotated to the flare time (13.2° day⁻¹); the upstream longitudes of
   Yohkoh records are helioprojective arcseconds and many others are zero placeholders,
   so they are not used;
4. labels a flare as SEP-productive if it is the parent flare of a >10 MeV, ≥10 pfu
   event in the NOAA SWPC Solar Proton Event list (flare maximum within ±30 min, same
   GOES letter class). The union of two versions of the list is used — the SDAC mirror
   (1976–2017) and the SWPC web version (NOAA NCEI page, retrieved 2026-05-09) — because each omits
   events present in the other;
5. recomputes background-subtracted RSTN microwave peaks and integrals, NOAA SRS
   active-region descriptors (matched by region number) and the 24-h flare history of
   the source region.

### Step 2 – experiments

* 10× repeated stratified 5-fold cross-validation of CatBoost, Random Forest and a
  six-predictor logistic baseline; alert thresholds are chosen inside each training
  part by an inner 5-fold CV (max TSS or max HSS), so the reported contingency scores
  are not tuned on the evaluation data;
* chronological test: training on 1997–2010, test on 2011–2022;
* feature-group ablation and SHAP attribution;
* a step-by-step comparison with the protocol of the previous manuscript version
  (`legacy_comparison_*.csv`);
* a check of persistent false alarms against the SINP MSU proton-event catalogue
  (≥1 pfu, 2010–2022; `subthreshold_check.json`).

## Main results (reference run)

| | ROC AUC | AP | BSS | POD | FAR | TSS | HSS |
|---|---|---|---|---|---|---|---|
| CatBoost, max-HSS threshold (10×5 nested CV) | 0.935 ± 0.004 | 0.61 | 0.33 | 0.72 | 0.51 | 0.64 | 0.53 |
| CatBoost, max-TSS threshold | 0.935 ± 0.004 | 0.61 | 0.33 | 0.89 | 0.62 | 0.74 | 0.46 |
| Logistic baseline (6 predictors), max-HSS | 0.924 ± 0.002 | 0.56 | 0.33 | 0.61 | 0.53 | 0.54 | 0.47 |
| CatBoost, train 1997–2010 → test 2011–2022, max-HSS | 0.943 | 0.53 | 0.31 | 0.81 | 0.55 | 0.72 | 0.53 |

## Repository layout

```
data/input/        upstream feature table (one row per HEK flare record)
data/external/     public catalogue copies used in step 1
  ngdc_goes_xrs_reports/   NOAA NCEI GOES XRS flare reports 1997–2017
  swpc_events/             NOAA SWPC Edited Events reports (2017–2022 event days)
  noaa_spe/                NOAA SWPC Solar Proton Event list (SDAC mirror + SWPC web version)
  noaa_srs/                NOAA Solar Region Summaries for event days
  sinp/                    SINP MSU proton-event catalogue, cycles 24–25 (≥1 pfu)
  rstn_event_windows.csv.gz  RSTN 1-min flux densities around each flare
data/processed/    outputs of step 1 (cleaned sample, SPE list with parent flares)
sepforecast/       library code (catalogue parsers, sample building, models, scores)
scripts/           pipeline steps 00–03
results/           numerical results and LaTeX tables
figures/           Fig1–Fig6 (PDF and EPS)
provenance/        upstream scripts that produced data/input from the archives (see below)
```

## Data provenance (stage A, requires network and ~1 GB of caches)

`data/input/upstream_m2_cme_radio_table.csv` was produced by the scripts in
`provenance/` from: the Heliophysics Event Knowledgebase (HEK) flare and CME records;
the CDAW SOHO/LASCO CME catalogue; the CDAW WIND/WAVES type II list; GOES XRS 1–8 Å and
0.5–4 Å fluxes; RSTN 1-s data of Sagamore Hill, San Vito, Learmonth and Palehua (NOAA
NCEI); and NOAA SRS. These scripts are provided for transparency; they depend on local
caches and are not needed to reproduce the paper. `scripts/00_extract_rstn_windows.py`
reduces the RSTN day files (~200 MB) to `data/external/rstn_event_windows.csv.gz`.

Public sources:

* NOAA SWPC Solar Proton Events: SDAC mirror <https://umbra.nascom.nasa.gov/SEP/>; SWPC list hosted by NOAA NCEI <https://www.ngdc.noaa.gov/stp/space-weather/interplanetary-data/solar-proton-events/SEP%20page%20code.html> (source of `swpc_spe_page.html`)
* NOAA NCEI GOES XRS flare reports: <https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-features/solar-flares/x-rays/goes/xrs/>
* SWPC event reports and SRS: <https://www.ngdc.noaa.gov/stp/space-weather/swpc-products/daily_reports/>
* RSTN: <https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-features/solar-radio/rstn-1-second/>
* SOHO/LASCO CME catalogue: <https://cdaw.gsfc.nasa.gov/CME_list/>
* WIND/WAVES type II list: <https://cdaw.gsfc.nasa.gov/CME_list/radio/waves_type2.html>
* HEK: <https://www.lmsal.com/hek/>
* SINP MSU proton-event catalogue (cycle 24): Logachev et al. (2022), <https://doi.org/10.2205/ESDB-SAD-008>

## Column dictionary of the cleaned sample (main fields)

| Column | Meaning |
|---|---|
| `event_key` | identifier of the physical flare (`ref<N>` = row N of the NOAA flare list) |
| `t_start, t_peak, t_end` | GOES flare times (UTC) |
| `goes_class_final`, `goes_score` | GOES class; score = peak 1–8 Å flux / 10⁻⁸ W m⁻² |
| `lon_w_deg_clean`, `lon_source` | heliographic longitude (W > 0) and its source |
| `y_sep` | 1 if parent flare of a NOAA SWPC >10 MeV proton event |
| `spe_onset`, `peak_pfu_10mev` | onset and peak >10 MeV flux of that event |
| `upstream_label_max` | label of the previous manuscript version (24-h heuristic) |
| feature columns | see `sepforecast/models.py::FEATURE_GROUPS` and Table 2 of the paper |

## Licence

Code: MIT. Derived data and results: CC BY 4.0. External catalogue copies remain under
the terms of their providers (U.S. Government works).

## Citation

See `CITATION.cff`. Please cite the paper and the archived release of this repository
(Zenodo DOI to be added on release).
