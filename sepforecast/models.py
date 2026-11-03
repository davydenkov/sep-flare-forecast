"""Classifiers, feature groups, verification scores and resampling procedures."""
from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RSTN_FREQS = ["245", "410", "610", "1415", "2695", "4995", "8800", "15400"]

FEATURE_GROUPS: dict[str, list[str]] = {
    "flare_sxr": [
        "log10_peak_flux_wm2", "log10_sxr_fluence_jm2", "rise_time_min", "duration_min",
        "xs_ratio_10m", "goes_ratio_derivative", "sxr_peak_flux_derivative_max",
    ],
    "connection": ["lon_w_deg_clean", "connection_cos45"],
    "cme": ["cme_speed_kms", "cme_width_deg", "cme_shock_proxy"],
    "radio_sweep": [
        "type_ii_ground", "type_ii_log_power", "type_iv_present", "wind_type2_inverse_delay_clean",
        "wind_type2_log_power_clean", "wind_radio_combined_score_clean", "rstn_wind_bridge", "radio_coherence",
    ],
    "radio_rstn": [f"rstn_{f}_log_peak" for f in RSTN_FREQS] + ["rstn_log_integral_high", "rstn_log_integral_low"],
    "active_region": [
        "srs_area_msh", "srs_mag_class_score", "srs_delta_flag", "ar_energy_proxy", "ar_complexity_proxy",
        "flare_history_24h",
    ],
}
ALL_FEATURES = [f for g in FEATURE_GROUPS.values() for f in g]
BASELINE_FEATURES = ["log10_peak_flux_wm2", "log10_sxr_fluence_jm2", "cme_speed_kms", "cme_width_deg",
                     "connection_cos45", "type_ii_ground"]
GROUP_LABELS = {
    "flare_sxr": "Flare SXR", "connection": "Longitude/connection", "cme": "CME kinematics",
    "radio_sweep": "Type II/IV radio", "radio_rstn": "RSTN microwaves", "active_region": "Active region",
}

# ---------------------------------------------------------------- models


class Model:
    """Common interface: class-weighted fit, prior-corrected probabilities."""

    def __init__(self, kind: str, features: list[str], seed: int = 0):
        self.kind, self.features, self.seed = kind, list(features), seed

    def _impute(self, x: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Training-fold median imputation plus missing-value indicators (for RF and LR only)."""
        if fit:
            self.median_ = x.median(numeric_only=True).fillna(0.0)
            self.flag_cols_ = [c for c in x.columns if x[c].isna().any()]
        flags = x[self.flag_cols_].isna().astype(float).add_suffix("_missing")
        return pd.concat([x.fillna(self.median_).fillna(0.0), flags], axis=1)

    def fit(self, data: pd.DataFrame, y: np.ndarray) -> "Model":
        x = data[self.features].astype(float)
        n1, n0 = int(y.sum()), int(len(y) - y.sum())
        self.w_pos_ = n0 / max(n1, 1)
        if self.kind == "catboost":
            self.est_ = CatBoostClassifier(
                loss_function="Logloss", iterations=800, learning_rate=0.03, depth=3, l2_leaf_reg=30.0,
                rsm=0.5, class_weights=[1.0, self.w_pos_], random_seed=self.seed, verbose=False,
                allow_writing_files=False, thread_count=4,
            )
            self.est_.fit(x, y)
        elif self.kind == "rf":
            self.est_ = RandomForestClassifier(
                n_estimators=600, max_depth=8, min_samples_leaf=5, max_features="sqrt",
                class_weight="balanced_subsample", random_state=self.seed, n_jobs=-1,
            )
            self.est_.fit(self._impute(x, fit=True), y)
        elif self.kind == "logreg":
            self.est_ = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))
            self.est_.fit(self._impute(x, fit=True), y)
        else:
            raise ValueError(self.kind)
        return self

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        x = data[self.features].astype(float)
        if self.kind != "catboost":
            x = self._impute(x, fit=False)
        p = np.clip(self.est_.predict_proba(x)[:, 1], 1e-6, 1 - 1e-6)
        # Undo the class weighting: odds_true = odds_weighted / w_pos (prior correction).
        odds = p / (1 - p) / self.w_pos_
        return odds / (1 + odds)


# ---------------------------------------------------------------- scores


def contingency(y: np.ndarray, yhat: np.ndarray) -> dict[str, float]:
    tp = int(((yhat == 1) & (y == 1)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum())
    tn = int(((yhat == 0) & (y == 0)).sum())
    pod = tp / max(tp + fn, 1)
    far = fp / max(tp + fp, 1)
    pofd = fp / max(fp + tn, 1)
    den = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn, "POD": pod, "FAR": far, "POFD": pofd,
        "TSS": pod - pofd, "HSS": 2 * (tp * tn - fp * fn) / den if den else 0.0,
        "CSI": tp / max(tp + fp + fn, 1),
    }


def prob_scores(y: np.ndarray, p: np.ndarray, base_rate: float | None = None) -> dict[str, float]:
    base = y.mean() if base_rate is None else base_rate
    bs = float(np.mean((p - y) ** 2))
    bs_ref = float(np.mean((base - y) ** 2))
    return {"ROC_AUC": float(roc_auc_score(y, p)), "PR_AP": float(average_precision_score(y, p)),
            "BS": bs, "BSS": 1 - bs / bs_ref}


def best_threshold(y: np.ndarray, p: np.ndarray, score: str) -> float:
    grid = np.unique(np.quantile(p, np.linspace(0.5, 0.999, 400)))
    vals = [contingency(y, (p >= t).astype(int))[score] for t in grid]
    vals = np.asarray(vals)
    best = np.flatnonzero(vals >= vals.max() - 1e-12)
    return float(np.median(grid[best]))  # centre of the optimal plateau


# ---------------------------------------------------------------- resampling


def nested_cv(data: pd.DataFrame, y: np.ndarray, kind: str, features: list[str], seed: int,
              scores: tuple[str, ...] = ("TSS", "HSS"), n_outer: int = 5, n_inner: int = 5,
              inner: bool = True) -> dict:
    """Outer stratified K-fold OOF probabilities; thresholds chosen on inner OOF of each training part."""
    outer = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=seed)
    prob = np.zeros(len(y))
    yhat = {s: np.zeros(len(y), dtype=int) for s in scores}
    thr = {s: [] for s in scores}
    fold_id = np.zeros(len(y), dtype=int)
    for k, (tr, te) in enumerate(outer.split(data, y)):
        dtr, ytr = data.iloc[tr], y[tr]
        model = Model(kind, features, seed=seed * 100 + k).fit(dtr, ytr)
        prob[te] = model.predict_proba(data.iloc[te])
        fold_id[te] = k
        if not inner:
            continue
        pin = np.zeros(len(tr))
        for kk, (a, b) in enumerate(StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed + 1000).split(dtr, ytr)):
            pin[b] = Model(kind, features, seed=seed * 100 + 10 * k + kk).fit(dtr.iloc[a], ytr[a]).predict_proba(dtr.iloc[b])
        for s in scores:
            t = best_threshold(ytr, pin, s)
            thr[s].append(t)
            yhat[s][te] = (prob[te] >= t).astype(int)
    return {"prob": prob, "yhat": yhat, "thr": thr, "fold": fold_id}


def bootstrap_ci(y: np.ndarray, p: np.ndarray, yhat: np.ndarray, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() == 0:
            continue
        r = contingency(y[i], yhat[i])
        r.update(prob_scores(y[i], p[i]))
        rows.append(r)
    b = pd.DataFrame(rows)
    return b.quantile([0.025, 0.975]).T.rename(columns={0.025: "lo95", 0.975: "hi95"})
