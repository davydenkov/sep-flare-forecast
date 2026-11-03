"""Figures (Fig1-Fig6, EPS + PDF) and LaTeX tables for the manuscript, from results/."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sepforecast import models as M  # noqa: E402

RES, FIG = ROOT / "results", ROOT / "figures"
TAB = ROOT / "results" / "latex_tables"
MM = 1 / 25.4
WIDTH = 119 * MM  # Springer small-format journal figure width
STYLE = {  # colour (validated categorical slots 1-3) + line style for grey-scale print
    "catboost": dict(color="#2a78d6", ls="-", label="CatBoost"),
    "rf": dict(color="#eb6834", ls="--", label="Random Forest"),
    "logreg": dict(color="#1baf7a", ls=":", label="Logistic baseline"),
}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#d9d8d2"
PRETTY = {
    "wind_radio_combined_score_clean": r"WIND type II/IV score $X_W$",
    "wind_type2_log_power_clean": "WIND type II log power",
    "wind_type2_inverse_delay_clean": "WIND type II 1/delay",
    "connection_cos45": r"$C_{\rm W45}$",
    "type_ii_ground": "type II flag (ground)",
    "type_ii_log_power": "type II log power (ground)",
    "lon_w_deg_clean": r"longitude $\lambda$",
    "rise_time_min": "SXR rise time",
    "log10_peak_flux_wm2": r"log $F_{\rm peak}$",
    "log10_sxr_fluence_jm2": r"log $\Phi_{\rm SXR}$",
    "cme_speed_kms": r"$v_{\rm CME}$",
    "cme_width_deg": r"$w_{\rm CME}$",
    "cme_shock_proxy": r"$S_{\rm CME}$",
    "rstn_log_integral_high": r"RSTN $\sum I_\nu$ ($\geq$1.4 GHz)",
    "rstn_log_integral_low": r"RSTN $\sum I_\nu$ ($\leq$0.6 GHz)",
    "xs_ratio_10m": r"$R_{\rm XS,10}$",
    "rstn_wind_bridge": r"$B_{\rm RSTN/WIND}$",
    "radio_coherence": r"$R_{\rm coh}$",
    "flare_history_24h": r"$H_{24}$",
    "srs_area_msh": r"$A_{\rm AR}$",
    **{f"rstn_{f}_log_peak": f"RSTN {int(f) / 1000:g} GHz peak" for f in ["245", "410", "610", "1415", "2695", "4995", "8800", "15400"]},
}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.4, "lines.linewidth": 1.4, "savefig.dpi": 600,
    "ps.fonttype": 42, "pdf.fonttype": 42, "legend.frameon": False,
})


def save(fig: plt.Figure, name: str) -> None:
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.eps", bbox_inches="tight")
    plt.close(fig)


def panel(ax: plt.Axes, letter: str) -> None:
    ax.text(-0.16, 1.04, f"({letter})", transform=ax.transAxes, fontsize=8, fontweight="bold", color=INK)


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    d = pd.read_csv(ROOT / "data/processed/sep_m2_events_clean.csv", low_memory=False)
    p = pd.read_csv(RES / "oof_predictions_repeat0.csv")
    return d, p


# ------------------------------------------------------------------ figures


def fig1(d: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 0.42 * WIDTH))
    neg, pos = d[d.y_sep == 0], d[d.y_sep == 1]
    ax = axes[0]
    bins = np.arange(-90, 100, 15)
    ax.hist(neg.lon_w_deg_clean.dropna(), bins=bins, density=True, histtype="step", color=INK, lw=1.2, zorder=2,
            label=f"no SEP (n={neg.lon_w_deg_clean.notna().sum()})")
    ax.hist(pos.lon_w_deg_clean.dropna(), bins=bins, density=True, histtype="stepfilled", facecolor="#f8cdb9",
            hatch="////", edgecolor="#eb6834", lw=1.2, label=f"SEP (n={pos.lon_w_deg_clean.notna().sum()})", zorder=1)
    ax.set_xlabel("Flare longitude (deg, W > 0)")
    ax.set_ylabel("Probability density")
    ax.set_xticks([-90, -45, 0, 45, 90])
    ax.legend(loc="upper left")
    panel(ax, "a")
    ax = axes[1]
    ax.scatter(neg.log10_peak_flux_wm2, neg.cme_speed_kms, s=4, color="#b5b4ad", lw=0, label="no SEP")
    ax.scatter(pos.log10_peak_flux_wm2, pos.cme_speed_kms, s=16, marker="o", facecolor="#eb6834",
               edgecolor="white", lw=0.5, label="SEP")
    ax.set_yscale("log")
    ax.set_xlabel(r"log$_{10}$ GOES peak flux (W m$^{-2}$)")
    ax.set_ylabel(r"CME speed (km s$^{-1}$)")
    ax.legend(loc="lower right")
    panel(ax, "b")
    fig.tight_layout(w_pad=1.5)
    save(fig, "Fig1")


def fig2(d: pd.DataFrame, p: pd.DataFrame) -> None:
    y = p.y_sep.to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH * 1.0, 0.38 * WIDTH))
    for kind, st in STYLE.items():
        prob = p[f"{kind}_prob"].to_numpy()
        fpr, tpr, _ = roc_curve(y, prob)
        axes[0].plot(fpr, tpr, color=st["color"], ls=st["ls"], label=st["label"])
        prec, rec, _ = precision_recall_curve(y, prob)
        axes[1].plot(rec, prec, color=st["color"], ls=st["ls"], label=st["label"])
        bins = np.array([0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0])
        idx = np.digitize(prob, bins) - 1
        fx, fy = [], []
        for b in range(len(bins) - 1):
            m = idx == b
            if m.sum() >= 8:
                fx.append(prob[m].mean())
                fy.append(y[m].mean())
        axes[2].plot(fx, fy, color=st["color"], ls=st["ls"], marker="o", ms=3, label=st["label"])
    axes[0].plot([0, 1], [0, 1], color=GRID, lw=0.8)
    axes[0].set_xlabel("False positive rate (POFD)")
    axes[0].set_ylabel("Hit rate (POD)")
    axes[1].axhline(y.mean(), color=GRID, lw=0.8)
    axes[1].set_xlabel("Recall (POD)")
    axes[1].set_ylabel("Precision (1 - FAR)")
    axes[2].plot([0, 1], [0, 1], color=GRID, lw=0.8)
    axes[2].set_xlabel("Forecast probability")
    axes[2].set_ylabel("Observed frequency")
    axes[0].legend(loc="lower right", handlelength=2.2)
    for ax, l in zip(axes, "abc"):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_aspect("equal")
        panel(ax, l)
    fig.tight_layout(w_pad=0.8)
    save(fig, "Fig2")


def fig3(p: pd.DataFrame, cv: pd.DataFrame) -> None:
    y = p.y_sep.to_numpy()
    prob = p.catboost_prob.to_numpy()
    grid = np.linspace(0.005, 0.6, 240)
    rows = [M.contingency(y, (prob >= t).astype(int)) for t in grid]
    r = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(WIDTH * 0.62, 0.42 * WIDTH))
    for col, ls, c in (("POD", "-", "#2a78d6"), ("FAR", "--", "#eb6834"), ("TSS", "-.", INK), ("HSS", ":", "#1baf7a")):
        ax.plot(grid, r[col], ls=ls, color=c, label=col)
    for rule, ls in (("max TSS (inner CV)", (0, (1, 1))), ("max HSS (inner CV)", (0, (4, 2)))):
        t = cv[(cv.model == "catboost") & (cv.threshold_rule == rule)].threshold_mean.mean()
        ax.axvline(t, color=MUTED, lw=0.8, ls=ls)
        ax.text(t, 1.02, rule.split(" (")[0].replace("max ", "") + "-opt.", fontsize=6.5, color=MUTED,
                ha="center", va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("Probability threshold (CatBoost, prior-corrected)")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    save(fig, "Fig3")


def fig4() -> None:
    a = pd.read_csv(RES / "ablation_summary.csv")
    full = a.loc[a.variant == "all features"].iloc[0]
    shap = pd.read_csv(RES / "shap_importance.csv")
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 0.5 * WIDTH), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    groups = list(M.FEATURE_GROUPS)
    drop = [a.loc[a.variant == f"without {g}", "ROC_AUC_mean"].iloc[0] - full.ROC_AUC_mean for g in groups]
    only = [a.loc[a.variant == f"only {g}", "ROC_AUC_mean"].iloc[0] for g in groups]
    order = np.argsort(only)
    ypos = np.arange(len(groups))
    ax.barh(ypos, np.array(only)[order] - 0.5, left=0.5, height=0.6, color="#2a78d6")
    for i, k in enumerate(order):
        ax.text(0.965, i, f"{only[k]:.2f} / {drop[k]:+.3f}", va="center", fontsize=6.3, color=INK)
    ax.text(0.965, len(groups) - 0.35, "only / without", fontsize=6.3, color=MUTED)
    ax.axvline(full.ROC_AUC_mean, color=INK, lw=0.8, ls="--")
    ax.set_yticks(ypos, [M.GROUP_LABELS[groups[k]] for k in order])
    ax.set_xlim(0.5, 1.2)
    ax.set_xticks([0.5, 0.7, 0.9])
    ax.set_xlabel("ROC AUC, single group")
    ax.grid(axis="y", visible=False)
    panel(ax, "a")
    ax = axes[1]
    top = shap.head(14).iloc[::-1]
    ax.barh(np.arange(len(top)), top.mean_abs_shap, height=0.6, color="#2a78d6")
    ax.set_yticks(np.arange(len(top)), [PRETTY.get(f, f) for f in top.feature], fontsize=6.5)
    ax.set_xlabel("Mean |SHAP| (log-odds)")
    ax.grid(axis="y", visible=False)
    panel(ax, "b")
    fig.tight_layout(w_pad=3.0)
    save(fig, "Fig4")


def fig5(d: pd.DataFrame) -> None:
    sv = pd.read_csv(RES / "shap_values_full_fit.csv")
    feats = [("lon_w_deg_clean", "Longitude (deg, W > 0)", False), ("cme_speed_kms", r"CME speed (km s$^{-1}$)", True),
             ("log10_sxr_fluence_jm2", r"log$_{10}$ SXR fluence (J m$^{-2}$)", False),
             ("wind_radio_combined_score_clean", r"WIND/WAVES type II/IV score $X_W$", False)]
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 0.72 * WIDTH), sharey=True)
    for ax, (f, lab, logx), l in zip(axes.ravel(), feats, "abcd"):
        x, s = d[f], sv[f"shap_{f}"]
        m = x.notna()
        ax.scatter(x[m & (d.y_sep == 0)], s[m & (d.y_sep == 0)], s=4, color="#b5b4ad", lw=0, label="no SEP")
        ax.scatter(x[m & (d.y_sep == 1)], s[m & (d.y_sep == 1)], s=12, facecolor="#eb6834", edgecolor="white",
                   lw=0.4, label="SEP")
        ax.axhline(0, color=INK, lw=0.6)
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(lab)
        panel(ax, l)
    for ax in axes[:, 0]:
        ax.set_ylabel("SHAP value (log-odds)")
    axes[0, 0].legend(loc="upper left")
    fig.tight_layout(h_pad=1.0, w_pad=1.0)
    save(fig, "Fig5")


def fig6(d: pd.DataFrame) -> None:
    f = pd.read_csv(RES / "alert_frequency_over_repeats.csv")
    pos = d.merge(f, on="event_key")
    pos = pos[pos.y_sep == 1]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 0.42 * WIDTH), sharey=True)
    specs = [
        (axes[0], pd.cut(pos.peak_pfu_10mev, [10, 30, 100, 1000, 1e6], right=False),
         ["10-30", "30-100", "100-1000", ">1000"], r"Peak >10 MeV flux (pfu)"),
        (axes[1], pd.cut(pos.lon_w_deg_clean, [-91, -30, 0, 30, 60, 91]),
         ["E90–\nE30", "E30–\n0", "0–\nW30", "W30–\nW60", "W60–\nW90"], "Flare longitude"),
    ]
    for (ax, cats, labels, xl), l in zip(specs, "ab"):
        x = np.arange(len(labels))
        for off, (key, c, h, lab) in zip((-0.18, 0.18), (("catboost_TSS", "#2a78d6", "", "TSS-optimal"),
                                                         ("catboost_HSS", "#eb6834", "////", "HSS-optimal"))):
            g = pos.groupby(cats, observed=False)[key]
            mean, n = g.mean().to_numpy(), g.size().to_numpy()
            err = 1.96 * np.sqrt(np.clip(mean * (1 - mean), 0.02, None) / np.maximum(n, 1))
            ax.bar(x + off, mean, width=0.34, color=c, hatch=h, edgecolor="white", label=lab,
                   yerr=err, error_kw=dict(lw=0.6, ecolor=INK, capsize=1.5))
            if off < 0:
                for xi, ni in zip(x, n):
                    ax.text(xi, 1.11, f"n={ni}", ha="center", fontsize=6.3, color=MUTED)
        ax.set_xticks(x, labels, fontsize=6.3)
        ax.set_xlabel(xl)
        ax.set_ylim(0, 1.18)
        ax.grid(axis="x", visible=False)
        panel(ax, l)
    axes[0].set_ylabel("POD (CatBoost, 10 CV repeats)")
    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(w_pad=0.8)
    save(fig, "Fig6")


# ------------------------------------------------------------------ tables


def texnum(v: float) -> str:
    """Three significant digits in math mode; powers of ten for large values."""
    if not np.isfinite(v):
        return "--"
    if v != 0 and (abs(v) >= 1e4 or abs(v) < 1e-3):
        e = int(np.floor(np.log10(abs(v))))
        return f"${v / 10 ** e:.2f}\\times10^{{{e}}}$"
    if abs(v) >= 100:
        return f"${v:.0f}$"
    return f"${v:.3g}$"


def pm(mean: float, sd: float, nd: int = 3) -> str:
    return f"{mean:.{nd}f}$\\pm${sd:.{nd}f}"


def tables(d: pd.DataFrame) -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    qc = json.loads((RES / "qc_summary.json").read_text())
    s = pd.read_csv(RES / "cv_summary.csv")
    name = {k: v["label"] for k, v in STYLE.items()}
    lines = []
    for _, r in s.sort_values(["threshold_rule", "model"]).iterrows():
        lines.append(" & ".join([
            name[r.model], r.threshold_rule.split(" (")[0].replace("max ", "max\\,"),
            pm(r.ROC_AUC_mean, r.ROC_AUC_std), pm(r.PR_AP_mean, r.PR_AP_std), pm(r.BSS_mean, r.BSS_std, 2),
            pm(r.POD_mean, r.POD_std, 2), pm(r.FAR_mean, r.FAR_std, 2), pm(r.TSS_mean, r.TSS_std, 2),
            pm(r.HSS_mean, r.HSS_std, 2), f"{r.TP_mean:.0f}/{r.FP_mean:.0f}",
        ]) + r" \\")
    (TAB / "tab_cv.tex").write_text("\n".join(lines) + "\n")

    c = pd.read_csv(RES / "chronological_test.csv")
    lines = []
    for _, r in c.sort_values(["threshold_rule", "model"]).iterrows():
        lines.append(" & ".join([
            name[r.model], r.threshold_rule.split(" (")[0].replace("max ", "max\\,"), f"{r.ROC_AUC:.3f}",
            f"{r.PR_AP:.2f}", f"{int(r.TP)}/{int(r.FP)}/{int(r.FN)}",
            f"{r.POD:.2f} [{r.POD_lo95:.2f}, {r.POD_hi95:.2f}]", f"{r.FAR:.2f} [{r.FAR_lo95:.2f}, {r.FAR_hi95:.2f}]",
            f"{r.TSS:.2f} [{r.TSS_lo95:.2f}, {r.TSS_hi95:.2f}]", f"{r.HSS:.2f}",
        ]) + r" \\")
    (TAB / "tab_chrono.tex").write_text("\n".join(lines) + "\n")

    lg = pd.read_csv(RES / "legacy_comparison_summary.csv")
    lines = []
    for _, r in lg.sort_values("configuration").iterrows():
        lines.append(" & ".join([
            r.configuration.split(":")[0], f"{int(r.n_rows)}/{int(r.n_sep)}", pm(r.ROC_AUC_mean, r.ROC_AUC_std),
            pm(r.PR_AP_mean, r.PR_AP_std), pm(r.POD_mean, r.POD_std, 2), pm(r.FAR_mean, r.FAR_std, 2),
            pm(r.TSS_mean, r.TSS_std, 2), pm(r.HSS_mean, r.HSS_std, 2),
        ]) + r" \\")
    (TAB / "tab_legacy.tex").write_text("\n".join(lines) + "\n")

    a = pd.read_csv(RES / "ablation_summary.csv")
    order = ["all features"] + [f"without {g}" for g in M.FEATURE_GROUPS] + [f"only {g}" for g in M.FEATURE_GROUPS] + \
            ["logistic baseline set", "all features, strict CME timing"]
    lines = []
    for v in order:
        r = a[a.variant == v].iloc[0]
        lab = v
        for g, gl in M.GROUP_LABELS.items():
            lab = lab.replace(g, gl)
        lines.append(f"{lab} & {int(r.n_features)} & {pm(r.ROC_AUC_mean, r.ROC_AUC_std)} & "
                     f"{pm(r.PR_AP_mean, r.PR_AP_std)} & {pm(r.BSS_mean, r.BSS_std, 2)} \\\\")
    (TAB / "tab_ablation.tex").write_text("\n".join(lines) + "\n")

    # Feature table: coverage and class medians.
    defs = {
        "log10_peak_flux_wm2": r"$\log_{10}F_{\rm peak}$ (W m$^{-2}$)",
        "log10_sxr_fluence_jm2": r"$\log_{10}\Phi_{\rm SXR}$ (J m$^{-2}$)",
        "rise_time_min": r"$t_{\rm p}-t_0$ (min)",
        "duration_min": r"$t_1-t_0$ (min)",
        "xs_ratio_10m": r"$R_{\rm XS,10}$",
        "goes_ratio_derivative": r"$D_{\rm XS}$ (min$^{-1}$)",
        "sxr_peak_flux_derivative_max": r"$D_{\rm SXR}$ (min$^{-1}$)",
        "lon_w_deg_clean": r"$\lambda$ (deg)",
        "connection_cos45": r"$C_{\rm W45}=\cos(\lambda-45^\circ)$",
        "cme_speed_kms": r"$v_{\rm CME}$ (km s$^{-1}$)",
        "cme_width_deg": r"$w_{\rm CME}$ (deg)",
        "cme_shock_proxy": r"$S_{\rm CME}$",
        "type_ii_ground": "type II flag (ground)",
        "type_ii_log_power": "type II log power",
        "type_iv_present": "type IV flag",
        "wind_type2_inverse_delay_clean": r"$1/\Delta t_{\rm II}^{\rm WIND}$ (clean)",
        "wind_type2_log_power_clean": "WIND type II log power (clean)",
        "wind_radio_combined_score_clean": r"$X_{W,\rm comb}$ (clean)",
        "rstn_wind_bridge": r"$B_{\rm RSTN/WIND}$",
        "radio_coherence": r"$R_{\rm coh}$",
        "rstn_log_integral_high": r"$\log_{10}(1+\sum_{\nu\ge1.4\,\rm GHz}I_\nu)$",
        "rstn_log_integral_low": r"$\log_{10}(1+\sum_{\nu\le0.6\,\rm GHz}I_\nu)$",
        "srs_area_msh": r"$A_{\rm AR}$ (MSH)",
        "srs_mag_class_score": r"$M_{\rm AR}$",
        "srs_delta_flag": r"$\Delta_\delta$",
        "ar_energy_proxy": r"$E_{\rm AR}$",
        "ar_complexity_proxy": r"$C_{\rm AR}$",
        "flare_history_24h": r"$H_{24}$",
    }
    for f in M.RSTN_FREQS:
        defs[f"rstn_{f}_log_peak"] = rf"$\log_{{10}}(1+S_{{{f}}})$"
    lines = []
    for g, feats in M.FEATURE_GROUPS.items():
        lines.append(rf"\multicolumn{{4}}{{l}}{{\textit{{{M.GROUP_LABELS[g]}}}}} \\")
        for f in feats:
            x = d[f]
            cov = 100 * x.notna().mean()
            m0, m1 = x[d.y_sep == 0].median(), x[d.y_sep == 1].median()
            lines.append(f"\\quad {defs.get(f, f)} & {cov:.0f} & {texnum(m0)} & {texnum(m1)} \\\\")
    (TAB / "tab_features.tex").write_text("\n".join(lines) + "\n")
    (TAB / "qc_numbers.json").write_text(json.dumps(qc, indent=2))


def main() -> None:
    d, p = load()
    cv = pd.read_csv(RES / "cv_repeats.csv")
    fig1(d)
    fig2(d, p)
    fig3(p, cv)
    fig4()
    fig5(d)
    fig6(d)
    tables(d)
    print("figures:", sorted(x.name for x in FIG.glob("Fig*.pdf")))


if __name__ == "__main__":
    main()
