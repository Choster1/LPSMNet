import argparse
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats

from error_model import MODELS, consistency_table
from simulator import SCENARIOS

HERE = os.path.dirname(os.path.abspath(__file__))
COL = {"DNDM": "#d9822b", "SFM": "#2e7d32", "LPSMNet": "#1f5fa8"}
SHADES = {"evening": [(1.55, 2.45, "incident\naftermath"), (4.55, 6.45, "roadworks")],
          "morning": [(0.55, 2.45, "roadworks"), (6.55, 7.45, "incident,\npeak")]}
LEVELS = [("night_factor", 0.6, "Night demand\n×0.6"), ("base", "base", "Base\ncase"), ("night_factor", 1.4, "Night demand\n×1.4"),
          ("compliance", 0.4, "Compliance\n0.4"), ("compliance", 1.0, "Compliance\n1.0"),
          ("theta", 1 / 240, "θ = 1/240\ns$^{-1}$"), ("theta", 1 / 60, "θ = 1/60\ns$^{-1}$"),
          ("dt_ctrl", 30, "Interval\n30 s"), ("dt_ctrl", 180, "Interval\n180 s")]


def load(path):
    df = pd.read_csv(path)
    df["level"] = df["level"].astype(str)
    return df


def sel(df, cond, matching="-"):
    return df[(df.condition == cond) & (df.matching == matching)].set_index("rep")


def rel(d, ref, col):
    x, y = d[col].align(ref[col], join="inner")
    v = 100 * (x - y) / y
    return v.mean(), 1.96 * v.std(ddof=1) / math.sqrt(len(v))


def paired(base, lps, col):
    x, y = base[col].align(lps[col], join="inner")
    if (x == y).all():
        return 0.0, 0.0, 1.0
    r = 100 * (x - y) / x
    return r.mean(), 1.96 * r.std(ddof=1) / math.sqrt(len(r)), stats.ttest_rel(x, y).pvalue


def fmt_p(p):
    return "< 0.001" if p < 1e-3 else f"{p:.3f}"


def level_subset(df, factor, level, n_sens):
    if factor == "base":
        return df[(df.study == "main") & (df.rep < n_sens)]
    d = df[(df.study == "sens") & (df.factor == factor)]
    return d[d.level.astype(float).round(6) == round(float(level), 6)]


def main_summary(name, df, L):
    mn = df[df.study == "main"]
    ref, ng = sel(mn, "perfect"), sel(mn, "no_guidance")
    first = SCENARIOS[name]["first_period"]
    second = "night" if first == "peak" else "peak"
    ngx = rel(ng, ref, "TTS_all")[0]
    L.append(f"\n## {name.capitalize()} period\n")
    L.append(f"Perfect sensing: TTS = {ref.TTS_all.mean():.0f} veh·h. Guidance saves {100 * (1 - 1 / (1 + ngx / 100)):.1f}% "
             f"of TTS (no guidance: {ngx:+.1f}% above perfect sensing; {first} departures {rel(ng, ref, f'TTS_{first}')[0]:+.1f}%, "
             f"{second} departures {rel(ng, ref, f'TTS_{second}')[0]:+.1f}%).")
    if name == "evening":
        ods = [c for c in df.columns if c.startswith("TTS_OD")]
        L.append("Effect of guidance on travel time by OD (perfect sensing vs. no guidance): " + ", ".join(
            f"{c[4:]} {-rel(ng, ref, c)[0] / (1 + rel(ng, ref, c)[0] / 100):+.1f}%" for c in ods) + ".")
    L.append(f"\n| Matching | Model | TTS above perfect (%) | {first.capitalize()} | {second.capitalize()} | Kept (%) | "
             f"Wrong {first}/{second} (%) | Rejected {first}/{second} (%) | Candidates {first}/{second} |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for matching in ("naive", "traffic-aware"):
        for m in MODELS:
            d = sel(mn, m, matching)
            a = rel(d, ref, "TTS_all")
            L.append(f"| {matching} | {m} | {a[0]:+.1f} ± {a[1]:.1f} | {rel(d, ref, f'TTS_{first}')[0]:+.1f} | "
                     f"{rel(d, ref, f'TTS_{second}')[0]:+.1f} | {100 * (ngx - a[0]) / ngx:.0f} | "
                     f"{100 * d[f'wrong_{first}'].mean():.0f}/{100 * d[f'wrong_{second}'].mean():.0f} | "
                     f"{100 * d[f'reject_{first}'].mean():.0f}/{100 * d[f'reject_{second}'].mean():.0f} | "
                     f"{d[f'pool_{first}'].mean():.0f}/{d[f'pool_{second}'].mean():.0f} |")
    L.append("\nPaired TTS reduction of LPSMNet (% of the baseline TTS; positive = LPSMNet better):\n")
    L.append("| Matching | vs. | Whole period | p | " + f"{first.capitalize()} | {second.capitalize()} |")
    L.append("|---|---|---|---|---|---|")
    for matching in ("naive", "traffic-aware"):
        for b in ("DNDM", "SFM"):
            r = [paired(sel(mn, b, matching), sel(mn, "LPSMNet", matching), c) for c in ("TTS_all", f"TTS_{first}", f"TTS_{second}")]
            L.append(f"| {matching} | {b} | {r[0][0]:+.2f} | {fmt_p(r[0][2])} | {r[1][0]:+.2f} | {r[2][0]:+.2f} |")


def hourly(name, df):
    mn = df[df.study == "main"]
    t0 = int(SCENARIOS[name]["t0"])
    rows = []
    for h in range(8):
        row = dict(scenario=name, hour=(t0 + h) % 24)
        for b in ("SFM", "DNDM"):
            row[f"red_{b}"], row[f"ci_{b}"], row[f"p_{b}"] = paired(sel(mn, b, "naive"), sel(mn, "LPSMNet", "naive"), f"TTS_h{h}")
        for m in MODELS:
            row[f"wrong_{m}"] = 100 * sel(mn, m, "naive")[f"wrong_h{h}"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def sensitivity(df, n_sens):
    rows = []
    for factor, level, label in LEVELS:
        d0 = level_subset(df, factor, level, n_sens)
        ref = sel(d0, "perfect")
        ng = sel(d0, "no_guidance")
        if ng.empty:
            ng = sel(level_subset(df, "base", "base", n_sens), "no_guidance")
        row = dict(label=label.replace("\n", " ").replace("$^{-1}$", "^-1"), fig_label=label, no_guidance=rel(ng, ref, "TTS_all")[0])
        for m in MODELS:
            row[m] = rel(sel(d0, m, "naive"), ref, "TTS_all")[0]
            row[f"{m}_traffic_aware"] = rel(sel(d0, m, "traffic-aware"), ref, "TTS_all")[0]
        for b in ("SFM", "DNDM"):
            row[f"red_{b}"], row[f"ci_{b}"], row[f"p_{b}"] = paired(sel(d0, b, "naive"), sel(d0, "LPSMNet", "naive"), "TTS_all")
        rows.append(row)
    return pd.DataFrame(rows)


def style():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 8, "pdf.fonttype": 42, "axes.linewidth": 0.6,
                         "legend.frameon": True, "legend.edgecolor": "0.3", "legend.fancybox": False,
                         "legend.framealpha": 1.0, "legend.fontsize": 7.5})


def fig_moments(H, path):
    lo = min((H[f"red_{b}"] - H[f"ci_{b}"]).min() for b in ("SFM", "DNDM"))
    hi = max((H[f"red_{b}"] + H[f"ci_{b}"]).max() for b in ("SFM", "DNDM"))
    ytop, ybot = math.ceil(hi + 3.2), math.floor(lo - 0.8)
    fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.75), sharey=True)
    for ax, scen in zip(axs, ("evening", "morning")):
        d = H[H.scenario == scen].reset_index(drop=True)
        x = np.arange(len(d))
        for a, b, t in SHADES[scen]:
            ax.axvspan(a, b, color="0.92", zorder=0, lw=0)
            ax.text((a + b) / 2, ytop - 0.35, t, ha="center", va="top", fontsize=6.5, color="0.35", linespacing=0.95)
        for b in ("SFM", "DNDM"):
            dx = -0.07 if b == "SFM" else 0.07
            ax.errorbar(x + dx, d[f"red_{b}"], yerr=d[f"ci_{b}"], marker="o", ms=3.2, c=COL[b], lw=1.0,
                        elinewidth=0.7, capsize=2, zorder=3)
            sig = (d[f"p_{b}"] < 0.05).values
            ax.plot(x[sig] + dx, d[f"red_{b}"].values[sig], "o", ms=6.5, mfc="none", mec="k", mew=0.6, zorder=4)
        ax.axhline(0, c="k", lw=0.5, zorder=1)
        ax.set_xticks(x); ax.set_xticklabels([f"{h:02d}:00" for h in d.hour], fontsize=7)
        ax.set_xlim(-0.5, len(d) - 0.5); ax.set_ylim(ybot, ytop)
        ax.set_title(f"{scen.capitalize()} period"); ax.set_xlabel("Departure hour"); ax.grid(alpha=0.3, lw=0.4)
    axs[0].set_ylabel("TTS reduction by LPSMNet (%)")
    handles = [Line2D([], [], color=COL["SFM"], marker="o", ms=3.2, lw=1.0, label="vs. SFM"),
               Line2D([], [], color=COL["DNDM"], marker="o", ms=3.2, lw=1.0, label="vs. DNDM"),
               Line2D([], [], ls="", marker="o", ms=6.5, mfc="none", mec="k", mew=0.6, label="$p<0.05$ (paired t-test)")]
    axs[1].legend(handles=handles, loc="upper center", bbox_to_anchor=(4.5, ytop - 0.25), bbox_transform=axs[1].transData,
                  ncol=1, fontsize=7, borderpad=0.35, labelspacing=0.3, handlelength=1.8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_sensitivity(S, path):
    top = max(S[f"red_{b}"].max() + S[f"ci_{b}"].max() for b in ("SFM", "DNDM"))
    low = min(0, min((S[f"red_{b}"] - S[f"ci_{b}"]).min() for b in ("SFM", "DNDM")))
    fig, ax = plt.subplots(figsize=(7.16, 2.55))
    xx, w = np.arange(len(S)), 0.36
    for k, b in enumerate(("SFM", "DNDM")):
        ax.bar(xx + (k - 0.5) * w, S[f"red_{b}"], w * 0.95, yerr=S[f"ci_{b}"], color=COL[b], capsize=2, error_kw=dict(lw=0.6), zorder=2)
        for i, r in S.iterrows():
            if r[f"p_{b}"] < 0.05:
                yv = r[f"red_{b}"] + r[f"ci_{b}"] + 0.35 if r[f"red_{b}"] >= 0 else r[f"red_{b}"] - r[f"ci_{b}"] - 0.6
                ax.text(i + (k - 0.5) * w, yv, "*", ha="center", va="center", fontsize=10, zorder=3)
    ax.axhline(0, c="k", lw=0.5)
    ax.set_xticks(xx); ax.set_xticklabels(S.fig_label, fontsize=7, linespacing=0.95)
    ax.set_xlim(-0.6, len(S) - 0.4); ax.set_ylim(low - 0.5, top + 2.0)
    ax.set_ylabel("TTS reduction by LPSMNet (%)"); ax.grid(axis="y", alpha=0.3, lw=0.4, zorder=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COL["SFM"], label="vs. SFM"),
               plt.Rectangle((0, 0), 1, 1, color=COL["DNDM"], label="vs. DNDM"),
               Line2D([], [], ls="", marker="$*$", ms=7, color="k", label="$p<0.05$ (paired t-test)")]
    ax.legend(handles=handles, loc="upper left", ncol=1, fontsize=7, borderpad=0.35, labelspacing=0.3)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    ap.add_argument("--sens-reps", type=int, default=20)
    a = ap.parse_args()
    style()
    fig_dir = os.path.join(a.results, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    L = ["# Simulation results (measured-distance error model)"]
    C = consistency_table()
    C.to_csv(os.path.join(a.results, "error_model_consistency.csv"), index=False)
    L.append("\n## Error model\n")
    L.append("| Model | Pair | Same / different pairs | d' | Predicted Rank-1 (147 wrong candidates) | Correct with 1,000 candidates | "
             "False matches at one threshold (%) |")
    L.append("|---|---|---|---|---|---|---|")
    for r in C.itertuples():
        L.append(f"| {r.model} | {r.pair} | {r.n_same} / {r.n_diff} | {r.d_prime:.2f} | {100 * r.rank1_147:.1f}% | "
                 f"{100 * r.p_correct_1000:.0f}% | {100 * r.fpr_single_threshold:.1f} |")
    Hs = []
    for scen in ("evening", "morning"):
        path = os.path.join(a.results, f"{scen}.csv")
        if not os.path.exists(path):
            continue
        df = load(path)
        main_summary(scen, df, L)
        Hs.append(hourly(scen, df))
        if scen == "evening" and (df.study == "sens").any():
            S = sensitivity(df, a.sens_reps)
            S.drop(columns="fig_label").to_csv(os.path.join(a.results, "sensitivity.csv"), index=False)
            L.append("\n## Sensitivity study (evening, naive matching)\n")
            L.append("| Setting | No guidance | DNDM | SFM | LPSMNet | LPSMNet vs. SFM | LPSMNet vs. DNDM |")
            L.append("|---|---|---|---|---|---|---|")
            for r in S.itertuples():
                L.append(f"| {r.label} | {r.no_guidance:+.1f} | {r.DNDM:+.1f} | {r.SFM:+.1f} | {r.LPSMNet:+.1f} | "
                         f"{r.red_SFM:+.2f} (p {fmt_p(r.p_SFM)}) | {r.red_DNDM:+.2f} (p {fmt_p(r.p_DNDM)}) |")
            fig_sensitivity(S, os.path.join(fig_dir, "fig_sensitivity.pdf"))
    if Hs:
        H = pd.concat(Hs)
        H.to_csv(os.path.join(a.results, "hourly.csv"), index=False)
        L.append("\n## TTS reduction of LPSMNet by departure hour (naive matching)\n")
        L.append("| Period | Hour | vs. SFM (%) | p | vs. DNDM (%) | p | Wrong matches LPSMNet / SFM / DNDM (%) |")
        L.append("|---|---|---|---|---|---|---|")
        for r in H.itertuples():
            L.append(f"| {r.scenario} | {r.hour:02d}:00 | {r.red_SFM:+.1f} | {fmt_p(r.p_SFM)} | {r.red_DNDM:+.1f} | {fmt_p(r.p_DNDM)} | "
                     f"{r.wrong_LPSMNet:.1f} / {r.wrong_SFM:.1f} / {r.wrong_DNDM:.1f} |")
        if set(H.scenario) == {"evening", "morning"}:
            fig_moments(H, os.path.join(fig_dir, "fig_moments.pdf"))
    with open(os.path.join(a.results, "summary.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
