"""Generate manuscript tables and figures from validated CSV outputs."""

from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "results/figures"
DATA = ROOT / "results/summary"
HERE.mkdir(parents=True, exist_ok=True)

manifest = json.loads((DATA / "data_manifest.json").read_text())
for name, expected_hash in manifest.items():
    if hashlib.sha256((DATA / name).read_bytes()).hexdigest() != expected_hash:
        raise ValueError(f"Supporting data checksum mismatch: {name}")


def rows(name):
    with (DATA / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def number(value, decimals=3):
    return f"{float(value):.{decimals}f}"


def pvalue(value):
    value = float(value)
    if value < .001:
        exponent = int(np.floor(np.log10(value)))
        return rf"{value / 10**exponent:.2f}\times10^{{{exponent}}}"
    return f"{value:.5f}".rstrip("0").rstrip(".")


def effect(row, pkey):
    value = float(row["dz"])
    formatted = f"{value:+.4f}" if abs(value) < .001 else f"{value:+.3f}"
    star = "^{*}" if float(row[pkey]) < .05 else ""
    return f"${formatted}{star}$"


primary = rows("primary_statistics.csv")
assert len(primary) == 4 and all(int(r["n"]) == 106 for r in primary)
names = {"imagery-rest": "Imagery--rest", "execution-rest": "Execution--rest",
         "imagery-execution": "Imagery--execution", "real-surrogate": "Real--surrogate changes"}
table = [r"\begin{table*}[!t]", r"\centering",
         r"\caption{Primary all-epoch RSEC contrasts ($n=106$). Mean differences and their 95\% participant-bootstrap intervals are scaled by $10^{-3}$. $p_{\mathrm{H4}}$ adjusts the four primary tests; intervals are not multiplicity-adjusted. Negative counts refer to paired differences below zero.}",
         r"\label{tab:global_results}", r"\small", r"\setlength{\tabcolsep}{5pt}",
         r"\renewcommand{\arraystretch}{1.15}", r"\arrayrulecolor[gray]{0.70}",
         r"\begin{tabular}{|l|r|c|r|r|r|}", r"\hline", r"\rowcolor{light-gray}",
         r"\textbf{Contrast} & \textbf{Mean} & \textbf{95\% CI} & $d_z$ & $p_{\mathrm{H4}}$ & \textbf{Negative} \\ \hline"]
for row in primary:
    dz = number(row["dz"], 4 if abs(float(row["dz"])) < .001 else 3)
    cells = [names[row["contrast"]], f"${float(row['mean_delta'])*1000:.3f}$",
             rf"$[{float(row['ci_low'])*1000:.3f},\,{float(row['ci_high'])*1000:.3f}]$",
             f"${dz}$", f"${pvalue(row['p_holm'])}$", f"{row['n_decrease']}/106"]
    table.append(" & ".join(cells) + r" \\ \hline")
table.extend([r"\end{tabular}", r"\end{table*}"])
(HERE / "table_primary.tex").write_text("\n".join(table) + "\n", encoding="ascii")

comparison = rows("statistics_54.csv")
assert len(comparison) == 54
index = {(r["band"], r["contrast"], r["metric"]): r for r in comparison}
metrics = ["rsec_d3", "power_uv2", "coh", "plv", "pli", "wpli"]
bands = [("broadband", "Broadband"), ("mu", "Mu"), ("beta", "Beta")]
contrasts = [("imagery-rest", "I--R"), ("execution-rest", "E--R"), ("imagery-execution", "I--E")]
table = [r"\begin{table*}[!t]", r"\centering",
         r"\caption{Signed paired effect sizes for matched conventional comparisons ($n=106$). I: imagery; E: execution; R: rest. Broadband: 1--31 Hz; mu: 8--13 Hz; beta: 13--30 Hz. A star marks Holm-adjusted $p<0.05$ across all 54 tests, not superiority over another method. All methods use the same 15-epoch selections.}",
         r"\label{tab:comparisons}", r"\small", r"\setlength{\tabcolsep}{6pt}",
         r"\renewcommand{\arraystretch}{1.15}", r"\arrayrulecolor[gray]{0.70}",
         r"\begin{tabular}{|l|c|r|r|r|r|r|r|}", r"\hline", r"\rowcolor{light-gray}",
         r"\textbf{Band} & \textbf{Contrast} & \textbf{RSEC} & \textbf{Power} & \textbf{Coherence} & \textbf{PLV} & \textbf{PLI} & \textbf{wPLI} \\ \hline"]
for band, label in bands:
    for contrast, short in contrasts:
        table.append(" & ".join([label, short] + [effect(index[band, contrast, m], "p_holm_54") for m in metrics]) + r" \\ \hline")
table.extend([r"\end{tabular}", r"\end{table*}"])
(HERE / "table_comparisons.tex").write_text("\n".join(table) + "\n", encoding="ascii")

sensitivity = rows("sensitivity_statistics.csv")
idx = {(r["setting"], r["contrast"]): r for r in sensitivity}
settings = [("d2", "$d=2$"), ("d4", "$d=4$"), ("d5", "$d=5$"),
            ("average-reference", "Average reference"), ("no-frontopolar", "Without Fp1/Fpz/Fp2"),
            ("mu-8-13", "Mu, 8--13 Hz"), ("beta-13-30", "Beta, 13--30 Hz")]
table = [r"\begin{table}[!htbp]", r"\centering",
         r"\caption{All-epoch RSEC sensitivity ($n=106$). Entries are signed $d_z$; stars denote Holm $p<0.05$ across seven settings and three contrasts (21 tests). Only imagery-related contrasts are shown; the correction family also includes execution--rest tests, available in the supplementary results. The primary $d=3$ result belongs to Table~\ref{tab:global_results}'s separate family.}",
         r"\label{tab:sensitivity}", r"\small", r"\setlength{\tabcolsep}{4pt}",
         r"\renewcommand{\arraystretch}{1.15}", r"\arrayrulecolor[gray]{0.70}",
         r"\begin{tabular}{|l|r|r|}", r"\hline", r"\rowcolor{light-gray}",
         r"\textbf{Setting} & \textbf{I--R} & \textbf{I--E} \\ \hline"]
for setting, label in settings:
    table.append(" & ".join([label, effect(idx[setting, "imagery-rest"], "p_holm"), effect(idx[setting, "imagery-execution"], "p_holm")]) + r" \\ \hline")
table.extend([r"\end{tabular}", r"\end{table}"])
(HERE / "table_sensitivity.tex").write_text("\n".join(table) + "\n", encoding="ascii")

subjects = rows("primary_subject_summary.csv")
nodes = rows("imagery_node_deltas_by_subject.csv")
assert [r["subject"] for r in subjects] == [r["subject"] for r in nodes]
global_data = [np.array([float(r[a]) - float(r[b]) for r in subjects]) for a, b in
               [("imagery", "rest"), ("execution", "rest"), ("imagery", "execution")]]
fp = np.array([np.mean([float(r[k]) for k in ("Fp1", "Fpz", "Fp2")]) for r in nodes])
sm = np.array([np.mean([float(r[k]) for k in ("C3", "Cz", "C4")]) for r in nodes])
regional_data = [fp, sm, fp-sm]
regional = rows("imagery_roi_statistics.csv")
assert [np.sum(a < 0) for a in global_data] == [66, 49, 84]
assert [np.sum(a < 0) for a in regional_data] == [78, 54, 76]

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9, "axes.titlesize": 9,
                     "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": False})
fig, axes = plt.subplots(1, 2, figsize=(6.45, 2.45), layout="constrained")
labels = [["Imagery\nminus rest", "Execution\nminus rest", "Imagery minus\nexecution"],
          ["Frontopolar", "Sensorimotor", "Fp minus SM"]]
for ax, data, summaries, names, title, color in zip(
        axes, [global_data, regional_data], [primary[:3], regional], labels,
        ["(a) Global density", "(b) Regional node participation"], ["#246b78", "#755070"]):
    rng = np.random.default_rng(20261002)
    scaled = [a*1000 for a in data]
    ax.boxplot(scaled, widths=.45, showfliers=False, patch_artist=True,
               boxprops={"facecolor": "#eeeeee", "edgecolor": "#333333"},
               medianprops={"color": "#111111"}, whiskerprops={"color": "#555555"},
               capprops={"color": "#555555"})
    for x, values, record in zip(range(1, 4), scaled, summaries):
        np.testing.assert_allclose(values.mean(), float(record["mean_delta"])*1000, atol=1e-10)
        ax.scatter(x+rng.uniform(-.16, .16, len(values)), values, s=7, alpha=.4, color=color,
                   linewidths=0, zorder=2)
        mean = float(record["mean_delta"])*1000
        low, high = float(record["ci_low"])*1000, float(record["ci_high"])*1000
        ax.errorbar(x+.25, mean, yerr=[[mean-low], [high-mean]], color="#111111", marker="D",
                    markersize=3, capsize=2, linewidth=1, zorder=4)
    ax.axhline(0, color="#888888", linestyle="--", linewidth=.7)
    ax.set_xticks([1, 2, 3], [f"{name}\n{int(np.sum(a<0))}/106 lower" for name, a in zip(names, data)])
    ax.set_title(title, loc="left", pad=8)
    ax.set_ylabel(r"Difference ($\times 10^{-3}$)")
    ax.set_xlim(.6, 3.5)
fig.savefig(HERE / "participant_spatial_results.pdf", metadata={"Title": "Participant-level RSEC differences"})
fig.savefig(HERE / "participant_spatial_results.png", dpi=300)
plt.close(fig)

paired = rows("paired_effect_comparisons_45.csv")
imagery_comparisons = [r for r in paired if r["contrast"] != "execution-rest"]
assert len(paired) == 45 and len(imagery_comparisons) == 30
assert all(float(r["ci_low"]) < 0 < float(r["ci_high"]) for r in imagery_comparisons)
assert all(np.isclose(float(r["interval_confidence"]), 1 - .05/45) for r in paired)
paired_index = {(r["band"], r["contrast"], r["baseline"]): r for r in paired}
baseline_names = ["Power", "Coherence", "PLV", "PLI", "wPLI"]
fig, axes = plt.subplots(2, 3, figsize=(6.45, 3.65), sharex=True, layout="constrained")
panel_titles = []
for row_number, (contrast, contrast_name) in enumerate([
        ("imagery-rest", "Imagery minus rest"),
        ("imagery-execution", "Imagery minus\nexecution")]):
    for col_number, (band, band_name) in enumerate(bands):
        ax = axes[row_number, col_number]
        records = [paired_index[band, contrast, m] for m in metrics[1:]]
        points = np.array([float(r["baseline_minus_rsec_abs_dz"]) for r in records])
        lower = np.array([float(r["ci_low"]) for r in records])
        upper = np.array([float(r["ci_high"]) for r in records])
        y = np.arange(len(records))
        ax.errorbar(points, y, xerr=[points-lower, upper-points], fmt="o",
                    color="#246b78" if row_number == 0 else "#755070",
                    markersize=3.5, capsize=2.5, linewidth=1, zorder=3)
        ax.axvline(0, color="#777777", linestyle="--", linewidth=.8, zorder=1)
        ax.set_yticks(y, baseline_names)
        ax.set_ylim(4.6, -.6)
        panel = chr(ord("a") + row_number*3 + col_number)
        panel_titles.append(ax.set_title(f"({panel}) {band_name}\n{contrast_name}", loc="left", pad=8))
        ax.set_xlim(-.65, .85)
        ax.set_xticks([-.5, 0, .5])
        ax.grid(axis="x", color="#dddddd", linewidth=.5, zorder=0)
fig.supxlabel(r"Baseline $|d_z|$ minus RSEC $|d_z|$", fontsize=9)
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
for title in panel_titles:
    bounds = title.get_window_extent(renderer)
    assert 0 <= bounds.x0 < bounds.x1 <= fig.bbox.width
    assert 0 <= bounds.y0 < bounds.y1 <= fig.bbox.height
fig.savefig(HERE / "comparison_uncertainty.pdf",
            metadata={"Title": "Uncertainty in paired effect-size comparisons"})
fig.savefig(HERE / "comparison_uncertainty.png", dpi=300)
plt.close(fig)
print("Generated 3 tables and 2 figures from verified CSV inputs.")
