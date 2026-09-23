"""Draw the four article figures directly with Matplotlib. No LaTeX is used.

Run: python build_control_figures.py
Inputs: saved experiment JSON in output/. Exports: PNG, SVG and PDF there.
Overview uses one-based (source,receiver) links. Voltage edges use zero-based
(receiver,source) pairs. Voltage curves are marginal 90th percentiles, not CIs.
"""
from pathlib import Path
from fractions import Fraction
import json
import os
import sys
import tempfile

sys.dont_write_bytecode = True
_cache = tempfile.TemporaryDirectory(prefix="sls-figures-")
os.environ.setdefault("MPLCONFIGDIR", _cache.name)
os.environ.setdefault("XDG_CACHE_HOME", _cache.name)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
import numpy as np
from write_tradeoff_data import main as tradeoff_data

OUT = Path(__file__).resolve().parent / "output"
FORMATS = ("png", "svg", "pdf")
WIDTH = 178 / 25.4
INK, GRAY = "#171717", "#707070"
plt.rcParams.update({"text.usetex": False, "font.family": "DejaVu Sans",
    "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .65, "svg.fonttype": "none", "pdf.fonttype": 42,
    "figure.facecolor": "white", "savefig.facecolor": "white"})


def read(name):
    return json.loads((OUT / name).read_text())


def number(value):
    return float(Fraction(value["rational"])) if isinstance(value, dict) else float(value)


def save(fig, name):
    for extension in FORMATS:
        path = OUT / f"{name}.{extension}"
        fig.savefig(path, dpi=300)
        print(path)
    plt.close(fig)


def overview():
    """Candidate links and installed links, with the saved controller values."""
    data = read("shared_link_example.json")
    positions = {1: (0, 1.8), 2: (2, 1.8), 3: (4.5, 3), 4: (4.5, .6)}
    cost_positions = {(1, 2): (1, 2.12), (2, 3): (3.2, 2.74),
                      (2, 4): (3.2, .88), (1, 3): (1.6, 3.34), (1, 4): (1.6, .20)}
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 82/25.4))
    plans = [data["greedy_refit"], data["exact_best"]]
    titles = ["(a) Greedy addition and refitting", "(b) Joint graph and controller design"]
    for ax, plan, title in zip(axes, plans, titles):
        for installed in (False, True):
            for edge, (j, i, delay, cost) in enumerate(data["edges_source_receiver_latency_cost"]):
                if (edge in plan["edges"]) != installed:
                    continue
                rad = -.38 if (j, i) == (1, 3) else .38 if (j, i) == (1, 4) else 0
                ax.add_patch(FancyArrowPatch(positions[j], positions[i],
                    connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
                    shrinkA=10, shrinkB=10, mutation_scale=9,
                    color=INK if installed else "#BBBBBB", lw=1.3 if installed else .75))
                if installed:
                    ax.text(*cost_positions[j, i], f"cost {cost}", ha="center", va="center",
                            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
        for node, xy in positions.items():
            ax.add_patch(Circle(xy, .20, fc="white", ec=INK, lw=.7, zorder=4))
            ax.text(*xy, str(node), ha="center", va="center", zorder=5)
        ax.text(-.42, 1.8, r"$\xi$", ha="right", va="center")
        ax.text(2, 1.37, "relay", ha="center", color=GRAY)
        labels = []
        for receiver in ("3", "4"):
            u = [number(v) for v in plan["receivers"][receiver]["residual"]["u"]]
            first = next((t for t, v in enumerate(u) if abs(v) > 1e-12), None)
            labels.append(rf"$u_{{{receiver},t}}=0$" if first is None else
                          rf"$u_{{{receiver},{first}}}={u[first]:.3f}\xi$")
        ax.text(2.15, -.38, ", ".join(labels), ha="center")
        ax.text(2.15, -.87, rf"Control cost $J={number(plan['objective']):.3f}$", ha="center")
        ax.set(xlim=(-.7, 5), ylim=(-1.08, 4.02), aspect="equal")
        ax.set_title(title, loc="left", pad=10)
        ax.axis("off")
    handles = [Line2D([], [], color=INK, lw=1.3, label="Installed"),
               Line2D([], [], color="#BBBBBB", lw=.75, label="Candidate")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(.40, .015))
    fig.text(.73, .045, rf"Same budget $C={data['budget']}$", ha="center")
    fig.subplots_adjust(left=.04, right=.98, bottom=.16, top=.86, wspace=.16)
    save(fig, "overview")


def tradeoff(rows):
    """Original normalization and all five methods, with common panel scales."""
    styles = [("Greedy refit", "Greedy refit", INK, "-", "o", 4.4),
              ("Perspective refit", "Perspective", INK, "--", "o", 2.2),
              ("Magnitude refit", "Response norm", "#555555", (0, (5, 2)), "s", 3.5),
              ("Sensitivity refit", "Sensitivity", "#777777", "-.", "^", 4),
              ("Distance refit", "Feeder distance", "#999999", ":", "D", 3.5)]
    modes = [("No forecast", "(a) Baseline innovation covariance"),
             ("Forecast residual", "(b) Residual innovation covariance")]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 86/25.4), sharey=True)
    for ax, (mode, title) in zip(axes, modes):
        for method, label, color, line, marker, size in styles:
            values = sorted([r for r in rows if r["mode"] == mode and r["method"] == method],
                            key=lambda r: r["budget"])
            ax.plot([r["budget"] for r in values], [r["normalized_gap"] for r in values],
                label=label, color=color, ls=line, lw=1.1, marker=marker, ms=size,
                mfc=INK if method == "Perspective refit" else "white", mew=.8,
                markevery=list(range(1, len(values)-1)))
        ax.set_title(title, loc="left", pad=9)
        ax.set(xlim=(-1, 57), ylim=(-.02, 1.02), xticks=[0, 8, 16, 24, 56],
               yticks=np.linspace(0, 1, 6), xlabel="Communication budget C (directed links)")
        ax.grid(axis="y", color="#E2E2E2", lw=.45)
    axes[0].set_ylabel(r"Normalized cost gap $\bar J(C)$")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3,
               frameon=False, bbox_to_anchor=(.52, .99), columnspacing=1.8, handlelength=2.5)
    fig.subplots_adjust(left=.095, right=.98, bottom=.19, top=.72, wspace=.16)
    save(fig, "tradeoff")


def topology():
    """Physical feeder plus the selected C=12 Perspective communication matrix."""
    model = read("archived_codesign_records.json")["model"]
    records = read("voltage_record_verification.json")["records"]
    selected = next(r for r in records if r["mode"] == "Forecast residual"
                    and r["method"] == "Perspective refit" and r["budget"] == 12)
    parent = model["parent_zero_based"]
    if len(parent) != 8:
        raise ValueError("Update the feeder node positions for a non-eight-node model")
    xy = np.array([[.55, 2.65], [1.8, 4.2], [1.8, 1.1], [3.05, 4.2],
                   [3.05, 3.25], [3.05, 2.05], [3.05, 1.1], [4.4, 2.05]])
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 88/25.4))
    ax = axes[0]
    for i, p in enumerate(parent):
        if p >= 0:
            ax.plot(xy[[i, p], 0], xy[[i, p], 1], color="#888888", lw=1)
    ax.scatter(xy[:, 0], xy[:, 1], s=200, facecolors="white", edgecolors=INK, linewidths=.8, zorder=3)
    for i, (x, y) in enumerate(xy):
        ax.text(x, y, str(i+1), ha="center", va="center", zorder=4)
    ax.set(xlim=(0, 5), ylim=(.4, 4.9), aspect="equal")
    ax.axis("off")
    ax = axes[1]
    for i in range(8):
        ax.add_patch(Rectangle((i-.5, i-.5), 1, 1, fc="#EAEAEA", ec="none"))
    edges = np.asarray(selected["edges"])
    ax.scatter(edges[:, 1], edges[:, 0], marker="s", s=20, color=INK, zorder=3)
    ax.set(xlim=(-.5, 7.5), ylim=(7.5, -.5), aspect="equal", xticks=range(8), yticks=range(8),
           xticklabels=range(1, 9), yticklabels=range(1, 9),
           xlabel="Source bus j", ylabel="Receiver bus i")
    ax.set_xticks(np.arange(-.5, 8, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 8, 1), minor=True)
    ax.grid(which="minor", color="#CCCCCC", lw=.4)
    ax.tick_params(which="both", length=0)
    ax.xaxis.tick_top()
    fig.text(.06, .90, "(a) Radial electrical feeder", fontsize=9)
    fig.text(.59, .90, "(b) Communication graph, C=12", fontsize=9)
    fig.legend(handles=[Line2D([], [], color=INK, marker="s", ls="", ms=4, label="Installed j to i"),
                        Rectangle((0, 0), 1, 1, fc="#EAEAEA", label="Free local information")],
               loc="lower center", bbox_to_anchor=(.5, .008), ncol=2, frameon=False)
    fig.subplots_adjust(left=.035, right=.97, bottom=.20, top=.83, wspace=.27)
    save(fig, "topology")


def voltage_envelope():
    """Full-precision marginal p90 values, and their differences from Full."""
    data = read("voltage_envelope_verification.json")
    if data["selected_method"] != "Perspective refit":
        raise ValueError("Expected the original Perspective refit voltage experiment")
    curves = {method: np.asarray(data["curves"][method]["p90_percent"])
              for method in ("Local", "Perspective", "Full")}
    t = np.arange(len(curves["Full"]))
    styles = [("Local", "Local, C=0", "#888888", ":", None),
              ("Perspective", "Selected, C=12", INK, "-", "o"),
              ("Full", "Full, C=56", "#555555", "--", None)]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 81/25.4))
    for method, label, color, line, marker in styles:
        for ax, values in zip(axes, (curves[method], curves[method]-curves["Full"])):
            ax.plot(t, values, label=label, color=color, ls=line, lw=1.2,
                    marker=marker, ms=3.8, mfc="white", mew=.8)
    axes[0].set(ylabel="90th percentile (% of nominal voltage)", ylim=(2.8, 4.1),
                yticks=[2.8, 3.2, 3.6, 4.0])
    axes[1].set(ylabel="Difference (percentage points)", ylim=(-.006, .206),
                yticks=[0, .05, .1, .15, .2])
    for ax, title in zip(axes, ("(a) Maximum absolute bus deviation", "(b) Difference from full communication")):
        ax.set_title(title, loc="left", pad=9)
        ax.set(xlabel="Time index t", xticks=t, xlim=(-.1, len(t)-.9))
        ax.grid(axis="y", color="#E2E2E2", lw=.45)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3,
               frameon=False, bbox_to_anchor=(.52, .99), columnspacing=2, handlelength=2.5)
    fig.subplots_adjust(left=.09, right=.97, bottom=.21, top=.78, wspace=.37)
    save(fig, "voltage_envelope")


def main():
    OUT.mkdir(exist_ok=True)
    rows = tradeoff_data()
    overview()
    tradeoff(rows)
    topology()
    voltage_envelope()


if __name__ == "__main__":
    main()
