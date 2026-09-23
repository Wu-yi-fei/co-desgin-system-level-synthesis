"""Publication-size plots from saved theorem results. No optimizer is called."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

_CACHE = tempfile.TemporaryDirectory(prefix="sls-plot-")
os.environ.setdefault("MPLCONFIGDIR", _CACHE.name)
os.environ.setdefault("XDG_CACHE_HOME", _CACHE.name)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

INK, GRAY, TEAL = "#202020", "#686868", "#087E8B"


def plot_theorem():
    target = Path(__file__).resolve().parent/"output"
    source = target/"theorem.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    if not data["diagnostics"]["all_checks_pass"]:
        raise ValueError("Refusing to plot a run with failed checks")
    cases = sorted(data["cases"], key=lambda c: (c["preview"], -c["cap"]))
    plt.rcParams.update({"text.usetex": False, "font.family": "Times New Roman",
                         "mathtext.fontset": "stix", "font.size": 8,
                         "axes.labelsize": 8, "axes.titlesize": 9,
                         "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": .65, "lines.linewidth": 1.2,
                         "svg.fonttype": "none", "figure.facecolor": "white"})
    def save(fig, name, **kwargs):
        fig.canvas.draw()
        for suffix in ("png", "svg"):
            path = target/f"{name}.{suffix}"
            fig.savefig(path, dpi=300, facecolor="white", **kwargs)
            print(path)
        plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(247/25.4, 62/25.4),
                             squeeze=False, sharex=True, sharey=True)
    ymax = max(r["upper_bound"] for c in cases for r in c["budgets"])
    series = [("upper_bound", "Analytical upper bound", INK, "--", "^"),
              ("truncated_upper_loss", "Truncated-controller cost gap", GRAY, ":", "x"),
              ("loss", "Optimized cost gap", TEAL, "-", "o"),
              ("lower_bound", "Lower bound", GRAY, "-.", "s")]
    for ax, case in zip(axes.flat, cases):
        records = case["budgets"]
        C = np.array([r["budget"] for r in records])
        for key, label, color, linestyle, marker in series:
            y = np.array([r[key] for r in records])
            if key in ("loss", "truncated_upper_loss"):
                y[np.abs(y) <= data["verification_tolerance"]] = 0
            ax.plot(C, y, label=label, color=color, linestyle=linestyle,
                    marker=marker, markersize=3.4, markerfacecolor="white",
                    markeredgewidth=.8, drawstyle="steps-post", zorder=4 if key == "loss" else 3)
        ax.set_title(rf"$\rho_s=\max(0,s-{case['preview']})$,  $M_g={case['cap']:g}$",
                     loc="left", fontsize=8)
        ax.axvline(case["kappa_H"], color="#929292", lw=.8, zorder=1)
        ax.text(case["kappa_H"]+.12, .86, rf"$\kappa_H={case['kappa_H']}$",
                transform=ax.get_xaxis_transform(), fontsize=7.5)
        ax.set_yscale("symlog", linthresh=.001, linscale=.5)
        ax.set_ylim(-.00012, max(10, ymax*1.2))
        ax.set_yticks([0, .001, .01, .1, 1, 10])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        ax.set_xticks([0, 2, 4, 6, 8])
        ax.set_xlabel(r"Deployment budget $C$")
        ax.grid(axis="y", color="#DEDEDE", linewidth=.45, zorder=0)
    axes[0, 0].set_ylabel("Cost gap", labelpad=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(.5, -.02), columnspacing=1.4, handlelength=2.4,
               handletextpad=.5)
    fig.subplots_adjust(left=.09, right=.995, bottom=.30, top=.86, wspace=.16)
    save(fig, "theorem_bounds", bbox_inches="tight", pad_inches=.04)

    caps = sorted({c["cap"] for c in cases}, reverse=True)
    fig, axes = plt.subplots(1, len(caps), figsize=(178/25.4, 70/25.4), squeeze=False)
    styles = [(INK, "--", "s"), (TEAL, "-", "o"), (GRAY, "-.", "^")]
    for ax, cap in zip(axes.flat, caps):
        panel_cases = [c for c in cases if c["cap"] == cap]
        panel_values = [r["J_star"] for c in panel_cases for r in c["budgets"]]
        for k, case in enumerate(panel_cases):
            color, line, marker = styles[k % len(styles)]
            C = [r["budget"] for r in case["budgets"]]
            cost = [r["J_star"] for r in case["budgets"]]
            ax.plot(C, cost, color=color, linestyle=line, marker=marker, markersize=3.4,
                    markerfacecolor="white", drawstyle="steps-post")
            ax.text(C[-1]+.18, cost[-1], f"q={case['preview']}", color=color, va="center")
        ax.axvline(panel_cases[0]["kappa_H"], color="#929292", lw=.8)
        ax.text(panel_cases[0]["kappa_H"]+.15, .91,
                rf"$\kappa_H={panel_cases[0]['kappa_H']}$", transform=ax.get_xaxis_transform())
        ax.set_title(rf"$M_g={cap:g}$", loc="left")
        ax.set_xlabel(r"Deployment budget $C$")
        ax.set_xlim(-.2, max(C)+1.05)
        ax.set_xticks([0, 2, 4, 6, 8])
        ax.grid(axis="y", color="#DEDEDE", linewidth=.45)
        spread = max(panel_values)-min(panel_values)
        ax.set_ylim(min(panel_values)-.10*max(spread, .01), max(panel_values)+.18*max(spread, .01))
    axes[0, 0].set_ylabel(r"Optimal cost $J^\star(C)$")
    fig.subplots_adjust(left=.10, right=.97, bottom=.22, top=.90, wspace=.30)
    save(fig, "cost_vs_budget")


if __name__ == "__main__":
    plot_theorem()
