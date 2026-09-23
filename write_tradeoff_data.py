#!/usr/bin/env python3
"""Load checked voltage results and write the data used by the Python figures.

Run ``run_voltage_codesign.py --verify-recorded-study`` after changing the
experiment records. No LaTeX or third-party package is required here.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "output"
METHODS = {
    "Greedy": "Greedy refit",
    "Perspective": "Perspective refit",
    "Response": "Magnitude refit",
    "Sensitivity": "Sensitivity refit",
    "Distance": "Distance refit",
}
MODES = {"Baseline": "No forecast", "Residual": "Forecast residual"}
BUDGETS = [4, 8, 12, 16, 24]


def load_data():
    """Return checked objectives and normalized gaps, including both endpoints."""
    archived = json.loads((RESULTS / "archived_codesign_records.json").read_text())
    verified = json.loads((RESULTS / "voltage_record_verification.json").read_text())
    if verified["model"] != "directly_observed_innovations":
        raise ValueError("The plot records use a different information model")
    rows = {(row["mode"], row["method"], row["budget"]): row
            for row in verified["records"]}
    if len(rows) != 50 or len(rows) != len(verified["records"]):
        raise ValueError("Expected exactly fifty independently refitted graph records")
    plot_rows = []
    for mode in MODES.values():
        references = {row["method"]: row["objective"] for row in archived["records"]
                      if row["mode"] == mode and row["method"] in {"Local", "Full"}}
        denominator = references["Local"] - references["Full"]
        if denominator <= 0:
            raise ValueError("The local-to-full reference gap must be positive")
        for method in METHODS.values():
            values = [(0, references["Local"], 1.0)]
            for budget in BUDGETS:
                row = rows[mode, method, budget]
                source = next(record for record in archived["records"]
                              if (record["mode"], record["method"], record["budget"])
                              == (mode, method, budget))
                if (row["edges"] != source["edges"] or len(row["edges"]) != budget
                        or abs(row["refit_objective"] - source["objective"]) > 1e-8):
                    raise ValueError("The refitted graph or objective disagrees with its archive")
                gap = (row["refit_objective"] - references["Full"]) / denominator
                if not 0 <= gap <= 1:
                    raise ValueError("The plotted normalized cost is outside the reference range")
                values.append((budget, row["refit_objective"], gap))
            values.append((56, references["Full"], 0.0))
            for budget, objective, gap in values:
                plot_rows.append({
                    "mode": mode,
                    "method": method,
                    "budget": budget,
                    "objective": objective,
                    "local_objective": references["Local"],
                    "full_objective": references["Full"],
                    "normalized_gap": gap,
                })
    return plot_rows


def main():
    rows = load_data()
    output = RESULTS / "tradeoff_plot_data.csv"
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output)
    return rows


if __name__ == "__main__":
    main()
