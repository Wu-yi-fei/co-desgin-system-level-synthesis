#!/usr/bin/env python3
"""Verify output/voltage_certificate_trials.json using exact rational arithmetic.

Run python verify_voltage_certificates.py. Only the standard library is required.
The instances use direct local disturbance measurements and direct links.
No solver objective or saved certificate endpoint is trusted as a bound.
"""
from __future__ import annotations

import argparse
import itertools
import json
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, localcontext
from fractions import Fraction
from pathlib import Path

OUT = Path(__file__).resolve().parent / "output"

def rational(value):
    """Interpret JSON binary64 numbers exactly, or decimal strings as exact decimals."""
    return Fraction.from_float(value) if isinstance(value, float) else Fraction(value)

def fraction_fixed_bound(value: Fraction, places: int, rounding: str) -> str:
    """Format an exact rational with directed decimal rounding."""
    with localcontext() as context:
        context.prec = 120
        decimal_value = Decimal(value.numerator) / Decimal(value.denominator)
        quantum = Decimal(1).scaleb(-places)
        rounded = decimal_value.quantize(quantum, rounding=rounding)
    return format(rounded, f".{places}f")


def fraction_scientific_bound(value: Fraction, digits: int, rounding: str) -> str:
    """Format a positive exact rational as a directed scientific bound."""
    if value <= 0 or digits < 1:
        raise ValueError("Scientific certificate bounds must be positive")
    with localcontext() as context:
        context.prec = 120
        decimal_value = Decimal(value.numerator) / Decimal(value.denominator)
        exponent = decimal_value.adjusted()
        quantum = Decimal(1).scaleb(exponent - digits + 1)
        rounded = decimal_value.quantize(quantum, rounding=rounding)
        exponent = rounded.adjusted()
        mantissa = rounded.scaleb(-exponent)
    return f"{mantissa:.{digits - 1}f}e{exponent:+d}"


def exact_matrix(values):
    return [[rational(value) for value in row] for row in values]


def exact_zero_matrix(rows: int, columns: int) -> list[list[Fraction]]:
    return [[Fraction(0) for _ in range(columns)] for _ in range(rows)]


def exact_identity(size: int) -> list[list[Fraction]]:
    result = exact_zero_matrix(size, size)
    for index in range(size):
        result[index][index] = Fraction(1)
    return result


def exact_matmul(
    left: list[list[Fraction]], right: list[list[Fraction]]
) -> list[list[Fraction]]:
    rows = len(left)
    inner = len(right)
    columns = len(right[0])
    if not left or not right or len(left[0]) != inner:
        raise ValueError("Incompatible exact matrix product")
    return [
        [sum((left[i][k] * right[k][j] for k in range(inner)), Fraction(0)) for j in range(columns)]
        for i in range(rows)
    ]


def exact_transpose_matmul(
    left: list[list[Fraction]], right: list[list[Fraction]]
) -> list[list[Fraction]]:
    rows = len(left[0])
    inner = len(left)
    columns = len(right[0])
    if not left or not right or len(right) != inner:
        raise ValueError("Incompatible exact transposed matrix product")
    return [
        [sum((left[k][i] * right[k][j] for k in range(inner)), Fraction(0)) for j in range(columns)]
        for i in range(rows)
    ]


def exact_matrix_add(
    left: list[list[Fraction]], right: list[list[Fraction]]
) -> list[list[Fraction]]:
    if len(left) != len(right) or len(left[0]) != len(right[0]):
        raise ValueError("Incompatible exact matrix sum")
    return [
        [left[i][j] + right[i][j] for j in range(len(left[0]))]
        for i in range(len(left))
    ]


def exact_weighted_matrix(
    matrix: list[list[Fraction]], factor: Fraction, sigma_squared: list[Fraction]
) -> list[list[Fraction]]:
    return [
        [factor * matrix[i][j] * sigma_squared[j] for j in range(len(matrix[0]))]
        for i in range(len(matrix))
    ]


def exact_fixed_graph_certificate(
    a_mat: np.ndarray,
    b_mat: np.ndarray,
    horizon: int,
    sigma: np.ndarray,
    active_edges: set[tuple[int, int]],
    candidate_edges: list[tuple[int, int]],
    response: dict,
) -> dict:
    """Verify one masked quadratic graph problem with exact rational arithmetic."""
    n = len(a_mat)
    candidate_set = set(candidate_edges)
    if not active_edges.issubset(candidate_set):
        raise ValueError("Certificate graph is outside the candidate set")

    coordinates: list[tuple[int, int, int, int]] = []
    coordinates_by_block: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for t in range(horizon):
        for s in range(t + 1):
            for i in range(n):
                for j in range(n):
                    if i == j or (s < t and (i, j) in active_edges):
                        index = len(coordinates)
                        coordinates.append((t, s, i, j))
                        coordinates_by_block.setdefault((t, s), []).append((index, i, j))

    exact_u: dict[tuple[int, int], list[list[Fraction]]] = {}
    coordinate_values = [Fraction(0) for _ in coordinates]
    for t in range(horizon):
        for s in range(t + 1):
            matrix = exact_zero_matrix(n, n)
            approximate = response["U"][f"{t},{s}"]
            if len(approximate) != n or any(len(row) != n for row in approximate):
                raise ValueError("Certificate response contains a nonfinite block")
            for index, i, j in coordinates_by_block.get((t, s), []):
                value = rational(approximate[i][j])
                coordinate_values[index] = value
                matrix[i][j] = value
            exact_u[(t, s)] = matrix

    exact_a = exact_matrix(a_mat)
    exact_b = exact_matrix(b_mat)
    exact_sigma = [rational(value) for value in sigma]
    if any(value <= 0 for value in exact_sigma):
        raise ValueError("Certificate standard deviations must be positive")
    sigma_squared = [value * value for value in exact_sigma]
    q_weight = Fraction(22)
    r_weight = Fraction(35, 100)

    exact_x: dict[tuple[int, int], list[list[Fraction]]] = {}
    for s in range(horizon + 1):
        exact_x[(s, s)] = exact_identity(n)
        for t in range(s, horizon):
            exact_x[(t + 1, s)] = exact_matrix_add(
                exact_matmul(exact_a, exact_x[(t, s)]),
                exact_matmul(exact_b, exact_u[(t, s)]),
            )

    objective = Fraction(0)
    for matrix in exact_x.values():
        objective += q_weight * sum(
            (matrix[i][j] * matrix[i][j] * sigma_squared[j] for i in range(n) for j in range(n)),
            Fraction(0),
        )
    for matrix in exact_u.values():
        objective += r_weight * sum(
            (matrix[i][j] * matrix[i][j] * sigma_squared[j] for i in range(n) for j in range(n)),
            Fraction(0),
        )

    gradient = [Fraction(0) for _ in coordinates]
    for s in range(horizon):
        costate = exact_weighted_matrix(exact_x[(horizon, s)], 2 * q_weight, sigma_squared)
        for t in range(horizon - 1, s - 1, -1):
            block_gradient = exact_matrix_add(
                exact_weighted_matrix(exact_u[(t, s)], 2 * r_weight, sigma_squared),
                exact_transpose_matmul(exact_b, costate),
            )
            for index, i, j in coordinates_by_block.get((t, s), []):
                gradient[index] = block_gradient[i][j]
            costate = exact_matrix_add(
                exact_weighted_matrix(exact_x[(t, s)], 2 * q_weight, sigma_squared),
                exact_transpose_matmul(exact_a, costate),
            )

    gradient_squared = sum((value * value for value in gradient), Fraction(0))
    alpha = r_weight * min(sigma_squared)
    if alpha <= 0:
        raise ValueError("Certificate curvature must be positive")
    correction = gradient_squared / (4 * alpha)
    lower_bound = objective - correction

    group_squared = {
        edge: sum(
            (exact_u[(t, s)][edge[0]][edge[1]] ** 2 for t in range(horizon) for s in range(t)),
            Fraction(0),
        )
        for edge in active_edges
    }
    max_group_squared = max(group_squared.values(), default=Fraction(0))
    return {
        "graph": frozenset(active_edges),
        "free_coordinates": len(coordinates),
        "objective": objective,
        "gradient_squared": gradient_squared,
        "alpha": alpha,
        "correction": correction,
        "lower_bound": lower_bound,
        "max_group_squared": max_group_squared,
    }


def verify_instance(instance):
    a, b, sigma = instance["A"], instance["B"], instance["sigma"]
    n, horizon = instance["nodes"], instance["horizon"]
    if (n,horizon) not in {(5,3),(8,5)}:
        raise ValueError("Incorrect archived node count or horizon")
    if len(a) != n or len(b) != n or len(sigma) != n:
        raise ValueError("Incorrect archived instance dimensions")
    candidates = [tuple(edge) for edge in instance["candidate_edges"]]
    expected = ([(i, j) for i in range(n) for j in range(n) if i != j]
                if n == 8 else [edge for i in range(4) for edge in [(i, i + 1), (i + 1, i)]])
    if candidates != expected:
        raise ValueError("Archived candidate set differs from the manuscript")
    certificates = []
    for trial in instance["trials"]:
        graph = {tuple(edge) for edge in trial["edges"]}
        certificate = exact_fixed_graph_certificate(a, b, horizon, sigma, graph, candidates, trial)
        if trial.get("require_group_feasible", True) and certificate["max_group_squared"] > 25:
            raise ValueError("An exact response violates its group ball")
        certificates.append(certificate)
    return certificates

def verify(archive_path=OUT / "voltage_certificate_trials.json"):
    scalar = exact_fixed_graph_certificate([[Fraction(1,2)]],[[Fraction(1)]],1,
                                           [Fraction(1)],set(),[],
                                           {"U":{"0,0":[[Fraction(-1,4)]]}})
    if (scalar["objective"] != Fraction(14527,320)
            or scalar["gradient_squared"] != Fraction(433,40)**2
            or scalar["alpha"] != Fraction(7,20)):
        raise AssertionError("Scalar objective and adjoint-gradient self-test failed")
    archive = json.loads(archive_path.read_text())
    if archive["schema_version"] != 1 or archive["model"] != "directly_observed_innovations":
        raise ValueError("Unsupported certificate model or schema")
    if (archive["state_weight"] != 22 or Fraction(archive["input_weight"]) != Fraction(7,20)
            or archive["response_group_bound"] != 5):
        raise ValueError("Archived objective weights or group radius differ from the manuscript")
    small_instance, large_instance = archive["five_bus"], archive["eight_bus"]
    small = verify_instance(small_instance)
    candidates = [tuple(edge) for edge in small_instance["candidate_edges"]]
    expected_graphs = {frozenset(edges) for size in range(4)
                       for edges in itertools.combinations(candidates, size)}
    if len(small) != 93 or {c["graph"] for c in small} != expected_graphs:
        raise ValueError("The archive does not contain exactly the 93 admissible graphs")
    if any(c["max_group_squared"] > 25 for c in small):
        raise ValueError("A small-instance witness violates a declared group ball")
    incumbent_graph = frozenset(tuple(edge) for edge in small_instance["incumbent_edges"])
    incumbent = next(c for c in small if c["graph"] == incumbent_graph)
    small_lower = min(c["lower_bound"] for c in small)
    small_upper = incumbent["objective"]
    margins = []
    for edge in candidates:
        opposite = min(c["lower_bound"] for c in small
                       if (edge in c["graph"]) != (edge in incumbent_graph))
        margins.append(opposite - small_upper)
    if len(incumbent_graph) != 3 or min(margins) <= 0:
        raise ValueError("Unique optimal three-link topology is not certified")

    large = verify_instance(large_instance)
    if len(large) != 2:
        raise ValueError("Eight bus certificate needs one full and one selected trial")
    full, selected = large
    if len(full["graph"]) != 56 or len(selected["graph"]) != 12:
        raise ValueError("Incorrect full or selected graph")
    if full["free_coordinates"] != 680 or selected["free_coordinates"] != 240:
        raise ValueError("Incorrect number of free response coordinates")
    if selected["max_group_squared"] > 25:
        raise ValueError("The selected eight-bus witness violates a declared group ball")
    lower, upper = full["lower_bound"], selected["objective"]
    if min(small_lower, lower) <= 0 or small_upper < small_lower or upper < lower:
        raise ValueError("Invalid positive objective interval")

    summary = {
        "arithmetic": "exact rational arithmetic without an optimization solver",
        "model": archive["model"],
        "five_bus": {
            "graphs": len(small), "certified_link_decisions": len(candidates),
            "unique_optimal_graph": [list(edge) for edge in sorted(incumbent_graph)],
            "lower_floor": fraction_fixed_bound(small_lower, 16, ROUND_FLOOR),
            "upper_ceiling": fraction_fixed_bound(small_upper, 16, ROUND_CEILING),
            "gap_upper": fraction_scientific_bound(small_upper-small_lower, 3, ROUND_CEILING) if small_upper>small_lower else "0",
            "minimum_opposite_branch_margin_lower": fraction_scientific_bound(min(margins), 3, ROUND_FLOOR),
        },
        "eight_bus": {
            "full_free_coordinates": full["free_coordinates"],
            "selected_free_coordinates": selected["free_coordinates"],
            "selected_method": "Perspective refit",
            "lower_floor": fraction_fixed_bound(lower, 16, ROUND_FLOOR),
            "upper_ceiling": fraction_fixed_bound(upper, 16, ROUND_CEILING),
            "relative_gap_percent_upper": fraction_fixed_bound(100*(upper-lower)/upper, 3, ROUND_CEILING),
            "residual_correction_upper": fraction_scientific_bound(full["correction"], 3, ROUND_CEILING) if full["correction"] else "0",
            "certifies_graph_uniqueness": False,
        },
    }
    return summary

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, default=OUT / "voltage_certificate_trials.json")
    parser.add_argument("--output", type=Path, help="optional verification report path")
    args = parser.parse_args()
    summary = verify(args.trials)
    rendered = json.dumps(summary, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")

if __name__ == "__main__":
    main()
