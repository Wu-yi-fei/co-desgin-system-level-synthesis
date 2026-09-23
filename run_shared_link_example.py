#!/usr/bin/env python3
"""Reproduce a small routed controller and graph codesign example.

All graph objectives and controller coefficients are computed with Fraction.
Floating point conic solves only supply relaxation diagnostics and supporting
hyperplanes. The reported lower bounds are independently evaluated in rational
arithmetic using a nonnegative LP multiplier and an explicit box correction.

Run from any directory with Python, numpy, scipy and cvxpy installed.
"""

from __future__ import annotations

import itertools
import json
import argparse
from fractions import Fraction as F
from pathlib import Path

if not __debug__:
    raise RuntimeError("Run without -O so the exact checks remain enabled")

OUT = Path(__file__).resolve().parent / "output"
# Original shared-path experiment: scalar dynamics, horizon, cap and budget.
HORIZON = 4
DECAY = F(4, 5)
COUPLING = F(2, 5)
EFFORT = F(1, 10)
CAP = F(2)
BUDGET = 3
# source, receiver, integer latency, deployment cost
EDGES = ((1, 2, 1, 1), (2, 3, 1, 1), (2, 4, 1, 1),
         (1, 3, 1, 2), (1, 4, 1, 2))


def solve_linear(matrix, rhs):
    """Gauss Jordan elimination over the rationals."""
    n = len(rhs)
    a = [list(matrix[i]) + [rhs[i]] for i in range(n)]
    for j in range(n):
        pivot = next(i for i in range(j, n) if a[i][j])
        a[j], a[pivot] = a[pivot], a[j]
        p = a[j][j]
        a[j] = [x / p for x in a[j]]
        for i in range(n):
            if i != j:
                p = a[i][j]
                a[i] = [x - p * y for x, y in zip(a[i], a[j])]
    return [row[-1] for row in a]


def response_rows():
    """Affine state responses y_t = b_t + D_t u for one receiver."""
    rows = [(F(0), [F(0)] * HORIZON)]
    for t in range(HORIZON):
        base, coefficients = rows[-1]
        nxt = [DECAY * value for value in coefficients]
        nxt[t] += 1
        rows.append((DECAY * base + COUPLING * DECAY**t, nxt))
    return rows


ROWS = response_rows()


def scalar_controller(arrival):
    """Exact unique minimizer for one observed scalar signal."""
    active = list(range(max(0, arrival), HORIZON))
    h = [[sum(d[i] * d[j] for _, d in ROWS)
          + (EFFORT if i == j else 0) for j in active] for i in active]
    v = [sum(b * d[i] for b, d in ROWS) for i in active]
    solution = solve_linear(h, [-x for x in v]) if active else []
    u = [F(0)] * HORIZON
    for i, value in zip(active, solution):
        u[i] = value
    y = [b + sum(a * c for a, c in zip(d, u)) for b, d in ROWS]
    cost = sum(x * x for x in y) + EFFORT * sum(x * x for x in u)
    assert all(abs(x) <= CAP for x in u)
    # Positive effort gives a positive definite Hessian. Exact stationarity,
    # zero unavailable coefficients and the checked cap certify optimality.
    for i in active:
        assert sum(d[i] * value for (_, d), value in zip(ROWS, y)) + EFFORT * u[i] == 0
    return {"objective": cost, "u": u, "state": y}


def graph_cost(graph):
    return sum(EDGES[e][3] for e in graph)


def distance(graph, sink):
    dist = {1: 0, 2: 99, 3: 99, 4: 99}
    for _ in range(4):
        for e in graph:
            source, receiver, latency, _ = EDGES[e]
            dist[receiver] = min(dist[receiver], dist[source] + latency)
    return dist[sink]


def all_graphs():
    return [tuple(i for i in range(len(EDGES)) if bits >> i & 1)
            for bits in range(1 << len(EDGES))
            if graph_cost(tuple(i for i in range(len(EDGES)) if bits >> i & 1)) <= BUDGET]


def evaluate(graph, lead=0, predicted_variance=F(0)):
    result = {"edges": list(graph), "deployment_cost": graph_cost(graph),
              "objective": F(0), "receivers": {}}
    for sink in (3, 4):
        delay = distance(graph, sink)
        residual = scalar_controller(delay)
        forecast = scalar_controller(delay - lead if delay < 99 else 99)
        forecast_correction = [u - v for u, v in zip(forecast["u"], residual["u"])]
        assert all(abs(value) <= CAP for value in forecast_correction)
        value = (1 - predicted_variance) * residual["objective"] + predicted_variance * forecast["objective"]
        result["objective"] += value
        result["receivers"][str(sink)] = {
            "arrival": None if delay == 99 else delay,
            "residual": residual,
            "forecast": forecast,
            "predictive_response_correction": forecast_correction,
        }
    return result


def rank(graph, lead=0, predicted_variance=F(0)):
    return evaluate(graph, lead, predicted_variance)["objective"], graph_cost(graph), tuple(graph)


def greedy():
    """Add the affordable edge with greatest exact objective improvement."""
    selected = ()
    history = [selected]
    while True:
        candidates = [tuple(sorted(selected + (e,))) for e in range(len(EDGES))
                      if e not in selected and graph_cost(selected) + EDGES[e][3] <= BUDGET]
        if not candidates:
            break
        best = min(candidates, key=rank)
        if rank(best)[0] >= rank(selected)[0]:
            break
        selected = best
        history.append(selected)
    return selected, history


def exchange(start):
    """Refit every affordable deletion, addition or one for one exchange."""
    selected = start
    history = [selected]
    while True:
        neighborhood = set()
        for remove in (None,) + selected:
            base = tuple(e for e in selected if e != remove)
            for add in (None,) + tuple(e for e in range(len(EDGES)) if e not in selected):
                candidate = tuple(sorted(base + (() if add is None else (add,))))
                if graph_cost(candidate) <= BUDGET:
                    neighborhood.add(candidate)
        best = min(neighborhood, key=rank)
        if rank(best)[0] >= rank(selected)[0]:
            break
        selected = best
        history.append(selected)
    return selected, history


def linear_relaxation_data(add_covers):
    """McCormick AND/OR recursion, response caps and optional joint covers.

    x = (u_3,1:3, u_4,1:3, z_0:4, b_3,b_4,a_3,a_4).
    b_i denotes arrival through the two hop route and a_i arrival by time 2.
    In this directly observed innovation realization only control response
    masks are imposed. State responses obey the plant identity exactly.
    """
    n = 15
    rows, rhs = [], []

    def add(entries, value):
        row = [F(0)] * n
        for index, coefficient in entries.items():
            row[index] = F(coefficient)
        rows.append(row)
        rhs.append(F(value))

    add({6 + e: edge[3] for e, edge in enumerate(EDGES)}, BUDGET)
    for sink in range(2):
        trunk, branch, direct, both, avail = 6, 7 + sink, 9 + sink, 11 + sink, 13 + sink
        add({both: 1, trunk: -1}, 0)
        add({both: 1, branch: -1}, 0)
        add({trunk: 1, branch: 1, both: -1}, 1)
        add({direct: 1, avail: -1}, 0)
        add({both: 1, avail: -1}, 0)
        add({avail: 1, direct: -1, both: -1}, 0)
        for time in range(3):
            u = 3 * sink + time
            available = direct if time == 0 else avail
            add({u: 1, available: -CAP}, 0)
            add({u: -1, available: -CAP}, 0)

    # Complete availability vectors of every budget feasible graph.
    demands = ((3, 1), (4, 1), (3, 2), (4, 2))
    indices = (9, 10, 13, 14)
    vectors = [tuple(int(distance(g, i) <= deadline) for i, deadline in demands)
               for g in all_graphs()]
    covers = []
    for size in range(2, len(demands) + 1):
        for subset in itertools.combinations(range(len(demands)), size):
            if any(all(v[i] for i in subset) for v in vectors):
                continue
            if any(set(old).issubset(subset) for old in covers):
                continue
            covers.append(subset)
            if add_covers:
                add({indices[i]: 1 for i in subset}, len(subset) - 1)
    bounds = [(-CAP, CAP)] * 6 + [(F(0), F(1))] * 9
    return rows, rhs, bounds, covers


def rational(value, denominator=10**8):
    return F(float(value)).limit_denominator(denominator)


def tangent(x, perspective):
    """Exact rational affine minorant of the full convex objective."""
    gradient = [F(0)] * 15
    value = F(0)
    for sink in range(2):
        u = [F(0)] + x[3 * sink:3 * sink + 3]
        for base, coefficients in ROWS:
            y = base + sum(a * b for a, b in zip(coefficients, u))
            value += y * y
            for time in range(3):
                gradient[3 * sink + time] += 2 * y * coefficients[time + 1]
        for time in range(3):
            index = 3 * sink + time
            if perspective:
                availability = 9 + sink if time == 0 else 13 + sink
                a = max(F(1, 10**6), x[availability])
                value += EFFORT * x[index]**2 / a
                gradient[index] += 2 * EFFORT * x[index] / a
                gradient[availability] -= EFFORT * x[index]**2 / a**2
            else:
                value += EFFORT * x[index]**2
                gradient[index] += 2 * EFFORT * x[index]
    # A clipped positive availability is the tangent base point.
    base_point = list(x)
    if perspective:
        for i in (9, 10, 13, 14):
            base_point[i] = max(F(1, 10**6), base_point[i])
    constant = value - sum(a * b for a, b in zip(gradient, base_point))
    return constant, gradient


def relaxation(perspective, add_covers):
    import cvxpy as cp
    import numpy as np
    from scipy.optimize import linprog

    rows, rhs, bounds, covers = linear_relaxation_data(add_covers)
    a = np.array(rows, dtype=float)
    b = np.array(rhs, dtype=float)
    x = cp.Variable(15)
    constraints = [a @ x <= b]
    for i, (lower, upper) in enumerate(bounds):
        constraints += [x[i] >= float(lower), x[i] <= float(upper)]
    objective = 0
    for sink in range(2):
        u = cp.hstack([0, x[3 * sink:3 * sink + 3]])
        for base, coefficients in ROWS:
            objective += cp.square(float(base) + np.array(coefficients, dtype=float) @ u)
        for time in range(3):
            index = 3 * sink + time
            if perspective:
                available = 9 + sink if time == 0 else 13 + sink
                objective += float(EFFORT) * cp.quad_over_lin(x[index], x[available])
            else:
                objective += float(EFFORT) * cp.square(x[index])
    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver="CLARABEL", tol_gap_abs=1e-10, tol_gap_rel=1e-10,
                  tol_feas=1e-10, max_iter=500)
    assert problem.status in ("optimal", "optimal_inaccurate")
    primal = [rational(v) for v in x.value]
    constant, gradient = tangent(primal, perspective)
    lp = linprog(np.array(gradient, dtype=float), A_ub=a, b_ub=b,
                 bounds=[tuple(map(float, pair)) for pair in bounds], method="highs")
    assert lp.success
    # For Ax <= b, lambda >= 0, c*x >= -lambda*b +
    # min_box (c + A^T lambda)*x. No stationarity tolerance is assumed.
    multipliers = [max(F(0), rational(-v)) for v in lp.ineqlin.marginals]
    residual = [gradient[j] + sum(lam * row[j] for lam, row in zip(multipliers, rows))
                for j in range(15)]
    box_correction = sum(min(r * lower, r * upper)
                         for r, (lower, upper) in zip(residual, bounds))
    lower = constant - sum(lam * bound for lam, bound in zip(multipliers, rhs)) + box_correction
    assert all(lam >= 0 for lam in multipliers)
    return {
        "primal_objective_estimate": float(problem.value),
        "verified_lower_bound": lower,
        "maximum_linear_residual": float(max(0.0, np.max(a @ x.value - b))),
        "z_estimate": list(map(float, x.value[6:11])),
        "minimal_joint_covers": [list(c) for c in covers] if add_covers else [],
        "rational_certificate": {
            "tangent_base": primal,
            "tangent_constant": constant,
            "tangent_gradient": gradient,
            "nonnegative_multipliers": multipliers,
            "stationarity_residual": residual,
            "box_correction": box_correction,
        },
    }


def encode(obj):
    if isinstance(obj, F):
        return {"rational": str(obj), "decimal": float(obj)}
    raise TypeError(type(obj).__name__)


def decode(obj):
    if isinstance(obj, dict):
        if set(obj) == {"rational", "decimal"}:
            exact = F(obj["rational"])
            assert float(exact) == obj["decimal"]
            return exact
        return {key: decode(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [decode(value) for value in obj]
    return obj


def verify_saved():
    """Check stored certificates using only Python standard library."""
    result = decode(json.loads((OUT / "shared_link_example.json").read_text()))
    graphs = all_graphs()
    assert result["enumerated_graph_count"] == len(graphs)
    assert result["graph_results"] == [evaluate(g) for g in graphs]
    best = min(graphs, key=rank)
    assert result["exact_best"] == evaluate(best)
    for name, row in result["relaxations"].items():
        perspective = name != "plain"
        use_covers = name == "perspective_joint_covers"
        matrix, rhs, bounds, covers = linear_relaxation_data(use_covers)
        certificate = row["rational_certificate"]
        constant, gradient = tangent(certificate["tangent_base"], perspective)
        assert constant == certificate["tangent_constant"]
        assert gradient == certificate["tangent_gradient"]
        multipliers = certificate["nonnegative_multipliers"]
        assert len(multipliers) == len(matrix)
        assert all(value >= 0 for value in multipliers)
        residual = [gradient[j] + sum(lam * a[j] for lam, a in zip(multipliers, matrix))
                    for j in range(15)]
        correction = sum(min(r * lower, r * upper)
                         for r, (lower, upper) in zip(residual, bounds))
        value = constant - sum(lam * b for lam, b in zip(multipliers, rhs)) + correction
        assert residual == certificate["stationarity_residual"]
        assert correction == certificate["box_correction"]
        assert value == row["verified_lower_bound"] <= result["exact_best"]["objective"]
        assert row["minimal_joint_covers"] == ([list(c) for c in covers] if use_covers else [])
    for row in result["optional_prediction"]:
        lead = row["lead"]
        best = min(graphs, key=lambda g: rank(g, lead, F(3, 4)))
        assert row["best"] == evaluate(best, lead, F(3, 4))
    assert result["greedy_refit"] == evaluate(greedy()[0])
    assert result["one_exchange_refit"] == evaluate(exchange(greedy()[0])[0])
    witness = result["non_submodular_control_benefit"]
    empty_cost = rank(())[0]
    assert witness == {"empty_cost": empty_cost,
                       "trunk_only_cost": rank((0,))[0],
                       "branch_only_cost": rank((1,))[0],
                       "trunk_and_branch_cost": rank((0, 1))[0],
                       "single_edge_benefit": F(0),
                       "pair_benefit": empty_cost - rank((0, 1))[0]}
    assert witness["trunk_only_cost"] == witness["branch_only_cost"] == empty_cost
    assert witness["pair_benefit"] > 0
    print("Verified every graph objective, stationary controller, response cap, joint cover and rational lower bound.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true",
                        help="check saved rational certificates without numerical packages")
    args = parser.parse_args()
    if args.verify:
        verify_saved()
        return
    graphs = all_graphs()
    evaluated = [evaluate(g) for g in graphs]
    best = min(graphs, key=rank)
    greedy_graph, greedy_history = greedy()
    exchange_graph, exchange_history = exchange(greedy_graph)
    rows = {
        "plain": relaxation(False, False),
        "perspective": relaxation(True, False),
        "perspective_joint_covers": relaxation(True, True),
    }
    best_cost = rank(best)[0]
    for row in rows.values():
        assert row["verified_lower_bound"] <= best_cost
    predictions = []
    for lead in (0, 1, 2):
        # xi = p + eta, independent variances 3/4 and 1/4. To keep all
        # absolute releases nonnegative, event s=2 follows two idle samples.
        # Forecast p is released at node 1 at 2-lead, xi at 2. Controls are
        # zero before 2. ROWS uses relative time t-2 for every forecast case.
        selected = min(graphs, key=lambda g: rank(g, lead, F(3, 4)))
        predictions.append({"lead": lead, "forecast_variance": F(3, 4),
                            "residual_variance": F(1, 4),
                            "event_time": 2, "forecast_release": 2 - lead,
                            "control_start": 2, "absolute_horizon": 2 + HORIZON,
                            "best": evaluate(selected, lead, F(3, 4))})
    result = {
        "information_model": "directly observed innovation with lossless multihop forwarding",
        "controller_measurements": "local innovations and delivered messages only; controlled states are performance outputs",
        "plant": {"decay": DECAY, "source_coupling": COUPLING,
                  "horizon": HORIZON, "effort_weight": EFFORT, "response_cap": CAP},
        "edges_source_receiver_latency_cost": EDGES,
        "budget": BUDGET,
        "enumerated_graph_count": len(graphs),
        "graph_results": evaluated,
        "exact_best": evaluate(best),
        "greedy_refit": evaluate(greedy_graph),
        "greedy_history": greedy_history,
        "one_exchange_refit": evaluate(exchange_graph),
        "one_exchange_history": exchange_history,
        "non_submodular_control_benefit": {
            "empty_cost": rank(())[0],
            "trunk_only_cost": rank((0,))[0],
            "branch_only_cost": rank((1,))[0],
            "trunk_and_branch_cost": rank((0, 1))[0],
            "single_edge_benefit": F(0),
            "pair_benefit": rank(())[0] - rank((0, 1))[0],
        },
        "relaxations": rows,
        "optional_prediction": predictions,
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "shared_link_example.json").write_text(json.dumps(result, default=encode, indent=2) + "\n")
    print(json.dumps({"budget_feasible_graphs": len(graphs),
                      "exact_best": {"edges": best, "objective": best_cost},
                      "greedy_objective": rank(greedy_graph)[0],
                      "exchange_objective": rank(exchange_graph)[0],
                      "relaxations": {name: {"primal_objective_estimate": row["primal_objective_estimate"],
                                              "verified_lower_bound_decimal": float(row["verified_lower_bound"])}
                                      for name, row in rows.items()},
                      "optional_prediction_costs": [r["best"]["objective"] for r in predictions]},
                     default=encode, indent=2))
    verify_saved()


if __name__ == "__main__":
    main()
