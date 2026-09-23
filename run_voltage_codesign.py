#!/usr/bin/env python3
"""Run the original voltage SLS experiments with direct disturbance measurements.

Default: solve 93 five-bus graphs and two eight-bus certificate trials,
then verify the resulting objective bounds using exact rational arithmetic.
The controller uses disturbance responses only and direct unit-delay links.
The two covariance cases do not introduce explicit prediction responses.
All data are read from and written to output/ beside this script.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent / "output"
# Original voltage experiment: response cap and quadratic cost weights.
RESPONSE_GROUP_BOUND = 5.0
STATE_RESPONSE_WEIGHT = 22.0
CONTROL_RESPONSE_WEIGHT = 0.35
FIXED_GRAPH_ABSOLUTE_TOLERANCE = 1e-7
SOLVER_OBJECTIVE_CONSISTENCY_TOLERANCE = 1e-8
GREEDY_TIE_TOLERANCE_MULTIPLIER = 10.0

def dependencies():
    global cp, np
    import cvxpy as cp
    import numpy as np

def feeder_model(n: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a stable voltage model and a radial physical graph."""
    if n != 8:
        parent = np.arange(-1, n - 1)
        reactance = np.linspace(0.018, 0.032, n)
    else:
        parent = np.array([-1, 0, 0, 1, 1, 2, 2, 5], dtype=int)
        reactance = np.array([0.020, 0.024, 0.018, 0.030, 0.022, 0.027, 0.025, 0.031])

    paths: list[set[int]] = []
    for i in range(n):
        path: set[int] = set()
        node = i
        while node >= 0:
            path.add(node)
            node = int(parent[node])
        paths.append(path)

    sensitivity = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            common = paths[i].intersection(paths[j])
            sensitivity[i, j] = 2.0 * sum(reactance[k] for k in common)

    sensitivity /= np.linalg.norm(sensitivity, 2)
    adjacency = np.zeros((n, n))
    for child, par in enumerate(parent):
        if par >= 0:
            adjacency[child, par] = 1.0
            adjacency[par, child] = 1.0
    degree = np.maximum(adjacency.sum(axis=1), 1.0)
    coupling = adjacency / np.sqrt(np.outer(degree, degree))
    a_mat = 0.80 * np.eye(n) + 0.035 * coupling
    b_mat = -0.31 * sensitivity
    return a_mat, b_mat, parent, sensitivity


def response_problem(
    a_mat: np.ndarray,
    b_mat: np.ndarray,
    horizon: int,
    sigma: np.ndarray,
    active_edges: set[tuple[int, int]] | None = None,
    perspective_budget: int | None = None,
    candidate_edges: list[tuple[int, int]] | None = None,
    group_bound: float | None = RESPONSE_GROUP_BOUND,
    return_response: bool = False,
) -> dict:
    """Solve a direct innovation fixed topology problem or its relaxation."""
    n = a_mat.shape[0]
    if candidate_edges is None:
        candidate_edges = [(i, j) for i in range(n) for j in range(n) if i != j]
    candidate_set = set(candidate_edges)
    if active_edges is not None and not active_edges.issubset(candidate_set):
        raise ValueError("Active edges must belong to the candidate graph")
    if group_bound is not None and (not np.isfinite(group_bound) or group_bound <= 0.0):
        raise ValueError("Response group bound must be finite and positive")
    if perspective_budget is not None and group_bound is None:
        raise ValueError("Perspective relaxation requires a response group bound")

    u_blocks = {(t, s): cp.Variable((n, n)) for t in range(horizon) for s in range(t + 1)}

    def delayed_cross_group(i: int, j: int) -> cp.Expression:
        entries = [u_blocks[(t, s)][i, j] for t in range(horizon) for s in range(t)]
        return cp.hstack(entries) if entries else cp.Constant(np.zeros(1))

    x_blocks: dict[tuple[int, int], cp.Expression] = {}
    for s in range(horizon + 1):
        x_blocks[(s, s)] = np.eye(n)
        for t in range(s, horizon):
            x_blocks[(t + 1, s)] = a_mat @ x_blocks[(t, s)] + b_mat @ u_blocks[(t, s)]

    q_weight = STATE_RESPONSE_WEIGHT
    r_weight = CONTROL_RESPONSE_WEIGHT
    sigma_sqrt = np.diag(sigma)
    state_cost = 0
    for (t, s), x_expr in x_blocks.items():
        state_cost += q_weight * cp.sum_squares(x_expr @ sigma_sqrt)

    constraints: list[cp.Constraint] = []
    for t in range(horizon):
        for i in range(n):
            for j in range(n):
                if i != j:
                    constraints.append(u_blocks[(t, t)][i, j] == 0)
    for i in range(n):
        for j in range(n):
            if i != j and (i, j) not in candidate_set:
                constraints.append(delayed_cross_group(i, j) == 0)
    link_values: dict[tuple[int, int], cp.Variable] = {}
    tau_values: dict[tuple[int, int], cp.Variable] = {}

    self_control_cost = 0
    for i in range(n):
        entries = [sigma[i] * u_blocks[(t, s)][i, i] for t in range(horizon) for s in range(t + 1)]
        self_control_cost += r_weight * cp.sum_squares(cp.hstack(entries))

    if perspective_budget is not None:
        cross_control_cost = 0
        for edge in candidate_edges:
            i, j = edge
            raw = delayed_cross_group(i, j)
            weighted = np.sqrt(r_weight) * sigma[j] * raw
            z_var = cp.Variable(nonneg=True, name=f"z_{i}_{j}")
            tau = cp.Variable(nonneg=True, name=f"tau_{i}_{j}")
            link_values[edge] = z_var
            tau_values[edge] = tau
            constraints.extend(
                [
                    z_var <= 1.0,
                    cp.norm(raw, 2) <= group_bound * z_var,
                    cp.SOC(tau + z_var, cp.hstack([2.0 * weighted, tau - z_var])),
                ]
            )
            cross_control_cost += tau
        constraints.append(cp.sum(cp.hstack(list(link_values.values()))) <= perspective_budget)
        objective = cp.Minimize(state_cost + self_control_cost + cross_control_cost)
        solver = cp.ECOS
        solver_options = {
            "abstol": 1e-9,
            "reltol": 1e-9,
            "feastol": 1e-9,
            "max_iters": 2000,
        }
    else:
        if active_edges is None:
            active_edges = set(candidate_edges)
        for i, j in candidate_edges:
            raw = delayed_cross_group(i, j)
            if (i, j) not in active_edges:
                constraints.append(raw == 0)
            elif group_bound is not None:
                constraints.append(cp.norm(raw, 2) <= group_bound)
        control_cost = 0
        for u_expr in u_blocks.values():
            control_cost += r_weight * cp.sum_squares(u_expr @ sigma_sqrt)
        objective = cp.Minimize(state_cost + control_cost)
        solver = cp.CLARABEL
        solver_options = {
            "tol_gap_abs": FIXED_GRAPH_ABSOLUTE_TOLERANCE,
            "tol_feas": FIXED_GRAPH_ABSOLUTE_TOLERANCE,
            "max_iter": 500,
        }

    problem = cp.Problem(objective, constraints)
    value = problem.solve(solver=solver, verbose=False, **solver_options)
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"Codesign solve failed with status {problem.status}")

    off_diagonal = ~np.eye(n, dtype=bool)
    max_same_time_cross_response = max(
        float(np.max(np.abs(np.asarray(u_blocks[(t, t)].value)[off_diagonal])))
        for t in range(horizon)
    )
    if max_same_time_cross_response > 1e-8:
        raise RuntimeError("Unit latency regression failed")

    delayed_group_norms = []
    for i, j in candidate_edges:
        entries = [u_blocks[(t, s)].value[i, j] for t in range(horizon) for s in range(t)]
        delayed_group_norms.append(float(np.linalg.norm(entries)) if entries else 0.0)
    max_delayed_group_norm = max(delayed_group_norms, default=0.0)
    if not np.isfinite(max_delayed_group_norm):
        raise RuntimeError("Delayed response group norm is not finite")
    if group_bound is not None and max_delayed_group_norm > group_bound + 1e-6:
        raise RuntimeError("Delayed response group bound regression failed")

    result: dict = {
        "objective": float(value),
        "status": problem.status,
        "max_same_time_cross_response": max_same_time_cross_response,
        "max_delayed_group_norm": max_delayed_group_norm,
    }
    if perspective_budget is not None:
        solver_info = problem.solver_stats.extra_stats["info"]
        direct_constant = q_weight * (horizon + 1) * float(np.sum(sigma**2))
        ecos_primal_objective_estimate = float(solver_info["pcost"] + direct_constant)
        ecos_dual_objective_estimate = float(solver_info["dcost"] + direct_constant)
        objective_scale = max(
            1.0,
            abs(float(value)),
            abs(ecos_primal_objective_estimate),
            abs(ecos_dual_objective_estimate),
        )
        objective_tolerance = SOLVER_OBJECTIVE_CONSISTENCY_TOLERANCE * objective_scale
        objective_values = [
            float(value),
            ecos_primal_objective_estimate,
            ecos_dual_objective_estimate,
            objective_tolerance,
        ]
        if not all(np.isfinite(item) for item in objective_values):
            raise RuntimeError("ECOS returned a nonfinite objective diagnostic")
        if ecos_dual_objective_estimate > ecos_primal_objective_estimate + objective_tolerance:
            raise RuntimeError("ECOS dual objective estimate exceeds its primal objective beyond tolerance")
        if abs(ecos_primal_objective_estimate - float(value)) > objective_tolerance:
            raise RuntimeError("Restored ECOS primal objective disagrees with the CVXPY objective")
        solver_primal_residual = float(solver_info["pres"])
        solver_dual_residual = float(solver_info["dres"])
        solver_duality_gap = float(solver_info["gap"])
        if not all(
            np.isfinite(item)
            for item in [solver_primal_residual, solver_dual_residual, solver_duality_gap]
        ):
            raise RuntimeError("ECOS returned a nonfinite residual diagnostic")
        result["ecos_primal_objective_estimate"] = ecos_primal_objective_estimate
        result["ecos_dual_objective_estimate"] = ecos_dual_objective_estimate
        result["solver_objective_consistency_tolerance"] = objective_tolerance
        result["solver_primal_residual"] = solver_primal_residual
        result["solver_dual_residual"] = solver_dual_residual
        result["solver_duality_gap"] = solver_duality_gap
        result["z"] = {f"{i},{j}": float(link_values[(i, j)].value) for i, j in candidate_edges}
    if return_response:
        result["U"] = {f"{t},{s}": np.asarray(u.value) for (t, s), u in u_blocks.items()}
        result["X"] = {
            f"{t},{s}": np.asarray(x.value if hasattr(x, "value") else x)
            for (t, s), x in x_blocks.items()
        }
    return result


def edge_scores_from_response(response: dict, n: int, horizon: int) -> dict[tuple[int, int], float]:
    scores: dict[tuple[int, int], float] = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            vals = [response["U"][f"{t},{s}"][i, j] for t in range(horizon) for s in range(t)]
            scores[(i, j)] = float(np.linalg.norm(vals))
    return scores


def select_top(scores: dict[tuple[int, int], float], budget: int) -> set[tuple[int, int]]:
    ordered = sorted(scores, key=lambda edge: (scores[edge], -edge[0], -edge[1]), reverse=True)
    return set(ordered[:budget])


def greedy_marginal_refits(
    a_mat: np.ndarray,
    b_mat: np.ndarray,
    horizon: int,
    sigma: np.ndarray,
    candidate_edges: list[tuple[int, int]],
    initial_objective: float,
    checkpoint_budgets: list[int],
    retained_budget: int | None = None,
    progress_label: str = "",
) -> tuple[list[dict], dict, dict | None]:
    """Build one deterministic forward greedy graph sequence."""
    ordered_edges = sorted(candidate_edges)
    if len(ordered_edges) != len(set(ordered_edges)):
        raise ValueError("Greedy candidate edges must be unique")
    if not checkpoint_budgets:
        raise ValueError("Greedy refit requires at least one checkpoint")
    maximum_budget = max(checkpoint_budgets)
    minimum_budget = min(checkpoint_budgets)
    if minimum_budget < 1:
        raise ValueError("Greedy checkpoints must be positive")
    if maximum_budget > len(ordered_edges):
        raise ValueError("Greedy budget exceeds the candidate graph")
    if retained_budget is not None and retained_budget not in checkpoint_budgets:
        raise ValueError("Retained greedy response must be a checkpoint")
    if not np.isfinite(initial_objective):
        raise ValueError("Greedy initial objective must be finite")

    active: set[tuple[int, int]] = set()
    current_objective = float(initial_objective)
    checkpoints = set(checkpoint_budgets)
    checkpoint_rows: list[dict] = []
    trace: list[dict] = []
    retained: dict | None = None
    solve_count = 0

    for step in range(1, maximum_budget + 1):
        keep_responses = retained_budget == step
        trials: list[tuple[float, tuple[int, int], dict]] = []
        for edge in ordered_edges:
            if edge in active:
                continue
            solved = response_problem(
                a_mat,
                b_mat,
                horizon,
                sigma,
                active_edges=active | {edge},
                candidate_edges=candidate_edges,
                return_response=keep_responses,
            )
            solve_count += 1
            if solved["status"] != cp.OPTIMAL:
                raise RuntimeError(
                    f"Greedy candidate {edge} returned nonoptimal status {solved['status']}"
                )
            objective = float(solved["objective"])
            if not np.isfinite(objective):
                raise RuntimeError(f"Greedy candidate {edge} returned a nonfinite objective")
            trials.append((objective, edge, solved))

        best_objective = min(objective for objective, _, _ in trials)
        comparison_scale = max(1.0, abs(current_objective), abs(best_objective))
        tie_tolerance = (
            GREEDY_TIE_TOLERANCE_MULTIPLIER
            * FIXED_GRAPH_ABSOLUTE_TOLERANCE
            * comparison_scale
        )
        tied = [trial for trial in trials if trial[0] <= best_objective + tie_tolerance]
        chosen_objective, chosen_edge, chosen_solution = min(tied, key=lambda trial: trial[1])
        if chosen_objective > current_objective + tie_tolerance:
            raise RuntimeError("Greedy fixed graph objective increased beyond tolerance")

        previous_objective = current_objective
        active.add(chosen_edge)
        current_objective = chosen_objective
        if len(active) != step:
            raise RuntimeError("Greedy graph cardinality is inconsistent")
        trace.append(
            {
                "step": step,
                "chosen_edge": list(chosen_edge),
                "objective": chosen_objective,
                "best_candidate_objective": best_objective,
                "marginal_decrease": previous_objective - chosen_objective,
                "tie_tolerance": tie_tolerance,
                "tie_count": len(tied),
            }
        )
        if step in checkpoints:
            checkpoint_rows.append(
                {
                    "budget": step,
                    "objective": chosen_objective,
                    "edges": sorted(active),
                    "status": chosen_solution["status"],
                }
            )
        if step == retained_budget:
            retained = {"solution": chosen_solution, "edges": set(active)}
        print(
            f"Greedy {progress_label} step {step}/{maximum_budget} "
            f"objective {chosen_objective:.9f}",
            flush=True,
        )

    expected_solves = sum(len(ordered_edges) - step for step in range(maximum_budget))
    if solve_count != expected_solves:
        raise RuntimeError("Greedy candidate solve count is inconsistent")
    metadata = {
        "maximum_budget": maximum_budget,
        "checkpoint_budgets": sorted(checkpoints),
        "candidate_solve_count": solve_count,
        "tie_tolerance_multiplier": GREEDY_TIE_TOLERANCE_MULTIPLIER,
        "tie_tolerance_base": FIXED_GRAPH_ABSOLUTE_TOLERANCE,
        "tie_rule": "receiver_source_lexicographic_within_scale_aware_tolerance",
        "trace": trace,
    }
    return checkpoint_rows, metadata, retained


def graph_distances(parent: np.ndarray) -> np.ndarray:
    n = len(parent)
    distance = np.full((n, n), np.inf)
    np.fill_diagonal(distance, 0.0)
    for child, par in enumerate(parent):
        if par >= 0:
            distance[child, par] = 1.0
            distance[par, child] = 1.0
    for k in range(n):
        distance = np.minimum(distance, distance[:, [k]] + distance[[k], :])
    return distance


def archived_data():
    path = OUT / "archived_codesign_records.json"
    if not path.exists():
        raise FileNotFoundError(f"Required original experiment data are missing: {path}")
    return json.loads(path.read_text())

def serial_trial(edges, solved, require_group_feasible=True):
    return {"edges": [list(edge) for edge in sorted(edges)],
            "require_group_feasible": require_group_feasible,
            "solver_objective_estimate": solved["objective"],
            "U": {key: value.tolist() for key, value in solved["U"].items()}}

def regenerate_trials():
    data = archived_data()
    result = {"schema_version": 1, "model": "directly_observed_innovations",
              "edge_order": ["receiver", "source"],
              "coefficient_interpretation": "each JSON binary64 number is an exact rational",
              "response_group_bound": RESPONSE_GROUP_BOUND,
              "state_weight": 22, "input_weight": "7/20"}
    for n, horizon in [(5, 3), (8, 5)]:
        a, b, parent, _ = feeder_model(n)
        sigma = (np.array([0.014,0.017,0.015,0.020,0.018]) if n == 5 else
                 np.array(data["model"]["residual_standard_deviations_by_mode"]["Forecast residual"]))
        candidates = ([(i,j) for i in range(n) for j in range(n) if i != j] if n == 8 else
                      [edge for i in range(4) for edge in [(i,i+1),(i+1,i)]])
        instance = {"nodes":n, "horizon":horizon, "A":a.tolist(), "B":b.tolist(),
                    "sigma":sigma.tolist(), "parent":parent.tolist(),
                    "candidate_edges":[list(e) for e in candidates], "trials":[]}
        if n == 5:
            instance["incumbent_edges"] = data["small_enumeration_check"]["rounded_edges"]
            graphs = [(set(edges), None) for size in range(4)
                      for edges in itertools.combinations(candidates, size)]
        else:
            selected = next(row for row in data["records"] if row["mode"]=="Forecast residual"
                            and row["method"]=="Perspective refit" and row["budget"]==12)
            graphs = [(set(candidates),None), ({tuple(e) for e in selected["edges"]},5.0)]
            instance["selected_method"] = "Perspective refit"
        for index, (edges, cap) in enumerate(graphs):
            solved = response_problem(a,b,horizon,sigma,active_edges=edges,
                                      candidate_edges=candidates,group_bound=cap,return_response=True)
            instance["trials"].append(serial_trial(edges,solved,n==5 or cap is not None))
            if index % 20 == 0:
                print(f"Reconstructed {n} bus trial {index+1} of {len(graphs)}",flush=True)
        result["five_bus" if n==5 else "eight_bus"] = instance
    output = OUT / "voltage_certificate_trials.json"
    output.write_text(json.dumps(result,indent=2)+"\n")
    print(output)

def verify_recorded_study():
    data = archived_data()
    model = data["model"]
    n, horizon = model["nodes"], model["horizon"]
    a,b,parent,_ = feeder_model(n)
    if parent.tolist() != model["parent_zero_based"] or n != 8 or horizon != 5:
        raise ValueError("Archived instance differs from the current model")
    candidates = [(i,j) for i in range(n) for j in range(n) if i!=j]
    methods = {"Greedy refit","Perspective refit","Magnitude refit","Sensitivity refit","Distance refit"}
    rows, cache = [], {}
    for row in data["records"]:
        if row["method"] not in methods or row["budget"] not in {4,8,12,16,24}:
            continue
        sigma = np.array(model["residual_standard_deviations_by_mode"][row["mode"]])
        edges = {tuple(edge) for edge in row["edges"]}
        solved = response_problem(a,b,horizon,sigma,active_edges=edges,candidate_edges=candidates)
        error = abs(solved["objective"]-row["objective"])
        tolerance = 1e-6*max(1.0,abs(row["objective"]))
        if error > tolerance:
            raise ValueError(f"Recorded objective differs for {row['mode']} {row['method']} {row['budget']}")
        reference = {r["method"]:r["objective"] for r in data["records"]
                     if r["mode"]==row["mode"] and r["method"] in {"Local","Full"}}
        item = {"mode":row["mode"],"method":row["method"],"budget":row["budget"],
                "edges":row["edges"],"archived_objective":row["objective"],
                "refit_objective":solved["objective"],"absolute_difference":error,
                "normalized_gap":(row["objective"]-reference["Full"])/(reference["Local"]-reference["Full"]),
                "gap_reduction_percent":100*(reference["Local"]-row["objective"])/(reference["Local"]-reference["Full"])}
        rows.append(item)
        print(f"Verified {row['mode']} {row['method']} budget {row['budget']}",flush=True)
    greedy = [row for row in rows if row["method"]=="Greedy refit"]
    if len(greedy)!=10:
        raise ValueError("Expected ten greedy interior-budget records")
    (OUT/"greedy_baseline.json").write_text(json.dumps({
        "model":"directly_observed_innovations","candidate_links":56,"response_group_bound":5,
        "method":"sequential marginal objective refits",
        "construction":data["diagnostics"]["greedy_marginal_refit_by_mode"],
        "validation":"each retained graph was independently refitted; graph search trace is archived",
        "records":greedy},indent=2)+"\n")
    (OUT/"voltage_record_verification.json").write_text(json.dumps({
        "model":"directly_observed_innovations","records":rows},indent=2)+"\n")

def verify_envelope():
    data = archived_data()
    a,b,_,_ = feeder_model(8)
    sigma = np.array(data["model"]["residual_standard_deviations_by_mode"]["Forecast residual"])
    candidates = [(i,j) for i in range(8) for j in range(8) if i!=j]
    selected = next(row for row in data["records"] if row["mode"]=="Forecast residual"
                    and row["method"]=="Perspective refit" and row["budget"]==12)
    graphs = {"Local":set(), "Perspective":{tuple(e) for e in selected["edges"]},
              "Full":set(candidates)}
    rng = np.random.default_rng(8)
    innovations = rng.normal(size=(400,6,8))*sigma
    curves = {}
    for method, edges in graphs.items():
        response = response_problem(a,b,5,sigma,active_edges=edges,
                                    candidate_edges=candidates,return_response=True)
        maxima = np.empty((400,6))
        for t in range(6):
            x = sum(innovations[:,s,:]@response["X"][f"{t},{s}"].T for s in range(t+1))
            maxima[:,t] = np.max(np.abs(x),axis=1)
        curves[method] = {"median_percent":(100*np.quantile(maxima,.5,axis=0)).tolist(),
                          "p90_percent":(100*np.quantile(maxima,.9,axis=0)).tolist()}
    result = {"model":"directly_observed_innovations","selected_method":"Perspective refit",
              "draws":400,"seed":8,"common_innovations":True,
              "statistic":"marginal quantiles of maximum absolute bus deviation",
              "curves":curves,"compared_with_archive":False}
    archive_path = OUT / "archived_voltage_envelope.json"
    if archive_path.exists():
        archived = json.loads(archive_path.read_text())
        errors = [abs(value - expected)
                  for method in curves for statistic in curves[method]
                  for value, expected in zip(curves[method][statistic],
                                             archived["curves"][method][statistic], strict=True)]
        result.update(compared_with_archive=True, maximum_quantile_difference=max(errors))
        if max(errors) > 5.01e-5:
            raise ValueError("Recomputed quantiles differ from the archived data")
        print(f"Archived quantiles verified, maximum difference {max(errors):.3g}")
    else:
        print("Quantiles recomputed. No archived_voltage_envelope.json was provided for comparison.")
    (OUT/"voltage_envelope_verification.json").write_text(json.dumps(result,indent=2)+"\n")

def regenerate_greedy():
    data = archived_data()
    a,b,_,_ = feeder_model(8)
    candidates = [(i,j) for i in range(8) for j in range(8) if i!=j]
    fresh = {}
    for mode, deviations in data["model"]["residual_standard_deviations_by_mode"].items():
        sigma = np.array(deviations)
        local = response_problem(a,b,5,sigma,active_edges=set(),candidate_edges=candidates)
        rows,metadata,_ = greedy_marginal_refits(a,b,5,sigma,candidates,local["objective"],
                                                [4,8,12,16,24],progress_label=mode)
        fresh[mode] = {"records":rows,"construction":metadata}
    (OUT/"greedy_search_regenerated.json").write_text(json.dumps(fresh,indent=2)+"\n")

def regenerate_rankings():
    data = archived_data()
    a,b,parent,sensitivity = feeder_model(8)
    candidates = [(i,j) for i in range(8) for j in range(8) if i!=j]
    distance = graph_distances(parent)
    distance_scores = {(i,j):float(-distance[i,j]+.02*abs(sensitivity[i,j])) for i,j in candidates}
    sensitivity_scores = {(i,j):float(abs(sensitivity[i,j])) for i,j in candidates}
    output = {"model":"directly_observed_innovations","records":[],"full_group_audits":{}}
    for mode,deviations in data["model"]["residual_standard_deviations_by_mode"].items():
        sigma = np.array(deviations)
        full = response_problem(a,b,5,sigma,active_edges=set(candidates),return_response=True)
        uncapped = response_problem(a,b,5,sigma,active_edges=set(candidates),group_bound=None)
        output["full_group_audits"][mode] = {
            "bounded_objective":full["objective"],"uncapped_objective":uncapped["objective"],
            "absolute_objective_difference":abs(full["objective"]-uncapped["objective"]),
            "uncapped_max_group_norm":uncapped["max_delayed_group_norm"]}
        magnitude = edge_scores_from_response(full,8,5)
        for budget in [4,8,12,16,24]:
            relaxation = response_problem(a,b,5,sigma,perspective_budget=budget,candidate_edges=candidates)
            scores = {tuple(map(int,key.split(","))):value for key,value in relaxation["z"].items()}
            for method,ranking in [("Perspective refit",scores),("Magnitude refit",magnitude),
                                   ("Sensitivity refit",sensitivity_scores),("Distance refit",distance_scores)]:
                edges = select_top(ranking,budget)
                archived = next(row for row in data["records"] if row["mode"]==mode
                                and row["method"]==method and row["budget"]==budget)
                if edges != {tuple(edge) for edge in archived["edges"]}:
                    raise ValueError(f"Regenerated graph differs for {mode} {method} {budget}")
                solved = response_problem(a,b,5,sigma,active_edges=edges)
                if abs(solved["objective"]-archived["objective"])>1e-6:
                    raise ValueError("Regenerated ranking objective differs from the archived value")
                row = {"mode":mode,"method":method,"budget":budget,"edges":sorted(edges),
                       "objective":solved["objective"],"graph_matches_archive":True,
                       "objective_difference":abs(solved["objective"]-archived["objective"])}
                if method=="Perspective refit":
                    row["relaxation_diagnostics"] = relaxation
                output["records"].append(row)
            print(f"Reconstructed all four {mode} rankings at budget {budget}",flush=True)
    (OUT/"voltage_rankings_regenerated.json").write_text(json.dumps(output,indent=2)+"\n")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regenerate-trials",action="store_true",
                        help="solve 93 five-bus graphs and two eight-bus certificate trials (default)")
    parser.add_argument("--verify-recorded-study",action="store_true",
                        help="refit the archived graphs and compare objective values")
    parser.add_argument("--verify-envelope",action="store_true",
                        help="recompute voltage quantiles and compare the optional archived JSON")
    parser.add_argument("--regenerate-greedy",action="store_true",
                        help="rerun all 2136 greedy candidate solves; writes a separate JSON")
    parser.add_argument("--regenerate-rankings",action="store_true",
                        help="recompute the perspective and three static graph rankings")
    args = parser.parse_args()
    if not any(vars(args).values()):
        args.regenerate_trials = True
    OUT.mkdir(exist_ok=True)
    dependencies()
    if args.regenerate_trials:
        regenerate_trials()
        from verify_voltage_certificates import verify
        print(json.dumps(verify(), indent=2))
    if args.verify_recorded_study:
        verify_recorded_study()
    if args.verify_envelope:
        verify_envelope()
    if args.regenerate_greedy:
        regenerate_greedy()
    if args.regenerate_rankings:
        regenerate_rankings()

if __name__ == "__main__":
    main()
