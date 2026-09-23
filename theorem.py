"""Predictive SLS co-design: enumerate 32 graphs and check the main theorem.

Run: python theorem.py. Results go to output/theorem.{json,csv,npz}.
This is floating-point validation, not an exact certificate. Disturbances are
measured locally, source blocks may be relayed, and every scalar input-response
coefficient satisfies |g| <= M. No state or physical operating limits are used.
"""
from pathlib import Path
from types import SimpleNamespace
import csv
import json
import time

import cvxpy as cp
import numpy as np
from scipy import sparse


def model():
    """p,e are independent N(0,I), eta=.8*p+.6*e. Release p at rho_s.

    The forecast is exogenous information, not a future state measurement.
    Each link is (source, destination, delay, installation cost), zero indexed.
    """
    n, T = 3, 4
    A = np.array([[.70, .06, .04], [.26, .64, .08], [.17, .23, .66]])
    B = np.diag([.50, .55, .60])
    nx, nu = (T+1)*n, T*n
    D, B_lift = np.eye(nx), np.zeros((nx, nu))
    for t in range(T):
        D[(t+1)*n:(t+2)*n, t*n:(t+1)*n] = -A
        B_lift[(t+1)*n:(t+2)*n, t*n:(t+1)*n] = B
    D_inv = np.linalg.solve(D, np.eye(nx))
    G = D_inv @ B_lift
    X0 = np.hstack((D_inv, np.zeros_like(D_inv)))
    Q = np.kron(np.eye(T+1), np.diag([1., 1.2, 1.5]))
    Q[-n:, -n:] *= 2
    R = .2*np.eye(nu)
    Sigma = np.block([[np.eye(nx), .8*np.eye(nx)], [.8*np.eye(nx), np.eye(nx)]])
    links = np.array([(0, 1, 1, 1), (1, 2, 1, 1), (0, 2, 2, 3),
                      (2, 0, 1, 2), (2, 1, 1, 1)])
    return SimpleNamespace(n=n, T=T, A=A, B=B, D=D, B_lift=B_lift, G=G,
                           X0=X0, Q=Q, R=R, Sigma=Sigma, links=links)


def reachability(m, z):
    """D[j,i] is the shortest installed path delay from source j to receiver i."""
    D = np.full((m.n, m.n), np.inf)
    np.fill_diagonal(D, 0)
    for (j, i, delay, _), installed in zip(m.links, z):
        if installed:
            D[j, i] = min(D[j, i], delay)
    for k in range(m.n):
        D = np.minimum(D, D[:, k, None]+D[None, k, :])
    return D


def mask(m, D, q):
    """Rows are (t,i), columns are eta(s,j) followed by p(s,j)."""
    nx = (m.T+1)*m.n
    a = np.zeros((m.T*m.n, 2*nx), dtype=bool)
    for t in range(m.T):
        for s in range(m.T+1):
            rows, columns = slice(t*m.n, (t+1)*m.n), slice(s*m.n, (s+1)*m.n)
            a[rows, columns] = s+D.T <= t
            a[rows, slice(nx+s*m.n, nx+(s+1)*m.n)] = max(0, s-q)+D.T <= t
    return a


def cost(m, U):
    X = m.X0+m.G@U
    return float(np.trace((X.T@m.Q@X+U.T@m.R@U)@m.Sigma))


def solve(m, a, M):
    """Optimize available coefficients. Convert normalized cap duals to nu_g."""
    indices = np.flatnonzero(a.ravel())
    selection = sparse.csc_matrix((np.ones(len(indices)), (indices, np.arange(len(indices)))),
                                  shape=(a.size, len(indices)))
    v = cp.Variable(len(indices))
    U = cp.reshape(M*(selection@v), a.shape, order="C")
    X = m.X0+m.G@U
    L = np.linalg.cholesky(m.Sigma)
    objective = cp.sum_squares(np.diag(np.sqrt(np.diag(m.Q)))@X@L)
    objective += cp.sum_squares(np.diag(np.sqrt(np.diag(m.R)))@U@L)
    caps = cp.square(v) <= 1
    problem = cp.Problem(cp.Minimize(objective), [caps])
    problem.solve(solver="CLARABEL", tol_gap_abs=1e-10, tol_gap_rel=1e-10,
                  tol_feas=1e-10, max_iter=300)
    if problem.status != cp.OPTIMAL:
        raise RuntimeError(f"Solver status: {problem.status}")
    u, nu = np.zeros(a.size), np.zeros(a.size)
    u[indices] = M*v.value
    # mu multiplies (g/M)^2 <= 1, so nu=mu/M^2 multiplies g^2 <= M^2.
    nu[indices] = caps.dual_value/M**2
    u, nu = u.reshape(a.shape), nu.reshape(a.shape)
    H, F = m.R+m.G.T@m.Q@m.G, m.G.T@m.Q@m.X0
    gradient = 2*(H@u+F)@m.Sigma
    stationarity = np.max(np.abs((gradient+2*nu*u)[a]))/(1+np.max(np.abs(gradient[a])))
    diagnostics = dict(stationarity=float(stationarity),
                       complementarity=float(np.max(np.abs(nu*(u*u-M*M)))),
                       dual_min=float(nu[a].min()))
    return u, nu, diagnostics


def run(output_dir=Path(__file__).parent/"output"):
    started, m = time.perf_counter(), model()
    tolerance, support_tolerance, kkt_tolerance = 2e-6, 1e-6, 1e-5
    graphs = np.array([[(g >> e) & 1 for e in range(len(m.links))]
                       for g in range(1 << len(m.links))])
    graph_costs = graphs@m.links[:, 3]
    delays = [reachability(m, z) for z in graphs]
    eig = np.linalg.eigvalsh(m.Sigma)
    alpha = float(eig[0]*np.linalg.eigvalsh(m.R)[0])
    weaker_alpha = alpha/(1+np.linalg.norm(m.G, 2)**2)
    curvature = float(eig[-1]*np.linalg.eigvalsh(m.R+m.G.T@m.Q@m.G)[-1])
    cases, csv_rows = [], []
    arrays = {"A": m.A, "B": m.B, "D": m.D, "B_lift": m.B_lift, "Q": m.Q,
              "R": m.R, "Sigma": m.Sigma, "links": m.links, "graphs": graphs,
              "graph_costs": graph_costs, "graph_delays": np.array(delays)}
    checks = {"positive_definite_weights": bool(eig[0] > 0 and np.linalg.eigvalsh(m.R)[0] > 0)}
    for q in (0, 1):
        masks = np.array([mask(m, D, q) for D in delays])
        for M, cap_name in ((2., "loose"), (.1, "active")):
            key = f"q{q}_{cap_name}"
            solutions = [solve(m, a, M) for a in masks]
            U = np.array([s[0] for s in solutions])
            nu = np.array([s[1] for s in solutions])
            diagnostics = [s[2] for s in solutions]
            values = np.array([cost(m, u) for u in U])
            U_full, nu_full, V_full = U[-1], nu[-1], float(values[-1])
            missing = np.sum((~masks)*U_full**2, axis=(1, 2))
            weighted = np.sum((~masks)*(curvature+2*nu_full)*U_full**2, axis=(1, 2))
            truncated = masks*U_full
            trunc_values = np.array([cost(m, u) for u in truncated])
            thresholds = [int(graph_costs[np.all(masks | (np.abs(U_full) <= tol), axis=(1, 2))].min())
                          for tol in (1e-8, 1e-7, support_tolerance, 1e-5)]
            kappa_H = thresholds[2]
            same_mask_cost = int(graph_costs[np.all(masks == masks[-1], axis=(1, 2))].min())
            # Check SLS, communication masks and caps for optimized and truncated responses.
            feasibility = 0.
            for responses in (U, truncated):
                for u, a in zip(responses, masks):
                    X = m.X0+m.G@u
                    residual = m.D@X-m.B_lift@u-np.hstack((np.eye(m.D.shape[0]), np.zeros_like(m.D)))
                    feasibility = max(feasibility, np.max(np.abs(residual)),
                                      np.max(np.abs(u[~a]), initial=0.), np.max(np.abs(u))-M)
            checks[key+"_graph_bounds"] = bool(np.all(alpha*missing <= values-V_full+tolerance)
                and np.all(values <= trunc_values+tolerance)
                and np.all(trunc_values-V_full <= weighted+tolerance))
            checks[key+"_feasibility"] = bool(feasibility <= tolerance)
            checks[key+"_KKT"] = all(d["stationarity"] <= kkt_tolerance and
                d["complementarity"] <= kkt_tolerance and d["dual_min"] >= -kkt_tolerance for d in diagnostics)
            checks[key+"_support_stable"] = len(set(thresholds)) == 1
            checks[key+"_exact_full_mask"] = same_mask_cost == kappa_H
            active_caps = int(np.count_nonzero(nu_full > 1e-5))
            checks[key+"_cap_case"] = active_caps == 0 if M == 2 else active_caps > 0
            checks[key+"_prediction_used"] = bool(q == 0 or np.linalg.norm(U_full[:, m.D.shape[0]:]) > 1e-4)
            rows = []
            for C in range(int(graph_costs.max())+1):
                ids = np.flatnonzero(graph_costs <= C)
                best = int(ids[np.argmin(values[ids])])
                row = dict(case=key, preview=q, cap=M, budget=C, J_star=float(values[best]),
                           V_full=V_full, loss=float(values[best]-V_full),
                           lower_bound=float(alpha*missing[ids].min()),
                           paper_weaker_lower_bound=float(weaker_alpha*missing[ids].min()),
                           upper_bound=float(weighted[ids].min()),
                           truncated_upper_loss=float(trunc_values[ids].min()-V_full),
                           best_graph_id=best, best_graph_cost=int(graph_costs[best]), kappa_H=kappa_H)
                checks[f"{key}_budget{C}"] = bool(
                    row["lower_bound"] <= row["loss"]+tolerance and
                    row["loss"] <= row["truncated_upper_loss"]+tolerance and
                    row["truncated_upper_loss"] <= row["upper_bound"]+tolerance and
                    (C >= kappa_H) == (abs(row["loss"]) <= tolerance))
                rows.append(row)
            case_diagnostics = dict(feasibility_max=float(feasibility),
                stationarity_max=max(d["stationarity"] for d in diagnostics),
                complementarity_max=max(d["complementarity"] for d in diagnostics),
                dual_min=min(d["dual_min"] for d in diagnostics), active_full_caps=active_caps,
                thresholds_by_tolerance=thresholds, exact_full_mask_cost=same_mask_cost)
            cases.append(dict(case=key, preview=q, cap=M, V_full=V_full, kappa_H=kappa_H,
                              diagnostics=case_diagnostics, budgets=rows))
            csv_rows.extend(rows)
            arrays.update({key+"_U": U, key+"_nu": nu, key+"_availability": masks,
                           key+"_objectives": values, key+"_U_truncated": truncated,
                           key+"_truncated_objectives": trunc_values})
            print(f"{key}: V_full={V_full:.9f}, kappa_H={kappa_H}, active caps={active_caps}")
    result = dict(experiment="main_theorem", validation="floating_point_not_exact_certificate",
                  horizon=m.T, graph_count=len(graphs), solves=len(graphs)*len(cases),
                  release_rule="rho_s=max(0,s-q)", support_tolerance=support_tolerance,
                  verification_tolerance=tolerance, kkt_tolerance=kkt_tolerance,
                  lower_prefactor=alpha, paper_weaker_lower_prefactor=weaker_alpha,
                  upper_curvature=curvature, cases=cases,
                  diagnostics=dict(all_checks_pass=all(checks.values()), checks=checks,
                                   elapsed_seconds=time.perf_counter()-started))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir/"theorem.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    with (output_dir/"theorem.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    np.savez_compressed(output_dir/"theorem.npz", **arrays)
    if not all(checks.values()):
        raise RuntimeError(f"Failed checks: {[name for name, passed in checks.items() if not passed]}")
    print(f"PASS: {len(checks)} checks, {result['solves']} convex solves. Results: {output_dir}")
    return result


if __name__ == "__main__":
    run()
