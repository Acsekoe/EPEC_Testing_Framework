from __future__ import annotations

from typing import Any, Dict, List, Tuple
import pyomo.environ as pyo
from pyomo.common.errors import ApplicationError
from pyomo.opt import SolverStatus, TerminationCondition

from epec.core.sets import Sets
from epec.core.params import Params
from epec.core.theta import Theta
from epec.ulp.player_mpec import build_player_mpec


def _val(x, default: float = float("nan")) -> float:
    v = pyo.value(x, exception=False)
    return default if v is None else float(v)


def _fmt_map(d: Dict, key_order=None, width: int = 12, prec: int = 6) -> str:
    if key_order is None:
        key_order = list(d.keys())
    parts = []
    for k in key_order:
        v = d.get(k, float("nan"))
        try:
            s = f"{float(v):{width},.{prec}f}"
        except Exception:
            s = str(v)
        parts.append(f"{k}: {s}")
    return "{ " + ", ".join(parts) + " }"


def _fmt_arcs(
    d: Dict[Tuple[str, str], float],
    arc_order: List[Tuple[str, str]] | None = None,
    width: int = 12,
    prec: int = 6,
) -> str:
    if arc_order is None:
        arc_order = list(d.keys())
    lines = []
    for (e, r) in arc_order:
        v = d.get((e, r), float("nan"))
        lines.append(f"    {e}->{r}: {float(v):{width},.{prec}f}")
    return "\n".join(lines)


def _print_player_block(it: int, player: str, m: pyo.ConcreteModel, sets: Sets) -> None:
    R = list(sets.R)
    RR = list(sets.RR)
    RRx = list(sets.RRx)

    ulp_obj = _val(m.ULP_OBJ) if hasattr(m, "ULP_OBJ") else float("nan")
    llp_obj = _val(m.LLP_OBJ.expr) if hasattr(m, "LLP_OBJ") else float("nan")

    q_man = {r: _val(m.q_man[r]) for r in R} if hasattr(m, "q_man") else {}
    d_offer = {r: _val(m.d_offer[r]) for r in R} if hasattr(m, "d_offer") else {}
    tau = {(e, r): _val(m.tau[e, r]) for (e, r) in RR} if hasattr(m, "tau") else {}
    markup = {(e, r): _val(m.markup[e, r]) for (e, r) in RR} if hasattr(m, "markup") else {}

    x_man = {r: _val(m.x_man[r]) for r in R} if hasattr(m, "x_man") else {}
    u_short = {r: _val(m.u_short[r]) for r in R} if hasattr(m, "u_short") else {}

    x_flow = {(e, r): _val(m.x_flow[e, r]) for (e, r) in RRx} if hasattr(m, "x_flow") else {}

    lam = {r: _val(m.lam[r]) for r in R} if hasattr(m, "lam") else {}
    pi = {r: _val(m.pi[r]) for r in R} if hasattr(m, "pi") else {}

    mu_man = {r: _val(m.mu_man[r]) for r in R} if hasattr(m, "mu_man") else {}
    nu_ushort = {r: _val(m.nu_ushort[r]) for r in R} if hasattr(m, "nu_ushort") else {}

    mod_res = {}
    prod_res = {}
    if hasattr(m, "mod_balance"):
        for r in R:
            mod_res[r] = _val(m.mod_balance[r].body)
    if hasattr(m, "prod_balance"):
        for r in R:
            prod_res[r] = _val(m.prod_balance[r].body)

    sep = "-" * 78
    print(sep)
    print(f"[iter {it:>2}] player={player} | ULP_OBJ={ulp_obj:,.6f} | LLP_OBJ(expr)={llp_obj:,.6f}")
    print(sep)

    print("Upper-level decisions (all regions):")
    if q_man:
        print("  q_man:", _fmt_map(q_man, key_order=R))
    if d_offer:
        print("  d_offer:", _fmt_map(d_offer, key_order=R))
    if tau:
        print("  tau (tariff factors on shipping, trade arcs):")
        print(_fmt_arcs(tau, arc_order=RR))
    if markup:
        print("  markup (export markups, trade arcs):")
        print(_fmt_arcs(markup, arc_order=RR))

    print("\nLower-level variables / flows:")
    if x_man:
        print("  x_man:", _fmt_map(x_man, key_order=R))
    if u_short:
        print("  u_short (feasibility slack):", _fmt_map(u_short, key_order=R))
        print(f"  max_u_short: {max(u_short.values()):.6g}")
    if x_flow:
        print("  x_flow (all pairs incl. diagonal):")
        print(_fmt_arcs(x_flow, arc_order=RRx))

    print("\nDual variables (KKT):")
    if lam:
        print("  lam (module balance):", _fmt_map(lam, key_order=R))
    if pi:
        print("  pi  (production balance):", _fmt_map(pi, key_order=R))
    if mu_man:
        print("  mu_man (x_man<=q_man):", _fmt_map(mu_man, key_order=R))
    if nu_ushort:
        print("  nu_ushort (u_short>=0):", _fmt_map(nu_ushort, key_order=R))

    print("\nResiduals (equalities, should be ~0):")
    if mod_res:
        print("  mod_balance:", _fmt_map(mod_res, key_order=R, width=12, prec=3))
    if prod_res:
        print("  prod_balance:", _fmt_map(prod_res, key_order=R, width=12, prec=3))

    print(sep)


def solve_gauss_seidel(
    sets: Sets,
    params: Params,
    theta0: Theta,
    max_iter: int = 30,
    tol: float = 1e-4,
    damping: float = 0.8,
    price_sign: float = -1.0,
    eps_pen: float = 1e-8,  # kept for backwards compatibility (unused in new formulation)
    eps_reg: float = 1e-6,
    eps_price_reg: float = 0.5,  # NEW: price anchoring strength (pass through to player MPEC)
    M_dual: float = 1e6,
    kkt_mode: str = "bigM",
    use_shortage_slack: bool = True,
    solver_name: str = "gurobi_direct",
    gurobi_options: Dict[str, float] | None = None,
    verbose: bool = True,
    run_cfg: Dict[str, Any] | None = None,
) -> Tuple[Theta, List[dict]]:
    """
    Gauss-Seidel over players.

    Stopping criterion (FIXED):
      Uses a *normalized* infinity-norm change over strategic vars:
        d_offer scaled by D_hat
        q_man   scaled by Q_man_hat
        tau     scaled by (tau_ub-1)
        markup  scaled by m_ub
      Stop when max normalized change < tol for `conv_required` consecutive outer iterations.

    Solver requirements depend on `kkt_mode`:
      - `bigM`: MIP/MIQP solver (typically Gurobi).
      - `bilinear`: NLP/QCP-capable solver (Ipopt or Gurobi with NonConvex=2).
    """

    def _get_first_available_solver(names: List[str]):
        tried: List[str] = []
        for name in names:
            tried.append(name)
            s = pyo.SolverFactory(name)
            if s is not None and s.available(exception_flag=False):
                return s
        raise ApplicationError(
            "No executable found for solver(s): "
            + ", ".join(tried)
            + ". Install a supported solver (e.g., Gurobi/Ipopt) or set run_cfg['solver_name']."
        )

    # ---- pull overrides from run_cfg
    if run_cfg:
        max_iter = int(run_cfg.get("max_iter", max_iter))
        tol = float(run_cfg.get("tol", tol))
        damping = float(run_cfg.get("damping", damping))
        price_sign = float(run_cfg.get("price_sign", price_sign))
        eps_pen = float(run_cfg.get("eps_pen", eps_pen))
        eps_reg = float(run_cfg.get("eps_reg", eps_reg))
        eps_price_reg = float(run_cfg.get("eps_price_reg", eps_price_reg))
        M_dual = float(run_cfg.get("M_dual", M_dual))
        kkt_mode = str(run_cfg.get("kkt_mode", kkt_mode))
        use_shortage_slack = bool(run_cfg.get("use_shortage_slack", use_shortage_slack))
        solver_name = str(run_cfg.get("solver_name", run_cfg.get("solver", solver_name)))
        gurobi_options = run_cfg.get("gurobi_options", gurobi_options)
        solver_options = run_cfg.get("solver_options", run_cfg.get("solver_opts", None))
        conv_required = int(run_cfg.get("conv_required", 3))
    else:
        solver_options = None
        conv_required = 3

    theta = theta0.copy()
    kkt_mode_l = kkt_mode.strip().lower()

    # Pick sensible default:
    if kkt_mode_l == "bilinear" and solver_name == "gurobi_direct":
        solver_name = "ipopt"

    if kkt_mode_l in ("bigm", "big_m", "big-m"):
        candidate_solvers = [solver_name, "gurobi_direct", "gurobi"]
    else:
        candidate_solvers = [solver_name, "ipopt", "gurobi_direct", "gurobi"]

    # de-duplicate while preserving order
    seen = set()
    candidate_solvers = [s for s in candidate_solvers if not (s in seen or seen.add(s))]

    solver = _get_first_available_solver(candidate_solvers)

    solver_name_l = str(getattr(solver, "name", solver_name)).lower()
    if solver_name_l.startswith("gurobi"):
        solver.options["NonConvex"] = 2  # allow nonconvex QP/QCP if present
        solver.options["OutputFlag"] = 1 if verbose else 0
        solver.options["MIPGap"] = 1e-6
        solver.options["FeasibilityTol"] = 1e-8
        solver.options["OptimalityTol"] = 1e-8

    if gurobi_options:
        for k, v in gurobi_options.items():
            solver.options[k] = v

    if isinstance(solver_options, dict):
        for k, v in solver_options.items():
            solver.options[k] = v

    # ---- normalized change helpers (FIXED stopping logic)
    def _rel_change_scalar(delta: float, scale: float) -> float:
        scale = max(1.0, float(scale))
        return abs(float(delta)) / scale

    def _rel_change_tau(delta: float, e: str, rr: str) -> float:
        rng = float(params.tau_ub[(e, rr)] - 1.0)
        rng = max(1e-6, rng)
        return abs(float(delta)) / rng

    def _rel_change_markup(delta: float, e: str, rr: str) -> float:
        ub = float(params.m_ub[(e, rr)])
        ub = max(1.0, ub)
        return abs(float(delta)) / ub

    hist: List[dict] = []
    iters_done = 0
    conv_streak = 0

    for it in range(max_iter):
        max_change = 0.0

        for r in sets.R:
            m = build_player_mpec(
                r,
                sets,
                params,
                theta,
                price_sign=price_sign,
                eps_reg=eps_reg,
                eps_price_reg=eps_price_reg,  # IMPORTANT: pass through
                M_dual=M_dual,
                kkt_mode=kkt_mode,
                use_shortage_slack=use_shortage_slack,
            )

            res = solver.solve(m, tee=False, load_solutions=False)
            status = res.solver.status
            tc = res.solver.termination_condition

            ok = (status == SolverStatus.ok) and (tc in (TerminationCondition.optimal, TerminationCondition.locallyOptimal))
            if not ok:
                hist.append({"iter": it, "region": r, "accepted": False, "status": str(status), "term": str(tc)})
                if verbose:
                    print(f"[iter {it:>2}] player={r}  SOLVE FAILED  status={status} term={tc}")
                continue

            m.solutions.load_from(res)

            # ---- Scalars / objectives
            ulp_val = _val(m.ULP_OBJ) if hasattr(m, "ULP_OBJ") else float("nan")
            llp_val = _val(m.LLP_OBJ.expr) if hasattr(m, "LLP_OBJ") else float("nan")

            # ---- Per-region dicts (for Excel)
            lam_vals = {rr: _val(m.lam[rr]) for rr in sets.R} if hasattr(m, "lam") else {}
            pi_vals = {rr: _val(m.pi[rr]) for rr in sets.R} if hasattr(m, "pi") else {}

            u_short_vals = {rr: _val(m.u_short[rr]) for rr in sets.R} if hasattr(m, "u_short") else {}
            nu_ushort_vals = {rr: _val(m.nu_ushort[rr]) for rr in sets.R} if hasattr(m, "nu_ushort") else {}

            x_man_vals = {rr: _val(m.x_man[rr]) for rr in sets.R} if hasattr(m, "x_man") else {}
            x_flow_vals = {(e, rr): _val(m.x_flow[e, rr]) for (e, rr) in sets.RRx} if hasattr(m, "x_flow") else {}

            max_u_short_val = max(u_short_vals.values()) if u_short_vals else float("nan")

            if verbose:
                _print_player_block(it=it, player=r, m=m, sets=sets)

            # ---- Best response (Gauss-Seidel update)
            br_q_man_r = _val(m.q_man[r])
            br_d_offer_r = _val(m.d_offer[r])

            # inbound tariffs (chosen by importer r)
            br_tau_in = {(e, r): _val(m.tau[e, r]) for e in sets.R if e != r}

            # outbound markups (chosen by exporter r)
            br_m_out = {(r, i): _val(m.markup[r, i]) for i in sets.R if i != r}

            def upd(old, new):
                return old + damping * (new - old)

            # update player scalars (normalized change)
            old_q, old_d = theta.q_man[r], theta.d_offer[r]
            theta.q_man[r] = upd(old_q, br_q_man_r)
            theta.d_offer[r] = upd(old_d, br_d_offer_r)

            max_change = max(
                max_change,
                _rel_change_scalar(theta.q_man[r] - old_q, params.Q_man_hat[r]),
                _rel_change_scalar(theta.d_offer[r] - old_d, params.D_hat[r]),
            )

            # update inbound taus into r (normalized change by range)
            for (e, rr), newv in br_tau_in.items():
                oldv = theta.tau[(e, rr)]
                theta.tau[(e, rr)] = upd(oldv, newv)
                max_change = max(max_change, _rel_change_tau(theta.tau[(e, rr)] - oldv, e, rr))

            # update outbound markups from r (normalized change by ub)
            for (e, i), newv in br_m_out.items():
                oldv = theta.markup[(e, i)]
                theta.markup[(e, i)] = upd(oldv, newv)
                max_change = max(max_change, _rel_change_markup(theta.markup[(e, i)] - oldv, e, i))

            # ---- Store history row (Excel)
            hist.append(
                {
                    "iter": it,
                    "region": r,
                    "accepted": True,
                    "status": str(status),
                    "term": str(tc),
                    "ulp_obj": ulp_val,
                    "llp_obj": llp_val,
                    "lambda": lam_vals,
                    "pi": pi_vals,
                    "u_short": u_short_vals,
                    "nu_ushort": nu_ushort_vals,
                    "max_u_short": max_u_short_val,
                    "x_man": x_man_vals,
                    "x_flow": x_flow_vals,
                    "br_q_man": float(br_q_man_r),
                    "br_d_offer": float(br_d_offer_r),
                    "br_tau_in": dict(br_tau_in),   # {(e,r): val} inbound for this player
                    "br_m_out": dict(br_m_out),     # {(r,i): val} outbound for this player
                    "theta_q_man": dict(theta.q_man),
                    "theta_d_offer": dict(theta.d_offer),
                    "theta_tau": dict(theta.tau),
                    "theta_markup": dict(theta.markup),
                    "gs_max_change_norm": float(max_change),
                    "eps_price_reg": float(eps_price_reg),
                }
            )

        iters_done = it + 1
        if verbose:
            print(f"\n=== end GS iter {it}: max_change_norm={max_change:.6g} (tol={tol}, streak={conv_streak}/{conv_required}) ===")

        # ---- consecutive convergence requirement
        if max_change < tol:
            conv_streak += 1
        else:
            conv_streak = 0

        if conv_streak >= conv_required:
            break

    if verbose:
        print(f"\n=== Gauss-Seidel finished: {iters_done} iteration(s) executed | conv_streak={conv_streak}/{conv_required} ===")

    return theta, hist
