from __future__ import annotations

import pyomo.environ as pyo

from epec.core.sets import Sets
from epec.core.params import Params


def add_llp_kkt_bigM(
    m: pyo.ConcreteModel,
    sets: Sets,
    params: Params,
    eps_reg: float,
    M_dual: float = 1e6,
    use_shortage_slack: bool = True,
) -> None:
    """Embed LLP KKT with Big-M + binaries (MIQP).

    Primal must already be present in `m` (from build_llp_primal). This adds:
    - duals: lam (MB), pi (PB), mu_man >=0 (CAP), and nonnegativity duals
    - stationarity equations (including eps_reg terms)
    - complementarity via binaries (Big-M)

    This is robust but introduces many binaries.
    """

    R, RR, RRx = sets.R, sets.RR, sets.RRx
    BIG = float(M_dual)

    # ---- dual variables (bounded for numerics)
    m.lam = pyo.Var(m.R, bounds=(-BIG, BIG), initialize=0.0)  # MB (price)
    m.pi = pyo.Var(m.R, bounds=(-BIG, BIG), initialize=0.0)   # PB

    m.mu_man = pyo.Var(m.R, within=pyo.NonNegativeReals, bounds=(0.0, BIG), initialize=0.0)  # CAP

    # nonnegativity duals
    m.nu_xman = pyo.Var(m.R, within=pyo.NonNegativeReals, bounds=(0.0, BIG), initialize=0.0)
    m.nu_xflow = pyo.Var(m.RRx, within=pyo.NonNegativeReals, bounds=(0.0, BIG), initialize=0.0)

    if use_shortage_slack:
        m.nu_ushort = pyo.Var(m.R, within=pyo.NonNegativeReals, bounds=(0.0, BIG), initialize=0.0)

    # ---- stationarity
    # x_man[r]: c_mod_man + pi + mu_man - nu_xman + eps_reg*x_man = 0
    m.stat_xman = pyo.Constraint(
        m.R,
        rule=lambda mm, r: mm.c_mod_man[r] + mm.pi[r] + mm.mu_man[r] - mm.nu_xman[r] + eps_reg * mm.x_man[r] == 0
    )

    # x_flow[e,r]: cost(e,r) + lam[r] - pi[e] - nu_xflow + eps_reg*x_flow = 0
    def _stat_xflow(mm: pyo.ConcreteModel, e: str, r: str):
        if e == r:
            cost = mm.c_mod_dom_use[r]
        else:
            # transport + specific tariff
            cost = mm.s_ship[e, r] + mm.T[e, r]
        return cost + mm.lam[r] - mm.pi[e] - mm.nu_xflow[e, r] + eps_reg * mm.x_flow[e, r] == 0

    m.stat_xflow = pyo.Constraint(m.RRx, rule=_stat_xflow)

    if use_shortage_slack:
        # u_short[r]: c_pen_llp + lam[r] - nu_ushort + eps_reg*u_short = 0
        m.stat_ushort = pyo.Constraint(
            m.R,
            rule=lambda mm, r: mm.c_pen_llp[r] + mm.lam[r] - mm.nu_ushort[r] + eps_reg * mm.u_short[r] == 0
        )

    # ---- Big-M complementarity
    # Helper: add nonneg complementarity nu >= 0 ⟂ x >= 0
    def add_nonneg_comp(nu: pyo.Var, x: pyo.Var, Mx: float, name: str):
        z = pyo.Var(within=pyo.Binary)
        setattr(m, f"z_{name}", z)
        setattr(m, f"bm_{name}_nu", pyo.Constraint(expr=nu <= BIG * z))
        setattr(m, f"bm_{name}_x", pyo.Constraint(expr=x <= float(Mx) * (1 - z)))

    # (CAP) mu_man ⟂ (q_man - x_man)
    m.z_man_cap = pyo.Var(m.R, within=pyo.Binary)
    for r in R:
        setattr(m, f"bm_mu_man_ub_{r}", pyo.Constraint(expr=m.mu_man[r] <= BIG * m.z_man_cap[r]))
        Qhat = float(params.Q_man_hat[r])
        setattr(
            m,
            f"bm_slack_man_ub_{r}",
            pyo.Constraint(expr=(m.q_man[r] - m.x_man[r]) <= Qhat * (1 - m.z_man_cap[r]))
        )

    # (NN) nu_xman ⟂ x_man
    for r in R:
        add_nonneg_comp(m.nu_xman[r], m.x_man[r], params.Q_man_hat[r], f"xman_{r}")

    # (NN) nu_xflow ⟂ x_flow, bound by importer potential demand
    for (e, r) in RRx:
        add_nonneg_comp(m.nu_xflow[e, r], m.x_flow[e, r], params.D_hat[r], f"xflow_{e}_{r}")

    if use_shortage_slack:
        # (NN) nu_ushort ⟂ u_short
        for r in R:
            add_nonneg_comp(m.nu_ushort[r], m.u_short[r], params.D_hat[r], f"ushort_{r}")


def add_llp_kkt_bilinear(
    m: pyo.ConcreteModel,
    sets: Sets,
    params: Params,
    eps_reg: float,
    use_shortage_slack: bool = True,
    dual_bound: float = 1e6,
) -> None:
    """Embed LLP KKT with *hard* (bilinear) complementarity.

    This avoids binaries, but introduces nonconvex bilinear equalities, e.g.
      mu_man[r] * (q_man[r] - x_man[r]) = 0.

    With Gurobi, set NonConvex=2.

    WARNING: This is often harder numerically than Big-M, but can be faster
    than MIQP if it converges cleanly.
    """

    R, RR, RRx = sets.R, sets.RR, sets.RRx
    BIG = float(dual_bound)

    # dual variables (bounded for numerics)
    m.lam = pyo.Var(m.R, bounds=(-BIG, BIG), initialize=0.0)
    m.pi = pyo.Var(m.R, bounds=(-BIG, BIG), initialize=0.0)

    m.mu_man = pyo.Var(m.R, within=pyo.NonNegativeReals, bounds=(0.0, BIG), initialize=0.0)
    m.nu_xman = pyo.Var(m.R, within=pyo.NonNegativeReals, bounds=(0.0, BIG), initialize=0.0)
    m.nu_xflow = pyo.Var(m.RRx, within=pyo.NonNegativeReals, bounds=(0.0, BIG), initialize=0.0)

    if use_shortage_slack:
        m.nu_ushort = pyo.Var(m.R, within=pyo.NonNegativeReals, bounds=(0.0, BIG), initialize=0.0)

    # stationarity
    m.stat_xman = pyo.Constraint(
        m.R,
        rule=lambda mm, r: mm.c_mod_man[r] + mm.pi[r] + mm.mu_man[r] - mm.nu_xman[r] + eps_reg * mm.x_man[r] == 0
    )

    def _stat_xflow(mm: pyo.ConcreteModel, e: str, r: str):
        if e == r:
            cost = mm.c_mod_dom_use[r]
        else:
            cost = mm.s_ship[e, r] + mm.T[e, r]
        return cost + mm.lam[r] - mm.pi[e] - mm.nu_xflow[e, r] + eps_reg * mm.x_flow[e, r] == 0

    m.stat_xflow = pyo.Constraint(m.RRx, rule=_stat_xflow)

    if use_shortage_slack:
        m.stat_ushort = pyo.Constraint(
            m.R,
            rule=lambda mm, r: mm.c_pen_llp[r] + mm.lam[r] - mm.nu_ushort[r] + eps_reg * mm.u_short[r] == 0
        )

    # complementarity as bilinear equalities
    m.comp_man_cap = pyo.Constraint(m.R, rule=lambda mm, r: mm.mu_man[r] * (mm.q_man[r] - mm.x_man[r]) == 0)
    m.comp_xman = pyo.Constraint(m.R, rule=lambda mm, r: mm.nu_xman[r] * mm.x_man[r] == 0)
    m.comp_xflow = pyo.Constraint(m.RRx, rule=lambda mm, e, r: mm.nu_xflow[e, r] * mm.x_flow[e, r] == 0)

    if use_shortage_slack:
        m.comp_ushort = pyo.Constraint(m.R, rule=lambda mm, r: mm.nu_ushort[r] * mm.u_short[r] == 0)
