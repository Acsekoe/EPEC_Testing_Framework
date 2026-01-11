from __future__ import annotations

import pyomo.environ as pyo

from epec.core.sets import Sets
from epec.core.params import Params
from epec.core.theta import Theta


def build_llp_primal(
    sets: Sets,
    params: Params,
    theta: Theta,
    eps_reg: float = 1e-6,
    use_shortage_slack: bool = True,
) -> pyo.ConcreteModel:
    """Build the follower (LLP) primal.

    Reformulated LLP (convex QP due to tiny regularization):

    - Unified flow variable x_flow[e,r] for *all* pairs (including diagonal).
      The diagonal x_flow[r,r] represents domestic use.
    - Off-diagonal flows pay transport + (specific) import tariff:
         (s_ship[e,r] + T[e,r]) * x_flow[e,r]
    - Tariffs T[e,r] are strategic (upper-level) variables (typically fixed except
      for the currently-optimizing importing region in the Gauss--Seidel loop).
    - Demand is committed via MB to d_offer[r].

    Optional feasibility safeguard:
    - shortage slack u_short[r] >= 0 in MB, penalized in LLP by c_pen_llp[r].
      If total supply is insufficient, the model remains feasible and prices cap
      near the shortage penalty.

    The objective includes a tiny strictly convex term (eps_reg) to pin down a
    unique primal (and usually more stable duals).
    """

    R, RR, RRx = sets.R, sets.RR, sets.RRx

    m = pyo.ConcreteModel("LLP_Primal")

    m.R = pyo.Set(initialize=R)
    m.RR = pyo.Set(dimen=2, initialize=RR)     # trade arcs (e!=r)
    m.RRx = pyo.Set(dimen=2, initialize=RRx)   # all pairs (incl. diagonal)

    # Strategic vars enter LLP as VARIABLES (fixed/bounded in player wrapper)
    m.q_man = pyo.Var(m.R, within=pyo.NonNegativeReals, initialize=theta.q_man)
    m.d_offer = pyo.Var(m.R, within=pyo.NonNegativeReals, initialize=theta.d_offer)
    m.T = pyo.Var(m.RR, within=pyo.NonNegativeReals, initialize=theta.T)

    # Params
    m.c_mod_man = pyo.Param(m.R, initialize=params.c_mod_man)
    m.c_mod_dom_use = pyo.Param(m.R, initialize=params.c_mod_dom_use)
    m.s_ship = pyo.Param(m.RR, initialize=params.s_ship)
    m.c_pen_llp = pyo.Param(m.R, initialize=params.c_pen_llp)

    # Primal vars
    m.x_man = pyo.Var(m.R, within=pyo.NonNegativeReals)
    m.x_flow = pyo.Var(m.RRx, within=pyo.NonNegativeReals)

    if use_shortage_slack:
        m.u_short = pyo.Var(m.R, within=pyo.NonNegativeReals)

    # Objective (convex QP)
    def llp_obj(mm: pyo.ConcreteModel):
        man = sum(mm.c_mod_man[r] * mm.x_man[r] for r in mm.R)

        # Domestic use costs on diagonal
        dom = sum(mm.c_mod_dom_use[r] * mm.x_flow[r, r] for r in mm.R)

        # Trade: transport + specific tariff
        trade = sum((mm.s_ship[e, r] + mm.T[e, r]) * mm.x_flow[e, r] for (e, r) in mm.RR)

        # Optional shortage penalty
        if use_shortage_slack:
            short_pen = sum(mm.c_pen_llp[r] * mm.u_short[r] for r in mm.R)
        else:
            short_pen = 0.0

        # tiny convex regularization (pins down solution/duals)
        reg = 0.5 * eps_reg * (
            sum(mm.x_man[r] ** 2 for r in mm.R)
            + sum(mm.x_flow[e, r] ** 2 for (e, r) in mm.RRx)
            + (sum(mm.u_short[r] ** 2 for r in mm.R) if use_shortage_slack else 0.0)
        )

        return man + dom + trade + short_pen + reg

    m.LLP_OBJ = pyo.Objective(rule=llp_obj, sense=pyo.minimize)

    # (MB) Module balance: domestic + imports (+ shortage slack) = committed demand
    def mod_balance(mm: pyo.ConcreteModel, r: str):
        inflow = sum(mm.x_flow[e, r] for e in mm.R)  # includes diagonal
        if use_shortage_slack:
            return inflow + mm.u_short[r] == mm.d_offer[r]
        return inflow == mm.d_offer[r]

    m.mod_balance = pyo.Constraint(m.R, rule=mod_balance)

    # (PB) Production balance: manufacturing = domestic use + exports (all outflows)
    def prod_balance(mm: pyo.ConcreteModel, r: str):
        return mm.x_man[r] == sum(mm.x_flow[r, i] for i in mm.R)

    m.prod_balance = pyo.Constraint(m.R, rule=prod_balance)

    # (CAP) Manufacturing capacity: x_man <= q_man
    m.man_cap = pyo.Constraint(m.R, rule=lambda mm, r: mm.x_man[r] <= mm.q_man[r])

    return m
