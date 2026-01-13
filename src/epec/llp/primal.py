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
    """Build the follower (LLP) primal (Option B).

    LLP takes (d_offer, q_man, tau, markup) as given/strategic variables
    (they are model Vars here, fixed/free by the upper-level player).

    Objective (LaTeX Option B):
      min  sum_r c_man[r]*x_man[r]
         + sum_r c_dom_use[r]*x_flow[r,r]
         + sum_{e!=r} ( markup[e,r] + tau[e,r]*s_ship[e,r] ) * x_flow[e,r]
         + optional shortage penalty (if enabled)

    Constraints:
      (MB)  x_flow[r,r] + sum_{e!=r} x_flow[e,r] (+ u_short[r]) = d_offer[r]
      (PB)  x_man[r] = x_flow[r,r] + sum_{i!=r} x_flow[r,i]
      (CAP) x_man[r] <= q_man[r]
      (NN)  nonnegativity

    A tiny strictly convex term (eps_reg) is added for numerical stability and
    to pin down duals in degenerate cases.
    """

    R, RR, RRx = sets.R, sets.RR, sets.RRx

    m = pyo.ConcreteModel("LLP_Primal")

    # ---- index sets
    m.R = pyo.Set(initialize=R)
    m.RR = pyo.Set(dimen=2, initialize=RR)     # trade arcs (e!=r)
    m.RRx = pyo.Set(dimen=2, initialize=RRx)   # all pairs (including diagonal)

    # ---- strategic / upper-level variables (embedded here)
    m.q_man = pyo.Var(m.R, within=pyo.NonNegativeReals, initialize=theta.q_man)
    m.d_offer = pyo.Var(m.R, within=pyo.NonNegativeReals, initialize=theta.d_offer)

    # multiplicative tariff factor on shipping (>=1) on trade arcs
    m.tau = pyo.Var(m.RR, within=pyo.Reals, bounds=(1.0, None), initialize=theta.tau)
    # nonnegative additive markup on trade arcs
    m.markup = pyo.Var(m.RR, within=pyo.NonNegativeReals, initialize=theta.markup)

    # ---- primal decision variables
    m.x_man = pyo.Var(m.R, within=pyo.NonNegativeReals, initialize=0.0)
    m.x_flow = pyo.Var(m.RRx, within=pyo.NonNegativeReals, initialize=0.0)

    # optional feasibility safeguard
    if use_shortage_slack:
        m.u_short = pyo.Var(m.R, within=pyo.NonNegativeReals, initialize=0.0)

    # ---- parameters
    m.c_mod_man = pyo.Param(m.R, initialize=params.c_mod_man)
    m.c_mod_dom_use = pyo.Param(m.R, initialize=params.c_mod_dom_use)
    m.s_ship = pyo.Param(m.RR, initialize=params.s_ship)
    m.c_pen_llp = pyo.Param(m.R, initialize=params.c_pen_llp)

    # ---- objective
    def llp_obj(mm: pyo.ConcreteModel):
        man = sum(mm.c_mod_man[r] * mm.x_man[r] for r in mm.R)
        dom = sum(mm.c_mod_dom_use[r] * mm.x_flow[r, r] for r in mm.R)

        # imports/trade offers: markup + (shipping * tariff factor)
        trade = sum((mm.markup[e, r] + mm.tau[e, r] * mm.s_ship[e, r]) * mm.x_flow[e, r] for (e, r) in mm.RR)

        # optional shortage penalty
        if use_shortage_slack:
            short_pen = sum(mm.c_pen_llp[r] * mm.u_short[r] for r in mm.R)
        else:
            short_pen = 0.0

        # tiny convex regularization
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
        outflow = sum(mm.x_flow[r, i] for i in mm.R)  # includes diagonal
        return mm.x_man[r] == outflow

    m.prod_balance = pyo.Constraint(m.R, rule=prod_balance)

    # (CAP) Manufacturing capacity
    m.cap_man = pyo.Constraint(m.R, rule=lambda mm, r: mm.x_man[r] <= mm.q_man[r])

    return m
