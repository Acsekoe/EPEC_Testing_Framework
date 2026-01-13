from __future__ import annotations

import pyomo.environ as pyo

from epec.core.sets import Sets
from epec.core.params import Params
from epec.core.theta import Theta
from epec.llp.primal import build_llp_primal
from epec.llp.kkt import add_llp_kkt_bigM, add_llp_kkt_bilinear


def build_player_mpec(
    region: str,
    sets: Sets,
    params: Params,
    theta_fixed: Theta,
    *,
    kkt_mode: str = "bigM",
    eps_reg: float = 1e-7,
    eps_price_reg: float = 0.5,  # <--- NEW: price anchoring strength (start tiny)
    M_dual: float = 1e6,
    use_shortage_slack: bool = True,
    price_sign: float = -1.0,
) -> pyo.ConcreteModel:
    """Build the single-player MPEC (ULP_r + embedded LLP-KKT), Option B.

    Player r controls:
      - d_offer[r] in [0, D_hat[r]]
      - q_man[r]   in [0, Q_man_hat[r]]
      - inbound tariff factors tau[e->r] for all e!=r, with 1 <= tau <= tau_ub
      - outbound markups markup[r->i] for all i!=r, with 0 <= markup <= m_ub

    All other players' strategic variables are fixed at theta_fixed.

    Market price in region i is the dual of (MB):  p_i := price_sign * lam[i].

    Price anchoring (Fix A):
      Adds a tiny penalty to keep prices close to a baseline reference level
      p_ref[i] := c_mod_man[i] + c_mod_dom_use[i].
      This prevents the leader from exploiting follower dual non-uniqueness to
      push lam (and thus prices) to extreme values unrelated to primal costs.
    """

    r = region
    R, RR = sets.R, sets.RR

    # ---- build LLP primal (with strategic variables embedded as Vars)
    m = build_llp_primal(
        sets=sets,
        params=params,
        theta=theta_fixed,
        eps_reg=eps_reg,
        use_shortage_slack=use_shortage_slack,
    )

    # ---- Fix other players' q_man/d_offer; free this player's
    for s in R:
        if s != r:
            m.q_man[s].fix(theta_fixed.q_man[s])
            m.d_offer[s].fix(theta_fixed.d_offer[s])
        else:
            m.q_man[s].setub(params.Q_man_hat[s])
            m.d_offer[s].setub(params.D_hat[s])

    # ---- Strategic trade vars (Option B):
    # tau: importer controls inbound tau[e->r]
    # markup: exporter controls outbound markup[r->i]
    for (e, dest) in RR:
        # tau
        if dest == r:
            m.tau[e, dest].setub(params.tau_ub[(e, dest)])
        else:
            m.tau[e, dest].fix(theta_fixed.tau[(e, dest)])

        # markup
        if e == r:
            m.markup[e, dest].setub(params.m_ub[(e, dest)])
        else:
            m.markup[e, dest].fix(theta_fixed.markup[(e, dest)])

    # ---- Tight UBs for primal vars (needed for Big-M; also helps numerics)
    for s in R:
        m.x_man[s].setub(params.Q_man_hat[s])
        if use_shortage_slack:
            m.u_short[s].setub(params.D_hat[s])

    for (e, dest) in sets.RRx:
        m.x_flow[e, dest].setub(params.D_hat[dest])

    # ---- Add KKT (either Big-M complementarity or bilinear complementarity)
    if kkt_mode.lower() == "bilinear":
        add_llp_kkt_bilinear(
            m,
            sets=sets,
            params=params,
            eps_reg=eps_reg,
            use_shortage_slack=use_shortage_slack,
        )
    else:
        add_llp_kkt_bigM(
            m,
            sets=sets,
            params=params,
            eps_reg=eps_reg,
            M_dual=M_dual,
            use_shortage_slack=use_shortage_slack,
        )

    # Deactivate LLP objective; solve leader objective with KKT constraints
    m.LLP_OBJ.deactivate()

    # ---- Linearize (D_hat - d_offer[r]) (always >=0 due to ub, but keeps things explicit)
    m.dem_short = pyo.Var(within=pyo.NonNegativeReals)
    m.dem_short_lb = pyo.Constraint(expr=m.dem_short >= params.D_hat[r] - m.d_offer[r])

    # ---- Helper: tariff wedge Delta^{tar}_{e->dest} = (tau-1)*s_ship on trade arcs
    def _delta_tar(e: str, dest: str):
        return (m.tau[e, dest] - 1.0) * m.s_ship[e, dest]

    # ---- Prices (buyer prices)
    p = {i: price_sign * m.lam[i] for i in R}

    # ---- ULP objective (LaTeX Option B)
    # Sales revenue net of tariff wedge (exporter r loses the wedge on sales into dest)
    sales_dom = p[r] * m.x_flow[r, r]
    sales_exp = sum((p[i] - _delta_tar(r, i)) * m.x_flow[r, i] for i in R if i != r)
    sales_rev = sales_dom + sales_exp

    # Real costs borne by exporter r
    man_cost = params.c_mod_man[r] * m.x_man[r]
    dom_use_cost = params.c_mod_dom_use[r] * m.x_flow[r, r]
    ship_cost = sum(m.s_ship[r, i] * m.x_flow[r, i] for i in R if i != r)

    # Tariff revenue collected on imports into r
    tariff_rev = sum(_delta_tar(e, r) * m.x_flow[e, r] for e in R if e != r)

    # Market bill for committed demand in r
    demand_bill = p[r] * m.d_offer[r]

    # Penalty for not offering full demand
    dem_pen = params.c_pen_ulp[r] * m.dem_short

    # ---- Fix A: price anchoring to baseline domestic cost
    # Reference (constant): p_ref[i] := c_man[i] + c_dom_use[i]
    p_ref = {i: float(params.c_mod_man[i] + params.c_mod_dom_use[i]) for i in R}

    # Penalize deviations of prices from reference levels (tiny weight!)
    price_reg = 0.5 * eps_price_reg * sum((p[i] - p_ref[i]) ** 2 for i in R)

    m.ULP_OBJ = pyo.Objective(
        expr=(
            sales_rev
            - man_cost
            - dom_use_cost
            - ship_cost
            + tariff_rev
            - demand_bill
            - dem_pen
            - price_reg
        ),
        sense=pyo.maximize,
    )

    return m
