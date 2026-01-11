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
    price_sign: float = -1.0,
    eps_reg: float = 1e-6,
    M_dual: float = 1e6,
    kkt_mode: str = "bigM",  # "bigM" or "bilinear"
    use_shortage_slack: bool = True,
) -> pyo.ConcreteModel:
    """Build one player's best-response MPEC.

    Player r solves:
      max Pi_r  s.t.  LLP optimality (KKT) given other players' strategic vars.

    Economics consistent with the LaTeX baseline:
      - export earnings at destination price: sum_{i!=r} p_i x_{r,i}
      - tariff revenue collected by importer r: sum_{e!=r} T_{e->r} x_{e,r}
      - market bill for committed demand: - p_r d_offer[r]
      - penalty for lowering offered demand below D_hat (optional): - c_dem_short * max(D_hat - d_offer, 0)

    where p_r := price_sign * lam[r]. (You used price_sign=-1 historically because lam is negative.)

    Notes:
      - kkt_mode="bigM" gives a MIQP (binaries + bilinear leader terms).
      - kkt_mode="bilinear" avoids binaries but yields a nonconvex QCP.
        With Gurobi set NonConvex=2.
    """

    R, RR = sets.R, sets.RR
    r = region

    # Build LLP primal (strategic variables are Vars here; fixed/bounded below)
    m = build_llp_primal(
        sets=sets,
        params=params,
        theta=theta_fixed,
        eps_reg=eps_reg,
        use_shortage_slack=use_shortage_slack,
    )

    # ------------------------------------------------------------
    # Fix other players' strategic vars; free this player's
    # ------------------------------------------------------------
    for s in R:
        if s != r:
            m.q_man[s].fix(theta_fixed.q_man[s])
            m.d_offer[s].fix(theta_fixed.d_offer[s])
        else:
            m.q_man[s].setub(params.Q_man_hat[s])
            m.d_offer[s].setub(params.D_hat[s])

    # Tariffs: player r controls T[e->r] only; all other T fixed
    for (e, dest) in RR:
        if dest == r:
            m.T[e, dest].setub(params.T_ub[(e, dest)])
        else:
            m.T[e, dest].fix(theta_fixed.T[(e, dest)])

    # ------------------------------------------------------------
    # Tight UBs for primal vars (needed for Big-M; also helps numerics)
    # ------------------------------------------------------------
    for s in R:
        m.x_man[s].setub(params.Q_man_hat[s])
        if use_shortage_slack:
            m.u_short[s].setub(params.D_hat[s])

    for (e, dest) in sets.RRx:
        # Flow into region dest cannot exceed its max possible demand cap
        m.x_flow[e, dest].setub(params.D_hat[dest])

    # ------------------------------------------------------------
    # Add KKT (either Big-M complementarity or bilinear complementarity)
    # ------------------------------------------------------------
    if kkt_mode.lower() == "bilinear":
        add_llp_kkt_bilinear(
            m,
            sets=sets,
            params=params,
            eps_reg=eps_reg,
            use_shortage_slack=use_shortage_slack,
            dual_bound=M_dual,
        )
    else:
        add_llp_kkt_bigM(
            m,
            sets=sets,
            params=params,
            eps_reg=eps_reg,
            use_shortage_slack=use_shortage_slack,
            M_dual=M_dual,
        )

    # Deactivate LLP objective; solve leader objective with KKT constraints
    m.LLP_OBJ.deactivate()

    # ------------------------------------------------------------
    # Leader objective components
    # ------------------------------------------------------------
    # Linearize max(D_hat - d_offer, 0)
    m.dem_short = pyo.Var(within=pyo.NonNegativeReals)
    m.dem_short_lb = pyo.Constraint(expr=m.dem_short >= params.D_hat[r] - m.d_offer[r])

    # (1) Export earnings at destination prices (p_i = price_sign*lam[i])
    export_rev = sum(price_sign * m.lam[i] * m.x_flow[r, i] for i in R if i != r)

    # (2) Tariff revenue collected by importer r: sum_e T_{e->r} * x_{e,r}
    tariff_rev = sum(m.T[e, r] * m.x_flow[e, r] for e in R if e != r)

    # (3) Market bill for committed demand in region r: - p_r * d_offer[r]
    demand_bill = -price_sign * m.lam[r] * m.d_offer[r]

    # (4) Penalty for lowering offered demand below cap
    dem_short_pen = params.c_dem_short_ulp[r] * m.dem_short

    # small regular capacity cost to avoid corner solutions
    cap_cost = 1e-3 * m.q_man[r]

    m.ULP_OBJ = pyo.Objective(
        expr=export_rev + tariff_rev + demand_bill - dem_short_pen - cap_cost,
        sense=pyo.maximize,
    )

    return m
