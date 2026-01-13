from __future__ import annotations

from epec.core.sets import build_sets
from epec.core.params import Params
from epec.core.theta import theta_init_from_bounds


def make_example():
    """3-region example with realistic-ish manufacturing costs and trade logistics.

    Option B:
      - tau_{e->r} is a multiplicative factor on shipping only (in LLP objective)
      - markup m_{e->r} is additive offer premium on trade arcs (in LLP objective)

    Data sources (from Input_Data_Overview.xlsx):
      - Manufacturing proxy: Avg module prices (USD/kW)
      - Shipping: TradeLogistics avg USD/container divided by 260.4 kW/container
    """

    regions = ["ch", "eu", "us"]
    sets = build_sets(regions)
    R, RR = sets.R, sets.RR

    # Demand (GW)  (keep your stylized values for now)
    D_hat = {"ch": 293.0, "eu": 321.0, "us": 86.0}

    # Production capacity (GW) (keep your stylized values for now)
    Q_man_hat = {"ch": 931.0, "eu": 22.0, "us": 23.0}

    # ----------------------------
    # Realistic cost magnitudes
    # ----------------------------
    # Manufacturing cost proxy (USD/kW) from Avg Module Prices (USD/W)*1000
    c_mod_man = {"ch": 163.00, "eu": 299.25, "us": 321.25}

    # Domestic-use friction (USD/kW). If you don’t want it, set to 0.0.
    # Keep it small vs manufacturing.
    c_mod_dom_use = {r: 3.0 for r in R}

    # Shipping/logistics (USD/kW), avg case (USD/container / 260.4 kW/container)
    # Directed arcs e->r
    s_ship = {(e, r): 0.0 for (e, r) in RR}  # will overwrite all RR arcs
    s_ship.update(
        {
            ("ch", "eu"): 16.118,
            ("ch", "us"): 11.133,
            ("eu", "ch"): 16.118,
            ("eu", "us"): 18.760,
            ("us", "ch"): 24.731,
            ("us", "eu"): 9.397,
        }
    )

    # ----------------------------
    # Tariff-factor bounds (tau_ub)
    # ----------------------------
    # tau = 1 + rate. Since tau is a decision var in the ULP, these are *upper bounds*.
    # Pick plausible “policy space” bounds (you can tighten/loosen later).
    tau_rate_ub = {
        ("ch", "eu"): 0.10,   # EU can charge up to 10% on CH imports (as an upper bound)
        ("ch", "us"): 0.25,   # US up to 25% on CH
        ("eu", "ch"): 0.10,
        ("eu", "us"): 0.05,
        ("us", "ch"): 0.10,
        ("us", "eu"): 0.05,
    }
    for (e, r) in RR:
        tau_rate_ub.setdefault((e, r), 0.05)
    tau_ub = {(e, r): 1.0 + float(tau_rate_ub[(e, r)]) for (e, r) in RR}

    # ----------------------------
    # Markup bounds (USD/kW)
    # ----------------------------
    # Markup should be “big enough” to allow strategic pricing but not absurd.
    # Relative to costs (163–321 + shipping ~ 10–25), 0..200 is already huge.
    m_ub = {(e, r): 200.0 for (e, r) in RR}

    # LLP shortage penalty (only relevant if you enable shortage slack)
    c_pen_llp = {r: 5000.0 for r in R}

    # ----------------------------
    # Elastic demand calibration
    # ----------------------------
    # Choose p0 around “typical delivered cost” scale (USD/kW) and a choke price.
    # This keeps the demand response in a realistic range.
    p0 = {"ch": 220.0, "eu": 320.0, "us": 260.0}         # roughly aligns with local supply costs
    p_choke = {"ch": 1200.0, "eu": 1200.0, "us": 1200.0}  # demand ~0 at very high price

    a_dem = {r: float(p_choke[r]) for r in R}
    b_dem = {r: float(p_choke[r] - p0[r]) / float(D_hat[r]) for r in R}

    # (Optional legacy, not used with elastic demand)
    c_pen_ulp = {r: 1500.0 for r in R}

    params = Params(
        c_mod_man=c_mod_man,
        c_mod_dom_use=c_mod_dom_use,
        s_ship=s_ship,
        c_pen_llp=c_pen_llp,
        D_hat=D_hat,
        Q_man_hat=Q_man_hat,
        tau_ub=tau_ub,
        m_ub=m_ub,
        a_dem=a_dem,
        b_dem=b_dem,
        c_pen_ulp=c_pen_ulp,
    )

    theta0 = theta_init_from_bounds(R, RR, params)
    return sets, params, theta0
