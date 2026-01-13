from __future__ import annotations

from epec.core.sets import build_sets
from epec.core.params import Params
from epec.core.theta import theta_init_from_bounds


def make_example():
    """Small stylized 3-region example.

    This example is intentionally simple. It implements the **Option B** semantics:

    - Import tariff is a **multiplicative factor** \tau_{e->r} applied to *shipping only* in the LLP.
    - The tariff wedge collected by the importer is:
        \Delta^{tar}_{e->r} = (\tau_{e->r}-1) * s^{ship}_{e,r}
    - Exporters choose nonnegative additive markups m_{e->r} (entering LLP imports offer term).

    The rates below are toy values to create asymmetry.
    """

    # regions
    regions = ["ch", "eu", "us"]
    sets = build_sets(regions)
    R, RR = sets.R, sets.RR

    # Demand (GW)
    D_hat = {
        "ch": 293.0,
        "eu": 321.0,
        "us": 86.0,
    }

    # Production capacity (GW)
    Q_man_hat = {
        "ch": 931.0,
        "eu": 22.0,
        "us": 23.0,
    }

    # Toy ad-valorem tariff rates (decimal). We map them to a multiplicative upper bound:
    #   tau_ub = 1 + rate
    tau_pct = {
        ("ch", "eu"): 0.0,
        ("ch", "us"): 0.50,
        ("eu", "ch"): 0.10,
        ("eu", "us"): 0.1425,
        ("us", "ch"): 0.10,
        ("us", "eu"): 0.0,
    }
    for (e, r) in RR:
        tau_pct.setdefault((e, r), 0.05)

    tau_ub = {(e, r): 1.0 + float(tau_pct[(e, r)]) for (e, r) in RR}

    # Markup bounds (toy, but you should set these based on economic reasoning / scaling)
    m_ub = {(e, r): 500 for (e, r) in RR}

    # Manufacturing costs (arbitrary, keep your old toy ordering)
    c_mod_man = {"ch": 5.0, "eu": 6.0, "us": 7.0}

    # Domestic-use cost (tiny, just to keep x_dom meaningful)
    c_mod_dom_use = {r: 0.2 for r in R}

    # Shipping cost matrix (ASSUMED)
    base = 99.0
    s_ship = {(e, r): base for (e, r) in RR}
    s_ship[("eu", "us")] = 5.0
    s_ship[("us", "eu")] = 5.0
    s_ship[("ch", "eu")] = 8.0
    s_ship[("eu", "ch")] = 8.0
    s_ship[("ch", "us")] = 10.0
    s_ship[("us", "ch")] = 10.0

    # Penalties (placeholders; behavior is sensitive to these!)
    c_pen_llp = {r: 1500 for r in R}  # only used if shortage slack enabled
    c_pen_ulp = {r: 1500 for r in R}

    params = Params(
        c_mod_man=c_mod_man,
        c_mod_dom_use=c_mod_dom_use,
        s_ship=s_ship,
        c_pen_llp=c_pen_llp,
        c_pen_ulp=c_pen_ulp,
        D_hat=D_hat,
        Q_man_hat=Q_man_hat,
        tau_ub=tau_ub,
        m_ub=m_ub,
    )

    theta0 = theta_init_from_bounds(R, RR, params)
    return sets, params, theta0
