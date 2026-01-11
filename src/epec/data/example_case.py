
from __future__ import annotations

from epec.core.sets import build_sets
from epec.core.params import Params
from epec.core.theta import theta_init_from_bounds


def make_example():
    """
    Data source: Input_Data_Overview.xlsx
      - Import sheet, block 'Current Policy 2024':
          Demand (GW), Production Capacity Module (GW)
      - TradeLogistics sheet, column 'Current Policy 2024':
          Trade Tariff (%)  -> used here only as a *relative* indicator to set a
          toy upper bound for specific tariffs (€/module).

    NOTE:
      - This example case is intentionally stylized.
      - In the reformulated model, tariffs are modelled as *specific* per-unit import
        taxes T_{e->r} (€/unit), not as ad-valorem multipliers on shipping cost.
    """
    # --- regions as in your codebase ---
    sets = build_sets(["ch", "eu", "us"])
    R, RR = sets.R, sets.RR

    # -----------------------------
    # Extracted (CURRENT POLICY 2024)
    # -----------------------------
    # Demand (GW): China=278, Europe=58.99, North America=38.265
    D_hat = {
        "ch": 278.0,
        "eu": 58.99,
        "us": 38.265,  # "North America" mapped to "us" bucket in your 3-region model
    }

    # Production Capacity Module (GW): China=931, Europe=22, North America=23
    Q_man_hat = {
        "ch": 931.0,
        "eu": 22.0,
        "us": 23.0,
    }

    # Trade Tariff (%) from TradeLogistics (decimal, used only to scale a toy T_ub)
    # China->Europe 0%, China->North America 50%, Europe->China 10%, Europe->NA 14.25%,
    # NA->China 10%, NA->Europe 0%
    tau_pct = {
        ("ch", "eu"): 0.0,
        ("ch", "us"): 0.50,
        ("eu", "ch"): 0.10,
        ("eu", "us"): 0.1425,
        ("us", "ch"): 0.10,
        ("us", "eu"): 0.0,
    }

    # Make sure every arc exists (defensive, in case you later change regions)
    for (e, r) in RR:
        tau_pct.setdefault((e, r), 0.05)

    # Convert toy ad-valorem rates to toy *specific* tariff bounds.
    # Choose a reference value (€/module) purely for this small example.
    price_ref = 1000.0
    T_ub = {(e, r): tau_pct[(e, r)] * price_ref for (e, r) in RR}

    # -----------------------------
    # Assumptions (placeholders)
    # -----------------------------
    # Manufacturing costs (arbitrary, keep your old toy ordering)
    c_mod_man = {"ch": 5.0, "eu": 6.0, "us": 9.0}

    # Domestic-use cost (tiny, just to keep x_dom meaningful)
    c_mod_dom_use = {r: 0.2 for r in R}

    # Shipping cost matrix (ASSUMED)
    # Keep it simple but not symmetric-random: EU<->US cheaper than CH<->{EU,US}
    base = 99.0
    s_ship = {(e, r): base for (e, r) in RR}
    s_ship[("eu", "us")] = 5.0
    s_ship[("us", "eu")] = 5.0
    s_ship[("ch", "eu")] = 8.0
    s_ship[("eu", "ch")] = 8.0
    s_ship[("ch", "us")] = 10.0
    s_ship[("us", "ch")] = 10.0

    # Penalties (placeholders; your model behavior will be very sensitive to these!)
    c_pen_llp = {r: 1000.0 for r in R}
    c_pen_ulp = {r: 1500.0 for r in R}
    c_dem_short_ulp = {"ch": 1200.0, "eu": 1200.0, "us": 1200.0}

    params = Params(
        c_mod_man=c_mod_man,
        c_mod_dom_use=c_mod_dom_use,
        s_ship=s_ship,
        c_pen_llp=c_pen_llp,
        c_pen_ulp=c_pen_ulp,
        D_hat=D_hat,
        Q_man_hat=Q_man_hat,
        T_ub=T_ub,
        c_dem_short_ulp=c_dem_short_ulp,
    )

    theta0 = theta_init_from_bounds(R, RR, params)
    return sets, params, theta0
