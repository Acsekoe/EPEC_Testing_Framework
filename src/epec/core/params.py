from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class Params:
    # LLP costs
    c_mod_man: Dict[str, float]                  # c^{mod,man}_r
    c_mod_dom_use: Dict[str, float]              # c^{mod,dom.use}_r
    s_ship: Dict[Tuple[str, str], float]         # s^{ship}_{e,r} (transport cost, €/unit)
    c_pen_llp: Dict[str, float]                  # optional shortage penalty in LLP (€/unit)

    # NEW: ULP penalty for choosing d_offer < D_hat
    c_dem_short_ulp: Dict[str, float]            # c^{dem,short}_r

    # ULP penalty on unmet offered demand (u_dem)
    c_pen_ulp: Dict[str, float]                  # c^{pen,ulp}_r

    # ULP bounds ("hats") and tariff upper bounds
    D_hat: Dict[str, float]                      # \hat{D}^{mod}_r
    Q_man_hat: Dict[str, float]                  # \hat{Q}^{mod,man}_r
    T_ub: Dict[Tuple[str, str], float]            # \overline{T}^{mod}_{e\to r} (specific tariff, €/unit)


