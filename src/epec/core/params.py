from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class Params:
    # LLP real costs
    c_mod_man: Dict[str, float]                  # c^{mod,man}_r
    c_mod_dom_use: Dict[str, float]              # c^{mod,dom.use}_r
    s_ship: Dict[Tuple[str, str], float]         # s^{ship}_{e,r} (€/unit), defined on trade arcs e!=r
    c_pen_llp: Dict[str, float]                  # optional shortage penalty in LLP (€/unit)

    # ULP penalty for not offering full demand
    c_pen_ulp: Dict[str, float]                  # c^{pen,ulp}_r

    # ULP bounds ("hats")
    D_hat: Dict[str, float]                      # \hat{D}^{mod}_r
    Q_man_hat: Dict[str, float]                  # \hat{Q}^{mod,man}_r

    # Strategic trade-policy / pricing bounds (Option B)
    tau_ub: Dict[Tuple[str, str], float]          # \overline{\tau}^{mod}_{e->r} (multiplicative), >= 1
    m_ub: Dict[Tuple[str, str], float]            # upper bound for markup m^{mod}_{e->r}, >= 0
