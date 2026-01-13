from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class Params:
    # LLP costs
    c_mod_man: Dict[str, float]                  # c^{mod,man}_r
    c_mod_dom_use: Dict[str, float]              # c^{mod,dom.use}_r
    s_ship: Dict[Tuple[str, str], float]         # s^{ship}_{e,r}
    c_pen_llp: Dict[str, float]                  # only used if shortage slack is enabled

    # ULP bounds ("hats")
    D_hat: Dict[str, float]                      # \hat{D}^{mod}_r
    Q_man_hat: Dict[str, float]                  # \hat{Q}^{mod,man}_r

    # Option B strategic bounds
    tau_ub: Dict[Tuple[str, str], float]         # \overline{\tau}^{mod}_{e->r}  (>=1)
    m_ub: Dict[Tuple[str, str], float]           # \overline{m}^{mod}_{e->r}     (>=0)

    # NEW: elastic demand utility parameters (per region)
    # Utility:  U_r(d) = a_dem[r]*d - 0.5*b_dem[r]*d^2
    a_dem: Dict[str, float]
    b_dem: Dict[str, float]

    # (Optional / legacy) keep if other code still expects it; not used in elastic-demand ULP
    c_pen_ulp: Dict[str, float] | None = None
