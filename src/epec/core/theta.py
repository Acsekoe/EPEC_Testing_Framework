from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class Theta:
    # strategic upper-level variables (current iterate)
    q_man: Dict[str, float]                      # q^{mod,man}_r
    d_offer: Dict[str, float]                    # d^{mod}_r
    tau: Dict[Tuple[str, str], float]            # \tau^{mod}_{e->r} on trade arcs (e!=r), multiplicative (>=1)
    markup: Dict[Tuple[str, str], float]         # m^{mod}_{e->r} on trade arcs (e!=r), >=0

    def copy(self) -> "Theta":
        """Return a fresh Theta with new dict objects."""
        return Theta(
            q_man=self.q_man.copy(),
            d_offer=self.d_offer.copy(),
            tau=self.tau.copy(),
            markup=self.markup.copy(),
        )


def theta_init_from_bounds(R, RR, params) -> Theta:
    """Safe initializer consistent with Option B (tau multiplicative, markup additive)."""
    q_man = {r: 0.8 * params.Q_man_hat[r] for r in R}
    d_offer = {r: 0.8 * params.D_hat[r] for r in R}

    # tau in [1, tau_ub]
    tau = {}
    for (e, r) in RR:
        ub = float(params.tau_ub[(e, r)])
        tau[(e, r)] = 0.5 * (1.0 + ub)

    # markup starts at 0
    markup = {(e, r): 0.0 for (e, r) in RR}
    return Theta(q_man=q_man, d_offer=d_offer, tau=tau, markup=markup)
