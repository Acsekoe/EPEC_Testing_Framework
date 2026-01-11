from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class Theta:
    q_man: Dict[str, float]                      # q^{mod,man}_r
    d_offer: Dict[str, float]                    # d^{mod}_r
    T: Dict[Tuple[str, str], float]              # T^{mod}_{e->r}, e!=r (specific, €/unit)

    def copy(self) -> "Theta":
        """
        Return a fresh Theta with new dict objects.
        Keys are str/tuple, values are floats -> shallow dict.copy() is enough.
        """
        return Theta(
            q_man=self.q_man.copy(),
            d_offer=self.d_offer.copy(),
            T=self.T.copy(),
        )


def theta_init_from_bounds(R, RR, params) -> Theta:
    # safe initializer
    q_man = {r: 0.8 * params.Q_man_hat[r] for r in R}
    d_offer = {r: 0.8 * params.D_hat[r] for r in R}
    T = {(e, r): 0.5 * params.T_ub[(e, r)] for (e, r) in RR}
    return Theta(q_man=q_man, d_offer=d_offer, T=T)

