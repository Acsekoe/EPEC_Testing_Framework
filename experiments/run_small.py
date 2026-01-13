from __future__ import annotations

from pathlib import Path
import sys

# Allow running from repo root without installing the package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from epec.data.example_case import make_example
from epec.algorithms.gauss_seidel import solve_gauss_seidel
from epec.utils.results_excel import save_run_results_excel


def _fmt_arcs(d):
    lines = []
    for (e, r), v in sorted(d.items()):
        lines.append(f"  {e}->{r}: {v:,.6f}")
    return "\n".join(lines)


if __name__ == "__main__":
    sets, params, theta0 = make_example()

    # ---- run config (single source of truth) ----
    run_cfg = {
        "max_iter": 100,
        "tol": 1e-2,
        "damping": 0.8,
        "price_sign": -1.0,
        "eps_pen": 1e-8,
        "eps_reg": 1e-8,   # start here; later try 1e-7, 1e-8
        "M_dual": 1e6,
        #"kkt_mode": "bigM",     # robust (MIQP)
        "kkt_mode": "bilinear", # no binaries, but nonconvex bilinear equalities
        "use_shortage_slack": False,
    }



    ipopt_opts = {
        "tol": 1e-7,
        "max_iter": 4000,
        "print_level": 5,
    }

    theta_star, hist = solve_gauss_seidel(
        sets=sets,
        params=params,
        theta0=theta0,
        run_cfg=run_cfg,
        verbose=True,
    )


    # Save one Excel file per run
    xlsx_path = save_run_results_excel(
        project_root=PROJECT_ROOT,
        sets=sets,
        params=params,
        theta_star=theta_star,
        hist=hist,
        run_cfg=run_cfg,
        filename_prefix="run_small",
)

    print("\n=== Final theta ===")
    print("q_man:", theta_star.q_man)
    print("d_offer:", theta_star.d_offer)
    print("tau (trade arcs):\n" + _fmt_arcs(theta_star.tau))
    print("markup (trade arcs):\n" + _fmt_arcs(theta_star.markup))
    print(f"\nSaved Excel results to: {str(xlsx_path)}")
