import numpy as np
from scipy.stats import norm

def calculate_selectivity_score(target_activations_clean, non_target_activations_clean):
    """
    Selectivity S(N,X) on CLEAN images.
    We compute Hedges' g (unbiased effect size) between target and non-target
    activations, then map to [0,1] via Φ(g/√2).
    Returns: S in [0,1]; 0.5 ≈ no separation; >0.5 means target > non-target.
    """
    x = np.asarray(target_activations_clean, dtype=float)
    y = np.asarray(non_target_activations_clean, dtype=float)

    if x.size < 2 or y.size < 2:
        return 0.5  # too few samples → neutral

    mx, my = x.mean(), y.mean()
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    nx, ny = x.size, y.size

    # pooled sample SD
    sp = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if sp <= 1e-12:
        # degenerate variance: fall back deterministically
        if np.isclose(mx, my):
            return 0.5
        return 1.0 if (mx > my) else 0.0

    d = (mx - my) / sp  # Cohen's d
    # small-sample correction → Hedges' g
    J = 1.0 - (3.0 / (4.0 * (nx + ny) - 9.0))
    g = d * J

    # map to [0,1] with normal CDF
    S = float(norm.cdf(g / np.sqrt(2.0)))
    return S



