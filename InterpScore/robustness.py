import numpy as np


def calculate_robustness_score(exp_df, concept_name, activation_col):
    """
    Robustness R(N,X): stability under benign transforms (levels 1–2)
    and small adversarial stress (level 3), using only human-recognizable items.
    DeepDream (level 4) is excluded.
    Returns (0,1], higher = more robust.
    """
    eps = 1e-8
    concept = concept_name.lower()
    df = exp_df.copy()

    # --- Clean baseline (level 5) ---
    clean = df[(df['ground_truth'] == concept) & (df['level'] == 5)][activation_col].dropna()
    if clean.empty:
        return 0.0
    # use absolute activations to avoid sign cancellation; remove abs() if you prefer
    mu_clean = float(np.mean(np.abs(clean)))

    def _sigma_ratio(r):
        # symmetric, bounded mapping: σ(r) = exp(-|log r|) = min(r, 1/r) ∈ (0,1]
        r = max(r, eps)
        return float(np.exp(-abs(np.log(r))))

    # --- Benign invariance (levels 1–2) ---
    ben = df[
        (df['ground_truth'] == concept) &
        (df['level'].isin([1, 2])) &
        (df['soft_correct'] == 1)
    ][activation_col].dropna()
    R_inv = None
    if not ben.empty:
        mu_ben = float(np.mean(np.abs(ben)))
        r_ben = (mu_ben + eps) / (mu_clean + eps)
        R_inv = _sigma_ratio(r_ben)

    # --- Adversarial stress (level 3) ---
    adv = df[
        (df['ground_truth'] == concept) &
        (df['level'] == 3) &
        (df['soft_correct'] == 1)
    ][activation_col].dropna()
    R_adv = None
    if not adv.empty:
        mu_adv = float(np.mean(np.abs(adv)))
        r_adv = (mu_adv + eps) / (mu_clean + eps)
        R_adv = _sigma_ratio(r_adv)

    parts = [p for p in (R_inv, R_adv) if p is not None]
    if not parts:
        return 0.0
    return float(np.mean(parts))