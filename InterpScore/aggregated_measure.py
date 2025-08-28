def calculate_interpretability_score(S, C, R, H, alpha=0.25, beta=0.25, gamma=0.25, delta=0.25):
    """
    Calculate overall Interpretability Score
    
    InterpScore(N,X) = αS(N,X) + βC(N,X) + γR(N,X) + δH(N,X)
    where α + β + γ + δ = 1
    """
    # Ensure weights sum to 1
    total_weight = alpha + beta + gamma + delta
    alpha /= total_weight
    beta /= total_weight
    gamma /= total_weight
    delta /= total_weight
    
    interp_score = alpha * S + beta * C + gamma * R + delta * H
    
    return interp_score
