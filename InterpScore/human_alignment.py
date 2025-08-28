import numpy as np


def calculate_human_consistency_score(high_activation_images, human_labels):
    """
    Calculate Human Consistency Score H(N,X)
    
    H(N,X) = (1/n) * Σ h_i
    where h_i is 1 if human labels image i as concept X, 0 otherwise
    """
    if len(high_activation_images) == 0:
        return 0.0
    
    # Calculate proportion of high-activation images correctly labeled by humans
    human_consistency = np.mean(human_labels)
    
    return human_consistency