import os
import numpy as np
from helpers import get_local_image_path
from causality_score_helpers import run_complete_intervention_pipeline


def calculate_causality_score(
    exp_df, neuron_id, concept_name, activation_col,
    n_images=1,                # keep fast: 30 image by default
    clean_only=True,
    tau=0.1,                   # remap scale so raw=0.1 -> 0.5
    remap_method="ratio",
    seed=0
):
    """
    C(N,X) per Eq. (2) with a monotone remap to [0,1]:
      C_raw = 0.5 * (E[Δ0] + E[Δ2]),  Δλ = ||E^λ - E||2 / ||E||2
      C_bounded = g(C_raw),  g(u)=u/(u+tau) by default.

    Returns: C_bounded in [0,1). Prints C_raw for transparency.
    """
    print(f"  Computing causality for neuron {neuron_id} ({concept_name})...")

    concept = concept_name.lower()
    mask = (exp_df['ground_truth'] == concept)
    if clean_only:
        mask &= (exp_df['level'] == 5)
    concept_images = exp_df[mask].copy()
    if len(concept_images) == 0:
        print(f"    No images found for {concept_name} (clean_only={clean_only})")
        return 0.0

    # sample up to n_images (default 30 for speed)
    k = min(n_images, len(concept_images))
    concept_images = concept_images.sample(k, random_state=seed)

    
    deltas0, deltas2 = [], []

    for _, row in concept_images.iterrows():
        try:
            local_image_path = get_local_image_path(row['image_filename'])
            if not os.path.exists(local_image_path):
                continue

            base = run_complete_intervention_pipeline(local_image_path, neuron_id, 'none')
            abl  = run_complete_intervention_pipeline(local_image_path, neuron_id, 'ablate')
            amp  = run_complete_intervention_pipeline(local_image_path, neuron_id, 'amplify', 2.0)
            if not (base.get('success') and abl.get('success') and amp.get('success')):
                continue

            e  = base['final_embedding'][0]
            e0 = abl ['final_embedding'][0]   # λ=0
            e2 = amp ['final_embedding'][0]   # λ=2

            denom = np.linalg.norm(e)
            if denom <= 1e-12:
                continue

            deltas0.append(np.linalg.norm(e0 - e) / denom)
            deltas2.append(np.linalg.norm(e2 - e) / denom)

        except Exception as ex:
            print(f"    Skipping one image due to error: {ex}")
            continue

    if not deltas0 or not deltas2:
        return 0.0

    C_raw = 0.5 * (float(np.mean(deltas0)) + float(np.mean(deltas2)))
    C_bounded = 1.0 - np.exp(-C_raw)   # monotone, in [0,1)
    print(f"    C_raw={C_raw:.4f}  ->  C_bounded={C_bounded:.4f}")
    return C_bounded