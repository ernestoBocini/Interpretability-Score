#!/usr/bin/env python3

import sys
import pandas as pd
import numpy as np
import json
import argparse

# Import your existing scoring functions from InterpScore folder
from InterpScore.selectivity import calculate_selectivity_score
from InterpScore.causality import calculate_causality_score
from InterpScore.robustness import calculate_robustness_score
from InterpScore.human_alignment import calculate_human_consistency_score
from InterpScore.aggregated_measure import calculate_interpretability_score

def implement_interpretability_benchmark(df, neuron_label_map, clean_level=5, 
                                       threshold_percentile=0.95, weights=None,
                                       recognizable_levels=None, min_p_value=0.05,
                                       human_score_col='soft_correct', verbose=True):
    """
    Implement the full Interpretability Mini-Benchmark
    
    Parameters:
    - df: DataFrame with experiment data
    - neuron_label_map: Dict mapping neuron IDs to concept names
    - clean_level: Level considered as 'clean' images (default: 5)
    - threshold_percentile: Percentile for high-activation threshold (default: 0.95)
    - weights: Dict with weights for S,C,R,H scores (default: {'S': 0.3, 'C': 0.3, 'R': 0.2, 'H': 0.2})
    - recognizable_levels: List of levels considered human-recognizable (default: [3,4,5])
    - min_p_value: Minimum p-value for human consistency score (default: 0.05)
    - human_score_col: Column name for human recognition scores (default: 'soft_correct')
    - verbose: Whether to print progress (default: True)
    """
    
    if recognizable_levels is None:
        recognizable_levels = [3, 4, 5]
    
    if verbose:
        print("INTERPRETABILITY MINI-BENCHMARK")
        print("=" * 50)
    
    # Filter to experiment trials
    exp_df = df[df['trial_category'] == 'experiment'].copy()
    
    # Initialize results storage
    benchmark_results = []
    
    for neuron_id, concept_name in neuron_label_map.items():
        if verbose:
            print(f"Analyzing {concept_name} (Neuron {neuron_id})...")
        
        activation_col = f'activation_{neuron_id}'
        
        if activation_col not in exp_df.columns:
            if verbose:
                print(f"  Warning: {activation_col} not found in data")
            continue
        
        # Get target and non-target activations FROM CLEAN IMAGES ONLY
        concept_clean = exp_df[(exp_df['ground_truth'] == concept_name.lower()) & 
                              (exp_df['level'] == clean_level)].copy()
        non_concept_clean = exp_df[(exp_df['ground_truth'] != concept_name.lower()) & 
                                  (exp_df['level'] == clean_level)].copy()

        if len(concept_clean) == 0 or len(non_concept_clean) == 0:
            if verbose:
                print(f"  Insufficient clean data for {concept_name}")
            continue
        
        target_activations_clean = concept_clean[activation_col].dropna()
        non_target_activations_clean = non_concept_clean[activation_col].dropna()
        
        # 1. SELECTIVITY SCORE (clean images only)
        S = calculate_selectivity_score(target_activations_clean, non_target_activations_clean)
        
        # 2. CAUSALITY SCORE (intervention-based)
        C = calculate_causality_score(exp_df, neuron_id, concept_name, activation_col)
        
        # 3. ROBUSTNESS SCORE (human-recognizable perturbed images only)
        R = calculate_robustness_score(exp_df, concept_name, activation_col, recognizable_levels)
        
        # 4. HUMAN CONSISTENCY SCORE (using clean + all concept data)
        concept_data = exp_df[exp_df['ground_truth'] == concept_name.lower()].copy()
        
        # Use high-activation threshold
        threshold = non_target_activations_clean.quantile(threshold_percentile)
        high_activation_mask = concept_data[activation_col] > threshold
        high_activation_images = concept_data[high_activation_mask]
        
        if len(high_activation_images) > 0 and human_score_col in high_activation_images.columns:
            H = calculate_human_consistency_score(
                high_activation_images[activation_col],
                high_activation_images[human_score_col],
                min_p_value
            )
        else:
            H = 0
        
        # 5. OVERALL INTERPRETABILITY SCORE
        interp_score = calculate_interpretability_score(S, C, R, H, weights)
        
        # Store results
        benchmark_results.append({
            'Neuron': neuron_id,
            'Concept': concept_name,
            'S': S,
            'C': C,
            'R': R,
            'H': H,
            'InterpScore': interp_score,
            'Notes': f"Threshold: {threshold:.4f}"
        })
        
        if verbose:
            print(f"  S: {S:.3f} | C: {C:.3f} | R: {R:.3f} | H: {H:.3f} | InterpScore: {interp_score:.3f}")
    
    # Convert to DataFrame and sort by InterpScore
    benchmark_df = pd.DataFrame(benchmark_results).sort_values('InterpScore', ascending=False)
    
    return benchmark_df

def main():
    parser = argparse.ArgumentParser(description='Run Interpretability Mini-Benchmark')
    parser.add_argument('data_file', help='CSV file containing experiment data')
    parser.add_argument('neuron_map_file', help='JSON file mapping neuron IDs to concept names')
    parser.add_argument('results_file', help='Output CSV file for results')
    
    # Optional parameters with defaults
    parser.add_argument('--clean-level', type=int, default=5, help='Level considered as clean images (default: 5)')
    parser.add_argument('--threshold-percentile', type=float, default=0.95, help='Percentile for high-activation threshold (default: 0.95)')
    parser.add_argument('--recognizable-levels', nargs='+', type=int, default=[3,4,5], help='Levels considered human-recognizable (default: 3 4 5)')
    parser.add_argument('--min-p-value', type=float, default=0.05, help='Minimum p-value for human consistency (default: 0.05)')
    parser.add_argument('--human-score-col', default='soft_correct', help='Column name for human recognition scores (default: soft_correct)')
    parser.add_argument('--weights', nargs=4, type=float, metavar=('S', 'C', 'R', 'H'), help='Weights for S C R H scores (default: 0.3 0.3 0.2 0.2)')
    parser.add_argument('--quiet', action='store_true', help='Run without progress output')
    
    args = parser.parse_args()
    
    # Parse weights if provided
    weights = None
    if args.weights:
        if len(args.weights) != 4:
            print("Error: --weights requires exactly 4 values for S C R H")
            sys.exit(1)
        weights = {'S': args.weights[0], 'C': args.weights[1], 'R': args.weights[2], 'H': args.weights[3]}
    
    try:
        # Load data
        if not args.quiet:
            print("Loading data...")
        df = pd.read_csv(args.data_file)
        
        # Load neuron label map
        with open(args.neuron_map_file, 'r') as f:
            neuron_label_map = json.load(f)
        
        # Convert string keys to integers if needed
        neuron_label_map = {int(k): v for k, v in neuron_label_map.items()}
        
        # Run benchmark
        results_df = implement_interpretability_benchmark(
            df, neuron_label_map,
            clean_level=args.clean_level,
            threshold_percentile=args.threshold_percentile,
            weights=weights,
            recognizable_levels=args.recognizable_levels,
            min_p_value=args.min_p_value,
            human_score_col=args.human_score_col,
            verbose=not args.quiet
        )
        
        # Save results
        results_df.to_csv(args.results_file, index=False)
        
        if not args.quiet:
            print(f"\nBenchmark completed! Results saved to {args.results_file}")
            print("\nTop 5 most interpretable neurons:")
            print(results_df.head().to_string(index=False))
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()