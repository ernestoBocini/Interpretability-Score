#!/bin/bash

# Interpretability Mini-Benchmark Runner
# Wrapper script for interpretability_score.py with defaults and validation

# EXAMPLES
# # Basic usage with defaults
# ./run_benchmark.sh data.csv neuron_map.json results.csv

# # Custom parameters
# ./run_benchmark.sh data.csv neuron_map.json results.csv --clean-level 4 --quiet

# # Custom weights and threshold
# ./run_benchmark.sh data.csv neuron_map.json results.csv --weights "0.4 0.3 0.2 0.1" --threshold-percentile 0.9

# # Different human score column
# ./run_benchmark.sh data.csv neuron_map.json results.csv --human-score-col "human_rating"

set -e  # Exit on any error

# Default values
DEFAULT_CLEAN_LEVEL=5
DEFAULT_THRESHOLD_PERCENTILE=0.95
DEFAULT_RECOGNIZABLE_LEVELS="3 4 5"
DEFAULT_MIN_P_VALUE=0.05
DEFAULT_HUMAN_SCORE_COL="soft_correct"
DEFAULT_WEIGHTS="0.3 0.3 0.2 0.2"
PYTHON_SCRIPT="interpretability_score.py"

# Function to display usage
show_usage() {
    cat << EOF
Usage: $0 <data_file> <neuron_map_file> <results_file> [OPTIONS]

Required arguments:
  data_file         CSV file containing experiment data
  neuron_map_file   JSON file mapping neuron IDs to concept names  
  results_file      Output CSV file for results

Optional arguments:
  --clean-level LEVEL              Level considered as clean images (default: $DEFAULT_CLEAN_LEVEL)
  --threshold-percentile FLOAT     Percentile for high-activation threshold (default: $DEFAULT_THRESHOLD_PERCENTILE)
  --recognizable-levels "L1 L2.." Levels considered human-recognizable (default: "$DEFAULT_RECOGNIZABLE_LEVELS")
  --min-p-value FLOAT              Minimum p-value for human consistency (default: $DEFAULT_MIN_P_VALUE)
  --human-score-col COLUMN         Column name for human recognition scores (default: $DEFAULT_HUMAN_SCORE_COL)
  --weights "S C R H"              Weights for S C R H scores (default: "$DEFAULT_WEIGHTS")
  --quiet                          Run without progress output
  --help                           Show this help message

Examples:
  $0 data.csv neuron_map.json results.csv
  $0 data.csv neuron_map.json results.csv --clean-level 4 --quiet
  $0 data.csv neuron_map.json results.csv --weights "0.4 0.3 0.2 0.1" --threshold-percentile 0.9
EOF
}

# Check for help flag or insufficient arguments
if [ $# -lt 3 ] || [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    show_usage
    exit 0
fi

# Required arguments
DATA_FILE="$1"
NEURON_MAP_FILE="$2"
RESULTS_FILE="$3"
shift 3

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script '$PYTHON_SCRIPT' not found in current directory"
    echo "Please ensure interpretability_score.py is in the same directory as this script"
    exit 1
fi

# Check if required input files exist
if [ ! -f "$DATA_FILE" ]; then
    echo "Error: Data file '$DATA_FILE' not found"
    exit 1
fi

if [ ! -f "$NEURON_MAP_FILE" ]; then
    echo "Error: Neuron map file '$NEURON_MAP_FILE' not found"
    exit 1
fi

# Check if Python and required packages are available
if ! python3 -c "import pandas, numpy, scipy" 2>/dev/null; then
    echo "Error: Required Python packages not found"
    echo "Please install: pip install pandas numpy scipy"
    exit 1
fi

# Build Python command with arguments
PYTHON_CMD="python3 $PYTHON_SCRIPT \"$DATA_FILE\" \"$NEURON_MAP_FILE\" \"$RESULTS_FILE\""

# Process optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --clean-level)
            if [[ -n "$2" ]] && [[ "$2" =~ ^[0-9]+$ ]]; then
                PYTHON_CMD="$PYTHON_CMD --clean-level $2"
                shift 2
            else
                echo "Error: --clean-level requires a valid integer"
                exit 1
            fi
            ;;
        --threshold-percentile)
            if [[ -n "$2" ]] && [[ "$2" =~ ^[0-9]*\.?[0-9]+$ ]]; then
                PYTHON_CMD="$PYTHON_CMD --threshold-percentile $2"
                shift 2
            else
                echo "Error: --threshold-percentile requires a valid float"
                exit 1
            fi
            ;;
        --recognizable-levels)
            if [[ -n "$2" ]]; then
                PYTHON_CMD="$PYTHON_CMD --recognizable-levels $2"
                shift 2
            else
                echo "Error: --recognizable-levels requires space-separated integers"
                exit 1
            fi
            ;;
        --min-p-value)
            if [[ -n "$2" ]] && [[ "$2" =~ ^[0-9]*\.?[0-9]+$ ]]; then
                PYTHON_CMD="$PYTHON_CMD --min-p-value $2"
                shift 2
            else
                echo "Error: --min-p-value requires a valid float"
                exit 1
            fi
            ;;
        --human-score-col)
            if [[ -n "$2" ]]; then
                PYTHON_CMD="$PYTHON_CMD --human-score-col \"$2\""
                shift 2
            else
                echo "Error: --human-score-col requires a column name"
                exit 1
            fi
            ;;
        --weights)
            if [[ -n "$2" ]]; then
                PYTHON_CMD="$PYTHON_CMD --weights $2"
                shift 2
            else
                echo "Error: --weights requires four space-separated floats"
                exit 1
            fi
            ;;
        --quiet)
            PYTHON_CMD="$PYTHON_CMD --quiet"
            shift
            ;;
        *)
            echo "Error: Unknown option $1"
            show_usage
            exit 1
            ;;
    esac
done

# Display configuration (unless quiet mode)
if [[ "$PYTHON_CMD" != *"--quiet"* ]]; then
    echo "INTERPRETABILITY MINI-BENCHMARK"
    echo "=================================================="
    echo "Data file: $DATA_FILE"
    echo "Neuron map: $NEURON_MAP_FILE"
    echo "Results output: $RESULTS_FILE"
    echo "Python script: $PYTHON_SCRIPT"
    echo ""
    echo "Running benchmark..."
fi

# Execute the Python script
eval "$PYTHON_CMD"

# Check if results were generated and display summary
if [ -f "$RESULTS_FILE" ] && [[ "$PYTHON_CMD" != *"--quiet"* ]]; then
    echo ""
    echo "Summary statistics:"
    echo "=================="
    
    # Count total neurons and calculate basic stats
    awk -F',' '
    NR==1 {next}
    {
        count++
        sum+=$7
        if($7>max || max=="") max=$7
        if(min=="" || $7<min) min=$7
    }
    END {
        if(count>0) {
            printf "Total neurons analyzed: %d\n", count
            printf "Average InterpScore: %.3f\n", sum/count
            printf "Highest InterpScore: %.3f\n", max
            printf "Lowest InterpScore: %.3f\n", min
        }
    }' "$RESULTS_FILE"
    
    echo ""
    echo "Benchmark analysis complete."
elif [ ! -f "$RESULTS_FILE" ]; then
    echo "Error: Results file was not created"
    exit 1
fi