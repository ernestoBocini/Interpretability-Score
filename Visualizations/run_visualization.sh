#!/bin/bash

# run_visualization.sh
# Script to generate publication-ready benchmark visualizations
# 
# Usage: ./run_visualization.sh [config_file]
# Example: ./run_visualization.sh config.yaml

set -e  # Exit on any error

# Default configuration
DEFAULT_CONFIG="config.yaml"
CONFIG_FILE=${1:-$DEFAULT_CONFIG}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}"
}

# Check if Python is available
check_python() {
    if ! command -v python3 &> /dev/null; then
        print_error "python3 could not be found. Please install Python 3.7+."
        exit 1
    fi
    
    python_version=$(python3 --version | cut -d' ' -f2)
    print_status "Using Python version: $python_version"
}

# Check if required packages are installed
check_dependencies() {
    print_status "Checking Python dependencies..."
    
    python3 -c "
import sys
required_packages = [
    'pandas', 'numpy', 'matplotlib', 'seaborn', 
    'scipy', 'yaml', 'pathlib'
]

missing_packages = []
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        missing_packages.append(package)

if missing_packages:
    print(f'Missing packages: {missing_packages}')
    print('Install with: pip install ' + ' '.join(missing_packages))
    sys.exit(1)
else:
    print('All required packages are available.')
"
}

# Validate configuration file
validate_config() {
    if [[ ! -f "$CONFIG_FILE" ]]; then
        print_error "Configuration file not found: $CONFIG_FILE"
        print_status "Creating default configuration file..."
        
        # Create default config if it doesn't exist
        cat > "$CONFIG_FILE" << 'EOF'
# Default configuration for benchmark visualization
data:
  input_file: "benchmark_results/benchmark_results.csv"
  output_dir: "paper_figures"

components:
  components: ["S", "C", "R", "H"]
  labels:
    S: "Selectivity"
    C: "Causality"
    R: "Robustness"
    H: "Human Consistency"
    InterpScore: "InterpScore"
  colors:
    S: "#1f77b4"
    C: "#d62728"
    R: "#2ca02c"
    H: "#ff7f0e"
    InterpScore: "#9467bd"
  composite_score_name: "InterpScore"

plotting:
  figure_format: "svg"
  dpi: 300
  figure_width: 12.0
  figure_height: 8.0
  font_size_base: 12
  font_family: "serif"
  color_palette: "Set2"
  bootstrap_samples: 1000
  alpha_level: 0.05

figures:
  ranking_comparison: true
  component_distributions: true
  correlation_matrix: true

tables:
  summary_statistics: true
EOF
        print_status "Created default configuration: $CONFIG_FILE"
        print_warning "Please review and modify the configuration as needed."
    fi
    
    print_status "Using configuration file: $CONFIG_FILE"
}

# Main execution
main() {
    print_header "Benchmark Visualization Generator"
    
    # Pre-flight checks
    check_python
    check_dependencies
    validate_config
    
    print_header "Generating Visualizations"
    
    # Run the Python script
    if python3 create_paper_viz.py --config "$CONFIG_FILE"; then
        print_status "Visualization generation completed successfully!"
        
        # Show output summary
        print_header "Output Summary"
        
        # Extract output directory from config
        output_dir=$(python3 -c "
import yaml
with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)
print(config['data']['output_dir'])
")
        
        if [[ -d "$output_dir" ]]; then
            print_status "Generated files in: $output_dir"
            echo "Figures:"
            find "$output_dir" -name "*.svg" -o -name "*.pdf" -o -name "*.png" | sort | sed 's/^/  /'
            echo "Tables:"
            find "$output_dir" -name "*.txt" | sort | sed 's/^/  /'
        else
            print_warning "Output directory not found: $output_dir"
        fi
        
    else
        print_error "Visualization generation failed!"
        exit 1
    fi
}

# Run main function
main "$@"