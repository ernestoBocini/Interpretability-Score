# Neural Network Interpretability Research Framework

A comprehensive toolkit for analyzing and measuring the interpretability of neural network activations, with a focus on CLIP vision models and human-aligned concept detection.

## 🤖 Model

This framework uses the **CLIP RN50x4** model as featured in the [OpenAI Multimodal Neuron Paper](https://distill.pub/2021/multimodal-neurons/). The model can be downloaded using:

```bash
wget https://raw.githubusercontent.com/openai/CLIP-featurevis/master/model.py
```

This choice ensures compatibility with existing interpretability research and provides access to the well-studied multimodal neurons identified in the OpenAI paper.

## 🎯 Overview

This repository provides tools for:
- **Extracting neural activations** from CLIP models at various layers
- **Measuring interpretability** through multiple complementary metrics
- **Running human perception experiments** to validate neural representations
- **Benchmarking interpretability** across different neurons and concepts
- **Visualizing results** and generating research insights

## 🏗️ Repository Structure

```
├── InterpScore/                    # Core interpretability measurement framework
├── Activation_Extraction/          # Tools for extracting neural activations
├── Data/                          # Datasets and data management
├── Human_Experiment_Setup/         # Human perception experiment tools
├── DeepDream_Setup/               # Neural visualization setup
├── Microscope/                    # Analysis and visualization tools
├── Visualizations/                # Paper-ready visualization generation
└── interpretability_score.py     # Main benchmark runner
```

## 🔬 Interactive Neuron Exploration

Before diving into quantitative analysis, explore neurons visually with our **CLIP Microscope** tool:

**[🌐 Launch Interactive Microscope](https://neuronbenchmark.streamlit.app/)**

This web application lets you:
- 🎨 **Visualize** what each of 2,560+ neurons detects through feature visualizations
- 🖼️ **Browse** top activating ImageNet images for any neuron
- 📊 **Analyze** activation patterns and neuron relationships
- 🔍 **Discover** interesting concepts before running interpretability benchmarks

**Perfect Research Workflow**:
```
Visual Discovery (Microscope) → Quantitative Analysis (InterpScore) → Research Insights
```

## 📊 InterpScore Framework

The **InterpScore** module is the core of this repository, providing a multi-dimensional approach to measuring neural interpretability through four key metrics:

### Metrics Overview

| Metric | Description | File | Purpose |
|--------|-------------|------|---------|
| **Selectivity (S)** | How well a neuron distinguishes target vs non-target concepts | `selectivity.py` | Measures specificity using Cohen's d |
| **Causality (C)** | How much neuron interventions affect model behavior | `causality.py` | Tests causal importance via ablation/amplification |
| **Robustness (R)** | How stable activations are across image perturbations | `robustness.py` | Evaluates consistency across noise levels |
| **Human Alignment (H)** | How well high activations correlate with human recognition | `human_alignment.py` | Validates human-interpretable patterns |

### Core Components

- **`aggregated_measure.py`** - Combines individual metrics into overall interpretability score
- **`causality_score_helpers.py`** - Helper functions for intervention analysis
- **`helpers.py`** - Shared utilities across all metrics

### Usage Example

```python
from InterpScore.selectivity import calculate_selectivity_score
from InterpScore.causality import calculate_causality_score
from InterpScore.robustness import calculate_robustness_score
from InterpScore.human_alignment import calculate_human_consistency_score
from InterpScore.aggregated_measure import calculate_interpretability_score

# Calculate individual metrics
S = calculate_selectivity_score(target_activations, non_target_activations)
C = calculate_causality_score(df, neuron_id, concept_name, activation_col)
R = calculate_robustness_score(df, concept_name, activation_col)
H = calculate_human_consistency_score(activations, human_scores)

# Combine into overall score
interpretability = calculate_interpretability_score(S, C, R, H)
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone <repository-url>
cd neural-interpretability

# Install dependencies
pip install pandas numpy scipy tensorflow pillow

# Download CLIP RN50x4 model (as used in OpenAI Multimodal Neuron Paper)
wget https://raw.githubusercontent.com/openai/CLIP-featurevis/master/model.py

# Ensure InterpScore is importable
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### 2. Explore Neurons Visually

**[🌐 Open CLIP Microscope](https://neuronbenchmark.streamlit.app/)**

Before running quantitative analysis, visually explore interesting neurons:
```
1. Browse through 2,560+ neurons
2. Look for clear conceptual patterns
3. Note neuron IDs that seem interpretable
4. Use these for targeted InterpScore analysis
```

```bash
# Download required datasets
cd Data/
./download_clip_activation_s3.sh    # Neural activation data
./download_images_data_s3.sh         # Image datasets
./download_prolific_data_s3.sh       # Human experiment data
```

### 3. Data Preparation

```bash
# Download required datasets
cd Data/
./download_clip_activation_s3.sh    # Neural activation data
./download_images_data_s3.sh         # Image datasets
./download_prolific_data_s3.sh       # Human experiment data
```

```python
from Activation_Extraction.activation_extractor import extract_activations

# Extract activations for your images
activations = extract_activations(
    image_paths=["path/to/image1.jpg", "path/to/image2.jpg"],
    layer_name="image_block_4/5/Relu_2",
    model_type="clip"
)
```

### 4. Run Interpretability Benchmark

```bash
# Basic benchmark run
python interpretability_score.py data.csv neuron_map.json results.csv

# With custom parameters
python interpretability_score.py data.csv neuron_map.json results.csv \
    --clean-level 5 \
    --threshold-percentile 0.95 \
    --weights "0.3 0.3 0.2 0.2"

# Batch processing
./run_10_neurons_scoring.sh
```

## 📁 Detailed Module Descriptions

### Activation_Extraction/
Tools for extracting neural activations from pre-trained models:
- **`activation_extractor.py`** - Main extraction interface
- **`clip_helpers.py`** - CLIP-specific utilities
- **`extract_full_layer_activations.py`** - Batch extraction for entire layers
- **`Example_Extraction.ipynb`** - Tutorial notebook

### Data/
Centralized data management with organized subdirectories:
- **`Image_Data/`** - Source images for experiments
- **`Clip_Activation_Layer/`** - Pre-computed neural activations
- **`Prolific_Data/`** - Human perception experiment results
- **`DIY/`** - Custom datasets and user-contributed data

### Human_Experiment_Setup/
Framework for running human perception studies to validate interpretability claims.

### Visualizations/
Tools for generating publication-ready figures and analysis plots:
- **`get_paper_visualizations.py`** - Automated figure generation for research papers

## 🔬 Research Applications

This framework supports various interpretability research directions:

### 1. Concept Detection Analysis
```python
# Analyze how well neurons detect specific concepts
neuron_map = {89: "cat", 156: "dog", 201: "car"}
results = run_interpretability_benchmark(data, neuron_map)
```

### 2. Layer Comparison Studies
```python
# Compare interpretability across different network layers
for layer in ["block_1", "block_2", "block_3", "block_4"]:
    activations = extract_layer_activations(images, layer)
    scores = calculate_interpretability_metrics(activations)
```

### 3. Intervention Analysis
```python
# Test causal effects of neuron interventions
from causality import test_neuron_intervention
effects = test_neuron_intervention(model, neuron_id, intervention_type="ablate")
```

## 📊 Benchmark Configuration

The interpretability benchmark is highly configurable:

### Command Line Options
```bash
--clean-level LEVEL              # Level considered as clean images (default: 5)
--threshold-percentile FLOAT     # High-activation threshold (default: 0.95)
--recognizable-levels "L1 L2.."  # Human-recognizable levels (default: "3 4 5")
--min-p-value FLOAT              # Statistical significance threshold (default: 0.05)
--human-score-col COLUMN         # Human recognition score column (default: soft_correct)
--weights "S C R H"              # Metric weights (default: "0.3 0.3 0.2 0.2")
--quiet                          # Suppress progress output
```

### Data Format Requirements

**Experiment Data CSV:**
```csv
trial_category,ground_truth,level,activation_89,activation_156,soft_correct
experiment,cat,5,0.85,0.12,0.95
experiment,dog,4,0.23,0.78,0.87
...
```

**Neuron Map JSON:**
```json
{
  "89": "cat",
  "156": "dog", 
  "201": "car"
}
```

## 📈 Expected Outputs

### Benchmark Results
The benchmark produces a comprehensive CSV with:
- Individual metric scores (S, C, R, H)
- Overall interpretability scores
- Statistical thresholds and notes
- Ranking by interpretability

### Visualization Outputs
- Neuron activation heatmaps
- Concept selectivity plots  
- Robustness across perturbation levels
- Human-AI alignment correlations

## 🤝 Contributing

We welcome contributions! Please see our contribution guidelines for:
- Adding new interpretability metrics
- Extending to other model architectures
- Improving human experiment protocols
- Adding visualization capabilities

## 📚 Citation

If you use this framework in your research, please cite:

```bibtex
@article{your_paper,
  title={Multi-Dimensional Neural Interpretability Framework},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

## 📞 Support

- **Issues**: Please use GitHub Issues for bug reports and feature requests
- **Documentation**: Check individual module READMEs for detailed usage
- **Examples**: See `Example_Extraction.ipynb` and other tutorial notebooks

## 🔬 Research Impact

This framework has been designed to support rigorous interpretability research by providing:
- **Reproducible benchmarks** with standardized metrics
- **Human-validated measurements** through perception experiments  
- **Flexible architecture** for extending to new models and domains
- **Publication-ready outputs** for research dissemination