# 🔬 CLIP Microscope

An interactive web application for exploring what individual neurons in OpenAI's CLIP model learn from ImageNet. This tool provides deep insights into neural network interpretability through beautiful visualizations and comprehensive analysis.

## 🚀 Live Demo

**[🌐 Visit the Interactive App](https://neuronbenchmark.streamlit.app/)**

Explore 2,560+ CLIP neurons interactively with feature visualizations, top activating images, and comprehensive analysis tools.

## ✨ Features

- 🎨 **Feature Visualizations**: Lucid-generated images showing what each neuron detects
- 🖼️ **Top Activating Images**: View the ImageNet images that most strongly activate each neuron
- 📊 **Analysis Dashboard**: Statistical analysis including activation distributions and concept discovery
- 🕸️ **Neuron Networks**: Explore relationships between similar neurons
- 📈 **Global Statistics**: Dataset-wide insights and neuron comparison tools
- 🎯 **Smart Navigation**: Categorized neuron suggestions and random exploration
- 📱 **Responsive Design**: Works beautifully on desktop and mobile

## 🧠 What You Can Discover

Explore fascinating insights like:
- **Neuron 89**: Responds strongly to Donald Trump images
- **Neuron 244**: Detects Spider-Man and superhero imagery  
- **Neuron 355**: Activates on puppies and cute animals
- **Neuron 432**: Recognizes smiles and happy expressions
- **Neuron 1095**: Specializes in sunglasses detection
- And 2,555+ more neurons with unique specializations!

## 🔗 Full Development Repository

For complete source code, detailed documentation, and contribution guidelines, visit the dedicated repository:

**[📁 rebuilt-microscope-CLIP](https://github.com/ernestoBocini/rebuilt-microscope-CLIP)**

This repository contains:
- Full Streamlit application source code
- Detailed setup and deployment instructions
- Data processing pipelines
- Contributing guidelines
- Performance optimization details

## 🔬 Technical Details

- **Model**: OpenAI CLIP RN50x4 (same as main framework)
- **Layer**: Image Encoder Blocks
- **Dataset**: ImageNet (training split)
- **Neurons Analyzed**: 2,560 total
- **Images per Neuron**: Top 100 activating examples
- **Data Hosting**: Hugging Face Datasets (free & fast)

## 📊 Dataset

The underlying data is hosted on Hugging Face:
**[📁 ernestoBocini/clip-microscope-imagenet](https://huggingface.co/datasets/ernestoBocini/clip-microscope-imagenet)**

Contains:
- 256,000+ top activating ImageNet images
- 2,560 Lucid-generated feature visualizations
- Comprehensive metadata with activation statistics
- All data organized for fast web access

## 🔗 Integration with Main Framework

This Microscope tool perfectly complements the InterpScore framework:

1. **Visual Validation**: Use Microscope to visually inspect neurons identified as highly interpretable by InterpScore
2. **Hypothesis Generation**: Discover interesting neurons in Microscope, then quantify their interpretability with InterpScore
3. **Result Interpretation**: After running benchmarks, use Microscope to understand *why* certain neurons scored high
4. **Research Workflow**: 
   ```
   Microscope Discovery → InterpScore Quantification → Research Insights
   ```

## 🎯 Use Cases

Perfect for:
- **AI Researchers**: Understanding what vision models learn and validating interpretability measurements
- **Students**: Learning about neural network interpretability with interactive examples
- **Educators**: Teaching computer vision concepts through hands-on exploration
- **Curious Minds**: Exploring the inner workings of AI systems

## 📈 Performance

- **Fast Loading**: Metadata cached for optimal performance
- **Efficient Images**: Direct loading from Hugging Face CDN
- **Responsive UI**: Optimized for both desktop and mobile
- **Global Access**: No geographic restrictions

## 🚀 Quick Start

1. **Explore Online**: Visit [neuronbenchmark.streamlit.app](https://neuronbenchmark.streamlit.app/)
2. **Browse Neurons**: Use the sidebar to navigate through different neurons
3. **Analyze Patterns**: Look for interesting visual patterns and concepts
4. **Generate Hypotheses**: Note neurons that seem to detect specific concepts
5. **Validate with InterpScore**: Use the main framework to quantify interpretability

## 🤝 Contributing

For contributions, please visit the [main development repository](https://github.com/ernestoBocini/rebuilt-microscope-CLIP). Areas for improvement include:
- Additional analysis visualizations
- More neuron similarity metrics
- Enhanced UI/UX features
- Performance optimizations

## 📞 Contact

- **Main Repository**: [rebuilt-microscope-CLIP](https://github.com/ernestoBocini/rebuilt-microscope-CLIP)
- **Dataset**: [Hugging Face](https://huggingface.co/datasets/ernestoBocini/clip-microscope-imagenet)
- **Live App**: [neuronbenchmark.streamlit.app](https://neuronbenchmark.streamlit.app/)

---

*Explore the hidden patterns in AI vision models interactively! 🔍*