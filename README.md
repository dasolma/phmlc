# PHMLC: Learning Curve Extrapolation Framework

A comprehensive framework for learning curve extrapolation and early stopping in neural network training, specifically designed for Prognostics and Health Management (PHM) applications.

## 📖 Overview

This project implements advanced learning curve extrapolation techniques to enable efficient neural architecture search and early stopping mechanisms in predictive maintenance scenarios. The framework leverages 62,000 learning curves obtained from training on 59 datasets using the [PHMD](https://github.com/dasolma/phmd) tool.

## 🎯 Key Features

- **Learning Curve Extrapolation**: Multiple approaches including neural networks, ARIMA models, and baseline methods
- **Early Stopping Mechanisms**: Intelligent termination of underperforming training runs
- **Multiple Neural Architectures**: Support for FCN, RNN, MSCNN, Transformer.
- **Decision Tree Rules**: Automated generation of stopping criteria based on curve characteristics
- **Comprehensive Evaluation**: Comparison against random and last-seen baselines
- **PHM Applications**: Specialized for predictive maintenance and health monitoring tasks

## 🏗️ Project Structure

```
├── README.md
└── src/
    └── phm_framework/
        ├── data/                    # Data handling and generation
        │   └── generators.py        # Synthetic data generators
        ├── models/                  # Neural network architectures
        │   ├── base.py             # Base model classes
        │   ├── fcn.py              # Fully Connected Networks
        │   ├── mscnn.py            # Multi-Scale CNN
        │   ├── protonet.py         # Prototypical Networks
        │   ├── rnn_cond.py         # Conditional RNN
        │   ├── rnn.py              # Recurrent Neural Networks
        │   ├── transformer.py      # Transformer architectures
        │   └── utils.py            # Model utilities
        ├── optimization/            # Optimization and curve analysis
        │   ├── curves/             # Learning curve analysis
        │   │   └── train.py        # Main curve training module
        │   ├── hyper_parameters.py # Hyperparameter definitions
        │   ├── train.py            # Training optimization
        │   └── utils.py            # Optimization utilities
        ├── trainers/               # Training frameworks
        │   ├── base.py             # Base trainer classes
        │   ├── net.py              # Network-specific trainers
        │   └── utils.py            # Training utilities
        ├── logging.py              # Logging infrastructure
        ├── scoring.py              # Model evaluation and scoring
        ├── typing.py               # Type definitions
        └── utils.py                # General utilities
```

## 🚀 Installation

### Prerequisites
- Python 3.8+
- TensorFlow 2.x
- scikit-learn
- pandas, numpy
- statsmodels (for ARIMA models)
- PHMD tool for dataset generation

## Methodology

### Learning Curve Extrapolation Approaches

1. **Neural Network Predictor**: Trains a neural network to predict final validation loss from partial learning curves
2. **Few-Shot Learning**: Uses ProtoNet to generate decision rules for early stopping
3. **ARIMA Models**: Classical time series forecasting for curve extrapolation
4. **Decision Trees**: Automated generation of interpretable stopping criteria

### Evaluation Metrics

- **Epochs Avoided**: Total training epochs saved through early stopping
- **Training Time Saved**: Actual computational time reduction
- **Performance Preservation**: Ability to find optimal solutions despite early stopping
- **Pruning Accuracy**: Precision of early stopping decisions

### Dataset

The framework is evaluated on 62,000 learning curves from:
- 59 different PHM datasets
- Multiple neural architectures (FCN, RNN, MSCNN, Transformer)
- Various hyperparameter configurations
- Generated using the [PHMD](https://github.com/dasolma/phmd) tool

## 📈 Results

The framework demonstrates significant computational savings while maintaining model performance quality. Key findings include:

- Substantial reduction in training time through intelligent early stopping
- Preservation of optimal model discovery rates
- Interpretable decision rules for stopping criteria
- Superior performance compared to random and heuristic baselines

## 📚 References

If you use this framework in your research, please cite:

```bibtex
@article{solis2025model,
  title={A Model for Learning-Curve Estimation in Efficient Neural Architecture Search and Its Application in Predictive Health Maintenance},
  author={Sol{\'\i}s-Mart{\'\i}n, David and Gal{\'a}n-P{\'a}ez, Juan and Borrego-D{\'\i}az, Joaqu{\'\i}n},
  journal={Mathematics},
  volume={13},
  number={4},
  pages={555},
  year={2025},
  publisher={MDPI}
}

@inproceedings{solis2024bayesian,
  title={Bayesian Model Selection Pruning in Predictive Maintenance},
  author={Solis-Martin, David and Galan-Paez, Juan and Borrego-Diaz, Joaquin},
  booktitle={International Conference on Hybrid Artificial Intelligence Systems},
  pages={263--274},
  year={2024},
  organization={Springer}
}
```

## 🔗 Related Tools

- **PHMD**: [https://github.com/dasolma/phmd](https://github.com/dasolma/phmd) - Tool used for generating the 62,000 learning curves dataset

