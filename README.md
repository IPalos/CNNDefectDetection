# CNN Defect Detection in Vector Fields

A deep learning project for detecting and classifying topological defects in 2D vector field data from N-MPCD (Nematic Multi-Particle Collision Dynamics) simulations. This project combines traditional topological analysis (winding number calculations) with modern CNN-based classification to identify and categorize defects in liquid crystal-like systems.

## 🎯 Project Overview

This project focuses on the automated detection and classification of topological defects in 2D vector fields, specifically:
- **Positive defects** (+): Topological charge +1/2
- **Negative defects** (-): Topological charge -1/2  
- **No defects** (0): Normal vector field regions

The system uses a hybrid approach combining:
1. **Winding number analysis** for initial defect candidate detection
2. **CNN classification** for precise defect type identification

## 📁 Project Structure

```
CNNDefectDetection/
├── Data/                          # Raw VTK simulation data
│   └── MAI_U060_L_096_A/         # N-MPCD simulation datasets
├── LabeledData/                   # Processed and labeled tensor data
├── Models/                        # Trained CNN models and metadata
│   ├── cnn_2025-08-12.pth        # Trained model weights
│   └── cnn_2025-08-12_metadata.pkl # Training metadata
├── Notebooks/                     # Jupyter notebooks for analysis
│   ├── cnn_trainer.ipynb         # Model training pipeline
│   ├── hybrid_detector.ipynb     # Hybrid detection system
│   ├── experiment_plotting.ipynb # Results visualization
│   ├── pr_curve_analysis.ipynb  # Performance analysis
│   └── timepline_plot.ipynb      # Temporal analysis
├── utils/                         # Core utilities and classes
│   ├── project_functions.py      # Main processing functions
│   ├── project_classes.py        # Data structures and models
│   └── plot_training_resuilts.py # Visualization utilities
└── requirements.txt              # Python dependencies
```

### Data Format
- **Input**: VTK files containing eigenvector and eigenvalue data
- **Processing**: 7×7 windows around defect candidates
- **Features**: 7-channel tensors (U, V, Magnitude, Orientation, Order Factor)

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd CNNDefectDetection
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Verify installation**:
```python
import torch
import vtk
import numpy as np
print("Installation successful!")
```

### Basic Usage

#### 1. Load and Process VTK Data
```python
import utils.project_functions as pf

# Load VTK file
eigen_vals, eigen_vecs = pf.load_and_pad_vtk("path/to/simulation.vtk")

# Find defect candidates using winding number
winding_numbers, flips = pf.calculate_winding_number(eigen_vecs, pad_width=3)
centroids = pf.find_candidates_by_winding_number(winding_numbers)
```

#### 2. Train CNN Model
```python
# See cnn_trainer.ipynb for complete training pipeline
model = pc.CNNClassifier()
# Training code in the notebook...
```

#### 3. Hybrid Detection
```python
# Load trained model
model = pf.load_model("Models/cnn_2025-08-12.pth")

# Predict defects
samples = pf.predict_field(model, eigen_vals, eigen_vecs, centroids, device)
```

## 🧠 Model Architecture

The CNN classifier uses a lightweight architecture optimized for 7×7×5 input tensors:

```python
class CNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(5, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 3)  # 3 classes: +, -, 0
```

### Key Features:
- **Input**: 5-channel 7×7 tensors
- **Architecture**: 2 convolutional layers + 2 fully connected layers
- **Classes**: 3 (positive defects, negative defects, no defects)
- **Regularization**: Batch normalization, dropout, weight decay

## 📊 Performance Metrics

The trained model achieves:
- **Validation F1-Score**: ~0.91
- **Validation Accuracy**: ~91%
- **Precision/Recall**: Balanced across all classes

### Class Distribution:
- Positive defects (+): 705 samples
- Negative defects (-): 637 samples  
- No defects (0): 643 samples

## 🔧 Core Functions

### Data Processing
- `load_and_pad_vtk()`: Load VTK files with padding
- `build_tensor()`: Extract 7×7 multi-channel features
- `calculate_winding_number()`: Compute topological charge
- `find_candidates_by_winding_number()`: Locate defect candidates

### Model Operations
- `train_model()`: Complete training pipeline
- `predict_field()`: Inference on vector fields
- `generate_pseudo_labels()`: Semi-supervised learning

### Visualization
- `plot_vector_field_with_confidence()`: Visualize predictions
- `plot_experiments_evolution()`: Temporal analysis
- `plot_training_results()`: Training metrics

## 📈 Notebooks Guide

### `cnn_trainer.ipynb`
Complete training pipeline including:
- Data loading and preprocessing
- Model training with class balancing
- Performance evaluation
- Model saving and metadata

### `hybrid_detector.ipynb`
Hybrid detection system demonstrating:
- Winding number candidate detection
- CNN-based classification
- Confidence-based filtering
- Results visualization

### `experiment_plotting.ipynb`
Advanced visualization and analysis:
- Multi-frame temporal analysis
- Defect evolution tracking
- Statistical summaries
- Publication-ready figures

## 🎛️ Configuration

### Training Parameters
```python
epochs = 50
batch_size = 64
learning_rate = 0.001
weight_decay = 0.0001
```

### Data Parameters
```python
window_size = 7          # Feature extraction window
pad_width = 3            # VTK padding
confidence_threshold = 0.8  # Pseudo-label threshold
```

## 📚 Dependencies

- **PyTorch**: Deep learning framework
- **VTK**: VTK file processing
- **NumPy/SciPy**: Numerical computations
- **Matplotlib/Seaborn**: Visualization
- **scikit-learn**: Metrics and evaluation
- **Pandas**: Data manipulation

## 🔬 Research Applications

This project is designed for:
- **Liquid crystal defect analysis**
- **Topological phase transitions**
- **N-MPCD simulation validation**
- **Automated defect tracking**
- **Temporal defect evolution studies**

## 📄 Citation

If you use this code in your research, please cite:
```
[Add your citation information here]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Contact

For questions or collaboration, please contact:
- email: ignaciopalos.r@gmail.com

## 📋 License

[Add your license information here]

---
