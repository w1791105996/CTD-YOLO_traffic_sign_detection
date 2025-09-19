# CTD-YOLO: Multi-scale YOLOv8 for Traffic Sign Detection in Complex Environments

This repository contains the implementation of **CTD-YOLO**, an enhanced YOLOv8-based model for traffic sign detection in complex environments.

## 📖 Introduction
Traffic sign detection in complex environments (e.g., rain, snow, fog, and blur) remains a challenging task due to image degradation and multi-scale variations. CTD-YOLO incorporates three key improvements over standard YOLOv8:

1. **MAA (Moving Average Aggregation)** - Stabilizes training and improves model convergence
2. **WIOU Loss** - Enhanced bounding box regression with wise IoU calculation
3. **E-DCN (Enhanced Deformable Convolution Network)** - Adaptive feature extraction for better object detection

## 🚀 Key Features
- **Advanced Loss Function**: WIOU loss with monotonous focusing mechanism and distance-based weighting
- **Model Stabilization**: MAA mechanism for exponential moving average of model weights
- **Adaptive Convolution**: E-DCN for deformable convolution with enhanced feature extraction
- **Multi-scale Detection**: Optimized for small and degraded traffic signs
- **Weather Robustness**: Enhanced performance in rain, snow, fog conditions
- **58-class Support**: Complete CE-CCTSDB dataset compatibility

## 📁 Project Structure
```
CTD-YOLO_traffic_sign_detection/
├── data/
│   ├── ce-cctsdb.yaml              # Dataset configuration
│   └── hyps/
│       └── hyp.scratch-low.yaml    # Training hyperparameters
├── models/
│   ├── common.py                   # Basic modules (Conv, C2f, SPPF, etc.)

- **CDM**: Classification Denoising Module for preprocessing
  - Challenge Classifier (0.25M parameters, based on FastestDet)
  - Denoising Blocks for rain, snow, fog, blur conditions
- **Backbone**: Enhanced with DCNv3 deformable convolutions
- **Neck**: Improved feature pyramid network
- **Head**: Optimized detection head with WIOU v3 loss
- **EMA**: Model weight averaging for better convergence

## CDM (Classification Denoising Module)

CDM is an innovative preprocessing module that contains two main components:

### 1. Challenge Classifier
- **Architecture**: Lightweight network based on FastestDet
- **Parameters**: Approximately 0.25M parameters
- **Function**: Identify image degradation types (rain, snow, fog, blur, normal)
- **Training**: Uses Adam optimizer with learning rate 0.001 and ReduceLROnPlateau scheduling

### 2. Denoising Blocks
- **Rain Denoising**: Improved version based on multi-scale collaborative method
- **Snow Denoising**: Based on hierarchical dual-tree complex wavelet representation
- **Fog Denoising**: U-Net based multi-scale defogging network with Strengthen-Operate-Subtract structure
- **Blur Denoising**: Based on self-supervised meta-assisted learning method
- **Normal Processing**: Direct pass-through without processing

## Performance Optimizations

### Training Strategy
- **Optimizer**: Adam with initial learning rate 0.001
- **Batch Size**: 32 (optimized for memory efficiency)
- **Epochs**: 200 (increased from standard 100)
- **Learning Rate Scheduler**: ReduceLROnPlateau (patience=3, factor=0.5)

### Loss Function
- **WIOU v3**: Advanced IoU loss with optimized parameters (α=1.9, δ=3)
- **Multi-scale training**: Enhanced for various input resolutions

### Data Processing
- **CDM preprocessing**: Automatic challenge classification and denoising
- **High-resolution support**: Automatic downsampling for 2048×2048 and 1024×768 images
- **Enhanced augmentation**: HSV color space transformations for weather adaptation
- **On-the-fly processing**: Dynamic data augmentation during training

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd CTD-YOLO_traffic_sign_detection

# Install dependencies
pip install -r requirements.txt
```

## Usage

### CDM Training

First train the CDM classifier:

```bash
# Create challenge classification dataset directory structure
python utils_cdm.py

# Train CDM classifier
python train_cdm.py --data-dir data/challenge_classification --epochs 100 --batch-size 32
```

### CTD-YOLO Training

```bash
# Training with CDM
python train.py --data data/ce-cctsdb.yaml --cfg models/ctd_yolo_config.yaml --epochs 200 --batch-size 32 --use-cdm --cdm-weights runs/train_cdm/exp/weights/best_cdm.pt

# Basic training without CDM
python train.py --data data/ce-cctsdb.yaml --cfg models/ctd_yolo_config.yaml --epochs 200 --batch-size 32
```

### Inference

```bash
# Inference with CDM
python detect.py --weights runs/train/exp/weights/best.pt --source data/images --use-cdm --cdm-weights runs/train_cdm/exp/weights/best_cdm.pt

# Basic inference
python detect.py --weights runs/train/exp/weights/best.pt --source data/images

# Video inference
python detect.py --weights runs/train/exp/weights/best.pt --source path/to/video.mp4 --use-cdm --cdm-weights runs/train_cdm/exp/weights/best_cdm.pt
```

### CDM Demo

```bash
# Generate CDM demo script
python utils_cdm.py

# Run CDM demo
python demo_cdm.py --image path/to/test/image.jpg --weights runs/train_cdm/exp/weights/best_cdm.pt
```

### Validation

```bash
# Validate model performance
python val.py --weights runs/train/exp/weights/best.pt --data data/ce-cctsdb.yaml --use-cdm --cdm-weights runs/train_cdm/exp/weights/best_cdm.pt
```

## Dataset

### Traffic Sign Detection Dataset
The model uses the CE-CCTSDB (Chinese Traffic Sign Detection Benchmark) dataset:

- **Training images**: High-resolution traffic sign images in various weather conditions
- **Annotations**: YOLO format bounding box annotations
- **Classes**: 58 different traffic sign categories

### CDM Challenge Classification Dataset
CDM requires an additional challenge classification dataset:

```
data/challenge_classification/
├── train/
│   ├── rain/       # Rainy images
│   ├── snow/       # Snowy images
│   ├── fog/        # Foggy images
│   ├── blur/       # Blurred images
│   └── normal/     # Normal images
├── val/
└── test/
```

### Dataset Structure
```
data/
├── ce-cctsdb.yaml              # Detection dataset configuration
├── cdm_config.yaml             # CDM configuration file
├── challenge_classification/   # CDM training data
├── images/
│   ├── train/                  # Training images
│   ├── val/                    # Validation images
│   └── test/                   # Test images
└── labels/
    ├── train/                  # Training labels
    ├── val/                    # Validation labels
    └── test/                   # Test labels
```

## Configuration Files

### Model Configuration
- `models/ctd_yolo_config.yaml`: Model architecture configuration
- `data/hyps/hyp.scratch-low.yaml`: Hyperparameter configuration
- `data/cdm_config.yaml`: CDM module configuration

### CDM Configuration
```yaml
# Challenge Classifier configuration
classifier:
  input_size: 224
  num_classes: 5
  dropout: 0.2

# Training configuration
training:
  batch_size: 32
  epochs: 100
  learning_rate: 0.001
  weight_decay: 1e-4
```

### Key Hyperparameters
```yaml
lr0: 0.001                  # Initial learning rate
momentum: 0.9               # Momentum for optimizer
weight_decay: 0.0005        # Weight decay
warmup_epochs: 3.0          # Warmup epochs
box: 0.05                   # Box loss gain
cls: 0.5                    # Class loss gain
obj: 1.0                    # Object loss gain
```

## Results

The CTD-YOLO model with CDM demonstrates significant improvements:

- **CDM Preprocessing**: Significantly improves image quality in complex environments
- **Enhanced accuracy** in complex weather conditions
- **Improved robustness** to motion blur and lighting variations
- **Better generalization** across different traffic sign types
- **Optimized inference speed** for real-time applications

## Model Components

### 1. CDM (Classification Denoising Module)
New preprocessing module that includes:
- Challenge Classifier: Lightweight challenge classifier (0.25M parameters)
- Denoising Blocks: Denoising modules for different degradation types

### 2. EMA Mechanism
Exponential Moving Average of model weights for improved training stability and better convergence.

### 3. WIOU v3 Loss
Advanced IoU-based loss function with:
- Optimized gradient flow
- Better handling of small objects
- Improved localization accuracy

### 4. DCNv3 Integration
Deformable convolutions for:
- Adaptive receptive fields
- Enhanced feature extraction
- Better handling of irregular shapes

## File Structure

```
CTD-YOLO_traffic_sign_detection/
├── models/
│   ├── cdm.py                  # CDM module implementation
│   ├── ctd_yolo.py            # Main model
│   ├── common.py              # Common components
│   ├── losses.py              # Loss functions
│   ├── maa.py                 # MAA mechanism
│   └── edcn.py                # DCNv3 implementation
���── data/
│   ├── cdm_config.yaml        # CDM configuration
│   └── hyps/
├── train_cdm.py               # CDM training script
├── utils_cdm.py               # CDM utility functions
├── train.py                   # Main training script
├── detect.py                  # Detection script
├── val.py                     # Validation script
└── README.md
```


## Acknowledgments

- CE-CCTSDB dataset contributors
- FastestDet for lightweight classifier design
- Research community for continuous improvements or issues, please open an issue in this repository.
