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
│   ├── ctd_yolo.py                # Main CTD-YOLO model
│   ├── ctd_yolo_config.yaml       # Model architecture config
│   ├── edcn.py                    # E-DCN implementation
│   ├── losses.py                  # WIOU loss functions
│   └── maa.py                     # MAA mechanism
├── train.py                       # Training script
├── val.py                         # Validation script
├── detect.py                      # Inference script
├── utils.py                       # Utility functions
├── requirements.txt               # Dependencies
└── README.md
```

## 📂 Dataset
The model is designed for the **CE-CCTSDB dataset** (58 traffic sign classes):

- 📥 Download: [Baidu Pan](https://pan.baidu.com/s/1gie-eZPECoKpBxGd1vCmyQ?pwd=tu5n)  
- Password: `tu5n`

**Dataset Setup:**
```
datasets/
└── CE-CCTSDB/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── labels/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

## ⚙️ Installation
```bash
# Clone repository
git clone https://github.com/your-username/CTD-YOLO_traffic_sign_detection.git
cd CTD-YOLO_traffic_sign_detection

# Install dependencies
pip install -r requirements.txt
```

## 🏃 Usage

### Training
```bash
# Basic training
python train.py --data data/ce-cctsdb.yaml --img 640 --batch 16 --epochs 100

# Training with MAA
python train.py --data data/ce-cctsdb.yaml --img 640 --batch 16 --epochs 100 --maa

# Custom hyperparameters
python train.py --data data/ce-cctsdb.yaml --hyp data/hyps/hyp.scratch-low.yaml --img 640 --batch 16 --epochs 100 --maa
```

### Validation
```bash
python val.py --data data/ce-cctsdb.yaml --weights runs/train/exp/weights/best.pt --img 640
```

### Inference
```bash
# Single image
python detect.py --weights runs/train/exp/weights/best.pt --source path/to/image.jpg

# Directory of images
python detect.py --weights runs/train/exp/weights/best.pt --source path/to/images/

# Video
python detect.py --weights runs/train/exp/weights/best.pt --source path/to/video.mp4

# Webcam
python detect.py --weights runs/train/exp/weights/best.pt --source 0
```

## 🔧 Model Architecture

### Core Components
- **Backbone**: YOLOv8 with E-DCN integration
- **Neck**: Enhanced feature pyramid with multi-scale fusion
- **Head**: Detection head with WIOU loss
- **Training**: MAA-stabilized optimization

### Key Parameters
- **Input Size**: 640×640 (configurable)
- **Classes**: 58 (CE-CCTSDB traffic signs)
- **Anchors**: 3 scales × 3 aspect ratios
- **Loss**: WIOU + Classification + Objectness

## 📊 Performance Features
- **WIOU Loss**: Improved bounding box regression with distance weighting
- **MAA Mechanism**: Exponential moving average for model stability (τ=2000, decay=0.9999)
- **E-DCN**: Deformable convolution with 4 groups for adaptive feature extraction
- **Multi-scale Training**: Dynamic image scaling (0.5-1.5×)

## 🛠️ Configuration
Key configuration files:
- `models/ctd_yolo_config.yaml`: Model architecture
- `data/hyps/hyp.scratch-low.yaml`: Training hyperparameters
- `data/ce-cctsdb.yaml`: Dataset configuration

## 📈 Training Tips
1. Use `--maa` flag for better model stability
2. Adjust batch size based on GPU memory
3. Monitor validation mAP for early stopping
4. Use multi-scale training for better generalization

## 🔍 Troubleshooting
- Ensure dataset path is correct in `data/ce-cctsdb.yaml`
- Check CUDA compatibility for GPU training
- Verify all dependencies are installed correctly
- Use smaller batch size if encountering OOM errors

---
For questions or issues, please open an issue in this repository.
