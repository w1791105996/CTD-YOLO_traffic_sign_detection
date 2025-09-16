# CTD-YOLO: Multi-scale YOLOv8 for Traffic Sign Detection in Complex Environments

This repository contains the official implementation of the paper:

**CTD-YOLO: Multi-scale YOLOv8 for Traffic Sign Detection in Complex Environments**

## 📖 Introduction
Traffic sign detection in complex environments (e.g., rain, snow, fog, and blur) remains a challenging task due to image degradation and multi-scale variations.  
We propose **CTD-YOLO**, an improved YOLOv8-based model designed to enhance traffic sign detection under challenging conditions. CTD-YOLO incorporates multi-scale feature fusion and robustness-enhancing modules to achieve higher detection accuracy while maintaining efficiency.

## 🚀 Features
- Based on **YOLOv8** with targeted improvements for traffic sign detection.  
- **MAA (Moving Average Aggregation)** mechanism for model stability.
- **WIOU loss** for better bounding box regression.
- **E-DCN (Enhanced Deformable Convolution Network)** for adaptive feature extraction.
- Enhanced **multi-scale feature extraction** for small and degraded signs.  
- Improved robustness in **complex weather environments**.  
- Supports training and evaluation on the **CE-CCTSDB dataset**.

## 📂 Dataset
We use the **CE-CCTSDB dataset**, an extended version of CCTSDB with additional weather-corrupted samples.

- 📥 Download link: [Baidu Pan](https://pan.baidu.com/s/1gie-eZPECoKpBxGd1vCmyQ?pwd=tu5n)  
- Password: `tu5n`  

Please download and unzip the dataset, then place it under the `datasets/` directory.

Directory structure example:

datasets/
 └── CE-CCTSDB/ 
 ├── images/ 
 ├── labels/ 
 ├── train.txt 
 ├── val.txt 
 └── test.txt 

## ⚙️ Installation
```bash
# Clone this repository
git clone https://github.com/your-username/CTD-YOLO.git
cd CTD-YOLO

# Install dependencies
pip install -r requirements.txt
```

## 🏃 Training & Evaluation

### Training

```
python train.py --data data/ce-cctsdb.yaml --img 640 --batch 16 --epochs 100 --weights yolov8n.pt
```

### Evaluation

```
python val.py --data data/ce-cctsdb.yaml --weights runs/train/exp/weights/best.pt
```

### Inference

```
python detect.py --weights runs/train/exp/weights/best.pt --source path/to/images
```



🔗 For questions or discussions, please contact the authors.
