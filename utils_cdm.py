"""
CDM-related utility functions
Includes data preprocessing, model evaluation and other functionalities
"""
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
import yaml
from typing import Dict, List, Tuple, Optional


def load_cdm_config(config_path: str) -> Dict:
    """Load CDM configuration file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def preprocess_image_for_cdm(image: np.ndarray, target_size: int = 224) -> torch.Tensor:
    """
    Preprocess image for CDM
    
    Args:
        image: Input image (H, W, C) in BGR format
        target_size: Target size
        
    Returns:
        Preprocessed tensor (1, C, H, W)
    """
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize image
    image = cv2.resize(image, (target_size, target_size))
    
    # Normalize to [0,1]
    image = image.astype(np.float32) / 255.0
    
    # Convert to tensor and add batch dimension
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    
    return image


def create_challenge_dataset_structure(base_dir: str):
    """
    Create directory structure for challenge classification dataset
    
    Args:
        base_dir: Dataset root directory
    """
    base_path = Path(base_dir)
    
    # Create directory structure
    splits = ['train', 'val', 'test']
    classes = ['rain', 'snow', 'fog', 'blur', 'normal']
    
    for split in splits:
        for cls in classes:
            dir_path = base_path / split / cls
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {dir_path}")
    
    # Create dataset description file
    readme_content = """# Challenge Classification Dataset

This dataset is used to train the Challenge Classifier in CDM (Classification Denoising Module).

## Directory Structure:
```
challenge_classification/
├── train/
│   ├── rain/       # Rainy images
│   ├── snow/       # Snowy images
│   ├── fog/        # Foggy images
│   ├── blur/       # Blurred images
│   └── normal/     # Normal images
├── val/
│   ├── rain/
│   ├── snow/
│   ├── fog/
│   ├── blur/
│   └── normal/
└── test/
    ├── rain/
    ├── snow/
    ├── fog/
    ├── blur/
    └── normal/
```

## Classes:
- **rain**: Rain-degraded images
- **snow**: Snow-degraded images  
- **fog**: Fog-degraded images
- **blur**: Motion-blurred images
- **normal**: Normal clear images

## Usage:
1. Place images of corresponding categories into respective directories
2. Use train_cdm.py to train the classifier
3. The trained model can be used for CDM preprocessing
"""
    
    with open(base_path / 'README.md', 'w', encoding='utf-8') as f:
        f.write(readme_content)


def evaluate_cdm_classifier(model, dataloader, device, class_names):
    """
    Evaluate CDM classifier performance
    
    Args:
        model: CDM model
        dataloader: Validation data loader
        device: Device
        class_names: List of class names
        
    Returns:
        Evaluation results dictionary
    """
    model.eval()
    correct = 0
    total = 0
    class_correct = [0] * len(class_names)
    class_total = [0] * len(class_names)
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            # Get classifier output
            if hasattr(model, 'challenge_classifier'):
                outputs = model.challenge_classifier(images)
            else:
                outputs = model(images)
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Calculate accuracy for each class
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1
    
    # Calculate overall accuracy
    overall_acc = 100.0 * correct / total
    
    # Calculate accuracy for each class
    class_acc = {}
    for i, class_name in enumerate(class_names):
        if class_total[i] > 0:
            class_acc[class_name] = 100.0 * class_correct[i] / class_total[i]
        else:
            class_acc[class_name] = 0.0
    
    return {
        'overall_accuracy': overall_acc,
        'class_accuracy': class_acc,
        'total_samples': total,
        'correct_predictions': correct
    }


def visualize_cdm_processing(original_image, processed_image, predicted_class, save_path=None):
    """
    Visualize CDM processing results
    
    Args:
        original_image: Original image
        processed_image: Processed image
        predicted_class: Predicted challenge class
        save_path: Save path
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    if isinstance(original_image, torch.Tensor):
        original_image = original_image.squeeze().permute(1, 2, 0).cpu().numpy()
    axes[0].imshow(original_image)
    axes[0].set_title(f'Original Image\nPredicted: {predicted_class}')
    axes[0].axis('off')
    
    # Processed image
    if isinstance(processed_image, torch.Tensor):
        processed_image = processed_image.squeeze().permute(1, 2, 0).cpu().numpy()
    axes[1].imshow(processed_image)
    axes[1].set_title('CDM Processed Image')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def calculate_model_complexity(model):
    """
    Calculate model complexity
    
    Args:
        model: Model
        
    Returns:
        Model complexity information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate model size (MB)
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': size_mb,
        'parameter_breakdown': {
            'classifier': sum(p.numel() for p in model.challenge_classifier.parameters()) if hasattr(model, 'challenge_classifier') else 0,
            'denoising_blocks': sum(p.numel() for p in model.denoising_blocks.parameters()) if hasattr(model, 'denoising_blocks') else 0
        }
    }


def test_cdm_inference_speed(model, input_size=(1, 3, 640, 640), device='cuda', num_runs=100):
    """
    Test CDM inference speed
    
    Args:
        model: CDM model
        input_size: Input size
        device: Device
        num_runs: Number of test runs
        
    Returns:
        Inference time statistics
    """
    import time
    
    model.eval()
    model = model.to(device)
    
    # Warmup
    dummy_input = torch.randn(input_size).to(device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Test inference time
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            _ = model(dummy_input)
            if device == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to milliseconds
    
    return {
        'mean_time_ms': np.mean(times),
        'std_time_ms': np.std(times),
        'min_time_ms': np.min(times),
        'max_time_ms': np.max(times),
        'fps': 1000.0 / np.mean(times)
    }


def create_cdm_demo_script():
    """Create CDM demo script"""
    demo_script = '''
"""
CDM Demo Script
Demonstrate CDM processing effects under different challenge conditions
"""
import torch
import cv2
import numpy as np
from pathlib import Path
from models.cdm import CDM
from utils_cdm import preprocess_image_for_cdm, visualize_cdm_processing

def demo_cdm_processing(image_path, cdm_weights_path=None):
    """
    Demonstrate CDM processing effects
    
    Args:
        image_path: Input image path
        cdm_weights_path: CDM weights path
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load CDM model
    cdm = CDM().to(device)
    
    if cdm_weights_path and Path(cdm_weights_path).exists():
        checkpoint = torch.load(cdm_weights_path, map_location=device)
        cdm.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded CDM weights from {cdm_weights_path}")
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Failed to load image: {image_path}")
        return
    
    # Preprocessing
    input_tensor = preprocess_image_for_cdm(image).to(device)
    
    # CDM processing
    cdm.eval()
    with torch.no_grad():
        processed_tensor, classification = cdm(input_tensor, return_classification=True)
        predicted_class_idx = torch.argmax(classification, dim=1).item()
        predicted_class = cdm.class_names[predicted_class_idx]
    
    print(f"Predicted challenge type: {predicted_class}")
    
    # Visualize results
    save_path = Path(image_path).parent / f"cdm_result_{Path(image_path).stem}.png"
    visualize_cdm_processing(
        input_tensor, 
        processed_tensor, 
        predicted_class, 
        save_path=save_path
    )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Input image path')
    parser.add_argument('--weights', type=str, default='', help='CDM weights path')
    
    args = parser.parse_args()
    demo_cdm_processing(args.image, args.weights)
'''
    
    return demo_script


if __name__ == "__main__":
    # Test utility functions
    print("CDM Utils Test")
    
    # Create dataset directory structure
    create_challenge_dataset_structure("data/challenge_classification")
    
    # Create demo script
    demo_script = create_cdm_demo_script()
    with open("demo_cdm.py", "w", encoding="utf-8") as f:
        f.write(demo_script)
    
    print("CDM utilities setup completed!")
