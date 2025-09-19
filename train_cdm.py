"""
CDM (Classification Denoising Module) Training Script
Used to train the challenge classifier to identify degradation types in images
"""
import argparse
import os
import sys
import time
import yaml
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import cv2
import numpy as np

# Add project root to path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.cdm import CDM, create_cdm_loss
from utils import colorstr, increment_path


class ChallengeDataset(Dataset):
    """
    Challenge classification dataset
    Used to train CDM's Challenge Classifier
    """
    
    def __init__(self, data_dir, split='train', transform=None, img_size=224):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.img_size = img_size
        
        # Class mapping
        self.class_names = ['rain', 'snow', 'fog', 'blur', 'normal']
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        # Load data
        self.samples = self._load_samples()
        
    def _load_samples(self):
        """Load data samples"""
        samples = []
        split_dir = self.data_dir / self.split
        
        for class_name in self.class_names:
            class_dir = split_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob('*.jpg'):
                    samples.append((str(img_path), self.class_to_idx[class_name]))
                for img_path in class_dir.glob('*.png'):
                    samples.append((str(img_path), self.class_to_idx[class_name]))
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image = cv2.resize(image, (self.img_size, self.img_size))
        
        # Convert to tensor
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


def create_data_transforms(img_size=224):
    """Create data augmentation transforms"""
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def train_cdm_classifier(opt):
    """Train CDM classifier"""
    save_dir = Path(opt.save_dir)
    device = torch.device(opt.device)
    
    # Create save directory
    w = save_dir / 'weights'
    w.mkdir(parents=True, exist_ok=True)
    
    # Data transforms
    train_transform, val_transform = create_data_transforms(opt.img_size)
    
    # Datasets
    train_dataset = ChallengeDataset(opt.data_dir, 'train', train_transform, opt.img_size)
    val_dataset = ChallengeDataset(opt.data_dir, 'val', val_transform, opt.img_size)
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=opt.batch_size, 
        shuffle=True, 
        num_workers=opt.workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=opt.batch_size, 
        shuffle=False, 
        num_workers=opt.workers,
        pin_memory=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Model
    cdm = CDM(classifier_input_size=opt.img_size, enable_training=True).to(device)
    
    # Loss function
    criterion = create_cdm_loss()
    
    # Optimizer - use Adam with learning rate 0.001
    optimizer = optim.Adam(
        cdm.challenge_classifier.parameters(), 
        lr=opt.lr, 
        betas=(0.9, 0.999),
        weight_decay=opt.weight_decay
    )
    
    # Learning rate scheduler - ReduceLROnPlateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',  # Monitor validation accuracy
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=True
    )
    
    # Training loop
    best_acc = 0.0
    start_time = time.time()
    
    print(f"Starting CDM classifier training for {opt.epochs} epochs...")
    
    for epoch in range(opt.epochs):
        # Training phase
        cdm.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{opt.epochs}')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            logits = cdm.challenge_classifier(images)
            loss = criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        # Validation phase
        cdm.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                logits = cdm.challenge_classifier(images)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate average metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        print(f'Epoch [{epoch+1}/{opt.epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': cdm.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'class_names': train_dataset.class_names
            }, w / 'best_cdm.pt')
            print(f'New best model saved with accuracy: {best_acc:.2f}%')
        
        # Save latest model
        torch.save({
            'epoch': epoch,
            'model_state_dict': cdm.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
            'class_names': train_dataset.class_names
        }, w / 'last_cdm.pt')
        
        print('-' * 50)
    
    # Training completed
    total_time = time.time() - start_time
    print(f'Training completed in {total_time/3600:.2f} hours')
    print(f'Best validation accuracy: {best_acc:.2f}%')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data/challenge_classification', 
                       help='challenge classification dataset directory')
    parser.add_argument('--img-size', type=int, default=224, help='input image size')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--device', default='cuda:0', help='cuda device')
    parser.add_argument('--workers', type=int, default=4, help='dataloader workers')
    parser.add_argument('--project', default='runs/train_cdm', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok')
    
    return parser.parse_args()


def main(opt):
    # 创建保存目录
    opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    
    # 训练CDM分类器
    train_cdm_classifier(opt)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
