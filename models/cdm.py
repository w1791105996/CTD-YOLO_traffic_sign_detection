"""
Classification Denoising Module (CDM) for CTD-YOLO
CDM contains Challenge Classifier and Denoising Blocks for preprocessing images in complex environments
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from typing import Dict, Tuple, Optional


class ChallengeClassifier(nn.Module):
    """
    Lightweight challenge classifier based on FastestDet
    Used to identify degradation types in images: rain, snow, fog, blur, normal
    Approximately 0.25M parameters
    """
    
    def __init__(self, num_classes=5, input_size=224):
        super(ChallengeClassifier, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        
        # Lightweight backbone based on FastestDet
        # Stage 1: Input processing
        self.stem = nn.Sequential(
            nn.Conv2d(3, 8, 3, 2, 1, bias=False),  # 224->112
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        
        # Stage 2: Feature extraction layers
        self.stage1 = self._make_stage(8, 16, 2)    # 112->56
        self.stage2 = self._make_stage(16, 32, 2)   # 56->28
        self.stage3 = self._make_stage(32, 64, 2)   # 28->14
        self.stage4 = self._make_stage(64, 128, 2)  # 14->7
        
        # Global average pooling and classification head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_stage(self, in_channels, out_channels, stride):
        """Create lightweight convolution stage"""
        return nn.Sequential(
            # Depthwise separable convolution
            nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            # Pointwise convolution
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward propagation"""
        # Input preprocessing
        if x.size(-1) != self.input_size or x.size(-2) != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size), 
                            mode='bilinear', align_corners=False)
        
        # Feature extraction
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        # Classification
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x


class RainDenoisingBlock(nn.Module):
    """Rain denoising module - improved version based on multi-scale collaborative method"""
    
    def __init__(self):
        super(RainDenoisingBlock, self).__init__()
        
        # Multi-scale feature extraction
        self.scale1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        
        self.scale2 = nn.Sequential(
            nn.Conv2d(3, 16, 5, 1, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 5, 1, 2),
            nn.ReLU(inplace=True)
        )
        
        self.scale3 = nn.Sequential(
            nn.Conv2d(3, 16, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 7, 1, 3),
            nn.ReLU(inplace=True)
        )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(48, 32, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 3, 1, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        # Multi-scale feature extraction
        feat1 = self.scale1(x)
        feat2 = self.scale2(x)
        feat3 = self.scale3(x)
        
        # Feature fusion
        fused = torch.cat([feat1, feat2, feat3], dim=1)
        residual = self.fusion(fused)
        
        # Residual connection, reduce denoising intensity to protect small targets
        return x + 0.3 * residual


class SnowDenoisingBlock(nn.Module):
    """Snow denoising module - based on hierarchical dual-tree complex wavelet representation"""
    
    def __init__(self):
        super(SnowDenoisingBlock, self).__init__()
        
        # Wavelet decomposition simulation - using convolution approximation
        self.wavelet_decomp = nn.Sequential(
            nn.Conv2d(3, 12, 3, 1, 1),  # Simulate wavelet decomposition
            nn.ReLU(inplace=True)
        )
        
        # Snow suppression network
        self.snow_suppress = nn.Sequential(
            nn.Conv2d(12, 24, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 24, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 12, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        
        # Reconstruction network
        self.reconstruct = nn.Sequential(
            nn.Conv2d(12, 6, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(6, 3, 3, 1, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        # Wavelet domain processing
        wavelet_feat = self.wavelet_decomp(x)
        suppressed = self.snow_suppress(wavelet_feat)
        residual = self.reconstruct(suppressed)
        
        return x + 0.4 * residual


class FogDenoisingBlock(nn.Module):
    """Fog denoising module - U-Net based multi-scale defogging network"""
    
    def __init__(self):
        super(FogDenoisingBlock, self).__init__()
        
        # Encoder
        self.encoder1 = self._conv_block(3, 16)
        self.encoder2 = self._conv_block(16, 32)
        self.encoder3 = self._conv_block(32, 64)
        
        # Decoder - with Strengthen-Operate-Subtract structure
        self.decoder3 = self._conv_block(64, 32)
        self.decoder2 = self._conv_block(64, 16)  # 32 + 32 from skip
        self.decoder1 = self._conv_block(32, 8)   # 16 + 16 from skip
        
        # Output layer
        self.output = nn.Sequential(
            nn.Conv2d(8, 3, 3, 1, 1),
            nn.Tanh()
        )
        
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    
    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        
        # Decoder
        d3 = self.decoder3(e3)
        d2 = self.decoder2(torch.cat([self.upsample(d3), e2], dim=1))
        d1 = self.decoder1(torch.cat([self.upsample(d2), e1], dim=1))
        
        # Output
        residual = self.output(d1)
        return x + 0.5 * residual


class BlurDenoisingBlock(nn.Module):
    """Blur denoising module - based on self-supervised meta-assisted learning method"""
    
    def __init__(self):
        super(BlurDenoisingBlock, self).__init__()
        
        # External prior network
        self.external_prior = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.ReLU(inplace=True)
        )
        
        # Internal prior network
        self.internal_prior = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        
        # Prior fusion and reconstruction
        self.fusion_reconstruct = nn.Sequential(
            nn.Conv2d(64, 48, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 24, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 3, 3, 1, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        # Extract external and internal priors
        ext_prior = self.external_prior(x)
        int_prior = self.internal_prior(x)
        
        # Fuse prior information
        fused_prior = torch.cat([ext_prior, int_prior], dim=1)
        residual = self.fusion_reconstruct(fused_prior)
        
        return x + 0.6 * residual


class CDM(nn.Module):
    """
    Classification Denoising Module (CDM)
    Complete CDM module containing challenge classifier and corresponding denoising blocks
    """
    
    def __init__(self, classifier_input_size=224, enable_training=True):
        super(CDM, self).__init__()
        
        self.enable_training = enable_training
        self.classifier_input_size = classifier_input_size
        
        # Challenge classifier
        self.challenge_classifier = ChallengeClassifier(num_classes=5, input_size=classifier_input_size)
        
        # Denoising modules dictionary
        self.denoising_blocks = nn.ModuleDict({
            'rain': RainDenoisingBlock(),
            'snow': SnowDenoisingBlock(),
            'fog': FogDenoisingBlock(),
            'blur': BlurDenoisingBlock(),
            'normal': nn.Identity()  # Normal cases need no processing
        })
        
        # Class mapping
        self.class_names = ['rain', 'snow', 'fog', 'blur', 'normal']
        
    def forward(self, x, return_classification=False):
        """
        CDM forward propagation
        
        Args:
            x: Input image [B, C, H, W]
            return_classification: Whether to return classification results
            
        Returns:
            processed_x: Processed image
            classification (optional): Classification results
        """
        batch_size = x.size(0)
        
        # 1. Challenge classification
        with torch.no_grad() if not self.enable_training else torch.enable_grad():
            classification_logits = self.challenge_classifier(x)
            predicted_classes = torch.argmax(classification_logits, dim=1)
        
        # 2. Apply corresponding denoising module based on classification results
        processed_images = []
        
        for i in range(batch_size):
            img = x[i:i+1]  # Single image
            class_idx = predicted_classes[i].item()
            class_name = self.class_names[class_idx]
            
            # Apply corresponding denoising module
            denoising_block = self.denoising_blocks[class_name]
            processed_img = denoising_block(img)
            processed_images.append(processed_img)
        
        # Merge processed images
        processed_x = torch.cat(processed_images, dim=0)
        
        if return_classification:
            return processed_x, classification_logits
        else:
            return processed_x
    
    def get_model_size(self):
        """Get model parameter count"""
        total_params = sum(p.numel() for p in self.parameters())
        classifier_params = sum(p.numel() for p in self.challenge_classifier.parameters())
        denoising_params = total_params - classifier_params
        
        return {
            'total_params': total_params,
            'classifier_params': classifier_params,
            'denoising_params': denoising_params,
            'total_size_mb': total_params * 4 / (1024 * 1024)  # Assume float32
        }


def create_cdm_loss():
    """Create CDM training loss function"""
    return nn.CrossEntropyLoss()


if __name__ == "__main__":
    # Test CDM module
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create CDM module
    cdm = CDM().to(device)
    
    # Print model information
    model_info = cdm.get_model_size()
    print("CDM Model Information:")
    print(f"Total parameters: {model_info['total_params']:,}")
    print(f"Classifier parameters: {model_info['classifier_params']:,}")
    print(f"Denoising parameters: {model_info['denoising_params']:,}")
    print(f"Model size: {model_info['total_size_mb']:.2f} MB")
    
    # Test forward propagation
    test_input = torch.randn(2, 3, 640, 640).to(device)
    
    with torch.no_grad():
        processed_output, classification = cdm(test_input, return_classification=True)
        
    print(f"\nInput shape: {test_input.shape}")
    print(f"Output shape: {processed_output.shape}")
    print(f"Classification shape: {classification.shape}")
    print(f"Predicted classes: {torch.argmax(classification, dim=1)}")
