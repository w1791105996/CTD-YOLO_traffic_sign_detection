"""
EMA (Exponential Moving Average) implementation for CTD-YOLO
"""
import math
import torch
import torch.nn as nn
from copy import deepcopy


class ModelEMA:
    """
    Exponential Moving Average of model weights
    """
    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # Create EMA
        self.ema = deepcopy(model.module if hasattr(model, 'module') else model).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)


def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)


class EMAHook:
    """
    EMA Hook for integration with training loop
    """
    def __init__(self, model, decay=0.9999, tau=2000):
        self.model = model
        self.ema = ModelEMA(model, decay=decay, tau=tau)
        
    def after_train_step(self, runner):
        """Called after each training step"""
        self.ema.update(self.model)
        
    def before_val_epoch(self, runner):
        """Called before validation epoch"""
        # Store current model state
        self._backup_and_load_ema()
        
    def after_val_epoch(self, runner):
        """Called after validation epoch"""
        # Restore original model state
        self._restore_model()
        
    def _backup_and_load_ema(self):
        """Backup current model and load EMA weights"""
        self._backup = deepcopy(self.model.state_dict())
        self.model.load_state_dict(self.ema.ema.state_dict())
        
    def _restore_model(self):
        """Restore original model weights"""
        self.model.load_state_dict(self._backup)
        del self._backup


class EMAWrapper(nn.Module):
    """
    Simple EMA wrapper that can be used directly in model definition
    """
    def __init__(self, model, decay=0.9999, device=None):
        super().__init__()
        self.model = model
        self.decay = decay
        self.device = device if device is not None else next(model.parameters()).device
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update EMA parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Apply EMA parameters to model"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
