"""
E-DCN (Enhanced Deformable Convolution Network) implementation for CTD-YOLO
Based on InternImage: Exploring Large-Scale Vision Foundation Models with Deformable Convolutions
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_


class EDCN(nn.Module):
    """
    Enhanced Deformable Convolution Network module
    """
    def __init__(self, channels, kernel_size=3, stride=1, pad=1, dilation=1, group=4, offset_scale=1.0, 
                 act_layer='GELU', norm_layer='LN', dw_kernel_size=None, center_feature_scale=False):
        super().__init__()
        if channels % group != 0:
            raise ValueError(f'channels must be divisible by group, but got {channels} and {group}')
        
        _d_per_group = channels // group
        dw_kernel_size = dw_kernel_size if dw_kernel_size is not None else kernel_size
        
        # you'd better set _d_per_group to a power of 2 which is more efficient in our CUDA implementation
        if not _d_per_group % 8 == 0:
            print(f"You'd better set channels in E-DCN to make the dimension of each attention head a multiple of 8, "
                  f"but got {channels} and {group} groups, which means {_d_per_group} channels per group.")
        
        self.offset_scale = offset_scale
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.group = group
        self.group_channels = channels // group
        self.offset_scale = offset_scale
        self.center_feature_scale = center_feature_scale

        self.dw_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=dw_kernel_size, stride=1, 
                     padding=(dw_kernel_size - 1) // 2, groups=channels),
            build_norm_layer(norm_layer, channels),
            build_act_layer(act_layer)
        )
        
        self.offset = nn.Linear(channels, group * kernel_size * kernel_size * 2)
        self.mask = nn.Linear(channels, group * kernel_size * kernel_size)
        self.input_proj = nn.Linear(channels, channels)
        self.output_proj = nn.Linear(channels, channels)
        self._reset_parameters()
        
        if center_feature_scale:
            self.center_feature_scale_proj_weight = nn.Parameter(torch.zeros((group, channels // group)))
            self.center_feature_scale_proj_bias = nn.Parameter(torch.tensor(0.0).view((1,)).repeat(group, ))
            self.center_feature_scale_module = CenterFeatureScaleModule()

    def _reset_parameters(self):
        constant_(self.offset.weight.data, 0.)
        constant_(self.offset.bias.data, 0.)
        constant_(self.mask.weight.data, 0.)
        constant_(self.mask.bias.data, 0.)
        xavier_uniform_(self.input_proj.weight.data)
        constant_(self.input_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, input):
        """
        Args:
            input (Tensor): of shape (N, H, W, C)
        """
        N, H, W, _ = input.shape

        x = self.input_proj(input)
        x_proj = x
        
        dtype = x.dtype
        x1 = input.permute(0, 3, 1, 2)  # (N, C, H, W)
        x1 = self.dw_conv(x1)
        x1 = x1.permute(0, 2, 3, 1)  # (N, H, W, C)
        
        offset = self.offset(x1)
        mask = self.mask(x1).reshape(N, H, W, self.group, -1)
        mask = F.softmax(mask, -1).reshape(N, H, W, -1)

        x = edcn_core_pytorch(
            x_proj, offset, mask,
            self.kernel_size, self.kernel_size,
            self.stride, self.stride,
            self.pad, self.pad,
            self.dilation, self.dilation,
            self.group, self.group_channels,
            self.offset_scale)
        
        if self.center_feature_scale:
            center_feature_scale = self.center_feature_scale_module(
                x1, self.center_feature_scale_proj_weight, self.center_feature_scale_proj_bias)
            # N, H, W, groups -> N, H, W, groups, 1 -> N, H, W, groups, _d_per_group -> N, H, W, channels
            center_feature_scale = center_feature_scale[..., None].repeat(
                1, 1, 1, 1, self.channels // self.group).flatten(-2)
            x = x * (1 - center_feature_scale) + x_proj * center_feature_scale
        
        x = self.output_proj(x)
        return x


def edcn_core_pytorch(input, offset, mask, kernel_h, kernel_w, stride_h, stride_w, 
                      pad_h, pad_w, dilation_h, dilation_w, group, group_channels, offset_scale):
    # for debug and test only,
    # need to use cuda version instead
    input = F.pad(input, [0, 0, pad_h, pad_h, pad_w, pad_w])
    N_, H_in, W_in, _ = input.shape
    _, H_out, W_out, _ = offset.shape

    ref = _get_reference_points(input.shape, input.device, kernel_h, kernel_w, dilation_h, dilation_w, pad_h, pad_w, stride_h, stride_w)
    grid = _generate_dilation_grids(input.shape, kernel_h, kernel_w, dilation_h, dilation_w, group, input.device)
    spatial_norm = _get_spatial_norm(grid, H_in, W_in)

    offset = offset * offset_scale
    pos = offset + ref
    
    x_sampled = F.grid_sample(
        input=input.reshape(N_, H_in, W_in, group, group_channels).permute(0, 3, 4, 1, 2).flatten(0, 1),
        grid=pos[..., :2].reshape(N_ * group, H_out, W_out, kernel_h * kernel_w, 2),
        mode='bilinear', padding_mode='zeros', align_corners=False)
    
    x_sampled = x_sampled.view(N_, group, group_channels, H_out, W_out, kernel_h * kernel_w)
    x_sampled = x_sampled.permute(0, 3, 4, 1, 5, 2).flatten(3, 4)  # N_, H_out, W_out, group * kernel_h * kernel_w, group_channels
    
    out = (x_sampled * mask.unsqueeze(-1)).sum(dim=3)
    return out


def _get_reference_points(spatial_shapes, device, kernel_h, kernel_w, dilation_h, dilation_w, pad_h, pad_w, stride_h, stride_w):
    _, H_, W_, _ = spatial_shapes
    H_out = (H_ - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
    W_out = (W_ - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1

    ref_y, ref_x = torch.meshgrid(
        torch.linspace(0.5, H_out - 0.5, H_out, dtype=torch.float32, device=device) * stride_h,
        torch.linspace(0.5, W_out - 0.5, W_out, dtype=torch.float32, device=device) * stride_w,
        indexing='ij')
    ref = torch.stack((ref_x, ref_y), -1)
    ref[..., 0] = ref[..., 0] / W_
    ref[..., 1] = ref[..., 1] / H_
    ref = ref * 2.0 - 1.0
    ref = ref[None, ...].expand(spatial_shapes[0], -1, -1, -1)  # N, H_out, W_out, 2
    
    return ref


def _generate_dilation_grids(spatial_shapes, kernel_h, kernel_w, dilation_h, dilation_w, group, device):
    _, H_, W_, _ = spatial_shapes
    points_list = []
    x, y = torch.meshgrid(
        torch.linspace(-((dilation_w * (kernel_w - 1)) // 2), -((dilation_w * (kernel_w - 1)) // 2) + (kernel_w - 1) * dilation_w, kernel_w, dtype=torch.float32, device=device),
        torch.linspace(-((dilation_h * (kernel_h - 1)) // 2), -((dilation_h * (kernel_h - 1)) // 2) + (kernel_h - 1) * dilation_h, kernel_h, dtype=torch.float32, device=device),
        indexing='ij')
    points_list.extend([x / W_, y / H_])
    grid = torch.stack(points_list, -1).reshape(-1, 1, 2).repeat(1, group, 1)  # kernel_h * kernel_w, group, 2
    grid = grid * 2.0
    
    return grid


def _get_spatial_norm(grid, H, W):
    # grid: kernel_h * kernel_w, group, 2
    spatial_norm = torch.tensor([W, H]).reshape(1, 1, 2).repeat(grid.size(0), grid.size(1), 1).to(grid.device)
    return spatial_norm


class CenterFeatureScaleModule(nn.Module):
    def forward(self, query, center_feature_scale_proj_weight, center_feature_scale_proj_bias):
        center_feature_scale = F.linear(query, weight=center_feature_scale_proj_weight, bias=center_feature_scale_proj_bias).sigmoid()
        return center_feature_scale


def build_norm_layer(norm_layer, num_features, eps=1e-6):
    if norm_layer == 'BN':
        return nn.BatchNorm2d(num_features, eps=eps)
    elif norm_layer == 'LN':
        return nn.GroupNorm(1, num_features, eps=eps)
    elif norm_layer == 'GN':
        return nn.GroupNorm(32, num_features, eps=eps)
    else:
        raise NotImplementedError(f'build_norm_layer does not support {norm_layer}')


def build_act_layer(act_layer):
    if act_layer == 'ReLU':
        return nn.ReLU(inplace=True)
    elif act_layer == 'SiLU':
        return nn.SiLU(inplace=True)
    elif act_layer == 'GELU':
        return nn.GELU()
    else:
        raise NotImplementedError(f'build_act_layer does not support {act_layer}')


class EDCN_Conv(nn.Module):
    """E-DCN convolution wrapper for easy integration"""
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        if p is None:
            p = k // 2
        
        # Use regular conv if input/output channels don't match E-DCN requirements
        if c1 != c2 or c1 % 8 != 0:
            self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
            self.bn = nn.BatchNorm2d(c2)
            self.act = nn.SiLU() if act else nn.Identity()
            self.use_edcn = False
        else:
            self.edcn = EDCN(c1, kernel_size=k, stride=s, pad=p)
            self.conv_proj = nn.Conv2d(c1, c2, 1) if c1 != c2 else nn.Identity()
            self.bn = nn.BatchNorm2d(c2)
            self.act = nn.SiLU() if act else nn.Identity()
            self.use_edcn = True

    def forward(self, x):
        if not self.use_edcn:
            return self.act(self.bn(self.conv(x)))
        
        # E-DCN expects (N, H, W, C) format
        B, C, H, W = x.shape
        x_hwc = x.permute(0, 2, 3, 1)  # (N, H, W, C)
        
        # Apply E-DCN
        out = self.edcn(x_hwc)
        
        # Convert back to (N, C, H, W)
        out = out.permute(0, 3, 1, 2)
        out = self.conv_proj(out)
        
        return self.act(self.bn(out))
