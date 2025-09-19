"""
Loss functions for CTD-YOLO including WIOU loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class WIOULoss(nn.Module):
    """
    Wise IoU Loss implementation with v3 support
    Paper: https://arxiv.org/abs/2301.10051
    """
    def __init__(self, version=3, monotonous=False, inner_iou=False, focaler_gamma=0.0, alpha=1.9, delta=3):
        super().__init__()
        self.version = version
        self.monotonous = monotonous
        self.inner_iou = inner_iou
        self.focaler_gamma = focaler_gamma
        self.alpha = alpha  # WIoU v3 alpha parameter
        self.delta = delta  # WIoU v3 delta parameter

    def forward(self, pred, target, anchor_points=None, mask_gt=None):
        """
        Args:
            pred: predicted boxes (N, 4) in xyxy format
            target: target boxes (N, 4) in xyxy format
            anchor_points: anchor points (N, 2)
            mask_gt: mask for valid targets (N,)
        """
        if mask_gt is not None:
            pred = pred[mask_gt]
            target = target[mask_gt]
            if anchor_points is not None:
                anchor_points = anchor_points[mask_gt]

        if pred.shape[0] == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        # Calculate IoU
        iou = bbox_iou(pred, target, xywh=False, CIoU=False, DIoU=False, GIoU=False, EIoU=False)
        
        # Calculate wise IoU
        if self.monotonous:
            # Monotonous focusing mechanism
            beta = iou.detach() / (1 - iou.detach() + 1e-7)
            alpha = beta / (beta + 1e-7)
        else:
            # Non-monotonous focusing mechanism
            alpha = 1.0

        # Calculate distance-based weight
        if anchor_points is not None:
            # Calculate center distance
            pred_center = (pred[:, :2] + pred[:, 2:]) / 2
            target_center = (target[:, :2] + target[:, 2:]) / 2
            center_distance = torch.norm(pred_center - target_center, dim=1)
            
            # Normalize distance
            pred_wh = pred[:, 2:] - pred[:, :2]
            target_wh = target[:, 2:] - target[:, :2]
            diagonal = torch.norm(torch.max(pred_wh, target_wh), dim=1)
            normalized_distance = center_distance / (diagonal + 1e-7)
            
            # Distance weight
            distance_weight = torch.exp(-normalized_distance)
            alpha = alpha * distance_weight

        # Calculate WIOU loss based on version
        if self.version == 3:
            # WIoU v3 implementation with optimized parameters
            loss = self._calculate_wiou_v3(pred, target, iou, alpha)
        else:
            # Original WIoU implementation
            if self.inner_iou:
                # Use inner IoU for better convergence
                inner_iou = self._calculate_inner_iou(pred, target)
                loss = alpha * (1 - inner_iou)
            else:
                loss = alpha * (1 - iou)

        # Apply focaler if specified
        if self.focaler_gamma > 0:
            loss = loss * torch.pow(1 - iou, self.focaler_gamma)

        return loss.mean()

    def _calculate_inner_iou(self, pred, target):
        """Calculate inner IoU for better convergence"""
        # Get intersection
        lt = torch.max(pred[:, :2], target[:, :2])
        rb = torch.min(pred[:, 2:], target[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, 0] * wh[:, 1]

        # Get areas
        pred_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
        target_area = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])

        # Calculate inner IoU (intersection / min(area1, area2))
        min_area = torch.min(pred_area, target_area)
        inner_iou = inter / (min_area + 1e-7)
        
        return inner_iou

    def _calculate_wiou_v3(self, pred, target, iou, alpha):
        """
        Calculate WIoU v3 loss with optimized parameters
        """
        # Calculate aspect ratio penalty
        pred_wh = pred[:, 2:] - pred[:, :2]
        target_wh = target[:, 2:] - target[:, :2]
        
        # Aspect ratio difference
        w_ratio = pred_wh[:, 0] / (target_wh[:, 0] + 1e-7)
        h_ratio = pred_wh[:, 1] / (target_wh[:, 1] + 1e-7)
        
        # WIoU v3 penalty term
        penalty = torch.exp(-self.alpha * torch.abs(w_ratio - 1)) + \
                 torch.exp(-self.alpha * torch.abs(h_ratio - 1))
        
        # Scale factor based on IoU
        scale_factor = torch.pow(1 - iou, self.delta)
        
        # Final WIoU v3 loss
        loss = alpha * (1 - iou) * penalty * scale_factor
        
        return loss


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, EIoU=False, eps=1e-7):
    """
    Calculate Intersection over Union (IoU) of box1(1, 4) to box2(n, 4).
    
    Args:
        box1 (torch.Tensor): A tensor representing a single bounding box with shape (1, 4).
        box2 (torch.Tensor): A tensor representing n bounding boxes with shape (n, 4).
        xywh (bool, optional): If True, input boxes are in (x, y, w, h) format. If False, in (x1, y1, x2, y2) format.
        GIoU (bool, optional): If True, calculate Generalized IoU. Defaults to False.
        DIoU (bool, optional): If True, calculate Distance IoU. Defaults to False.
        CIoU (bool, optional): If True, calculate Complete IoU. Defaults to False.
        EIoU (bool, optional): If True, calculate Efficient IoU. Defaults to False.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.
        
    Returns:
        (torch.Tensor): IoU, GIoU, DIoU, or CIoU values depending on the specified flags.
    """

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU or EIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU or EIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            elif EIoU:
                rho_w2 = ((b2_x2 - b2_x1) - (b1_x2 - b1_x1)) ** 2
                rho_h2 = ((b2_y2 - b2_y1) - (b1_y2 - b1_y1)) ** 2
                cw2 = cw ** 2 + eps
                ch2 = ch ** 2 + eps
                return iou - (rho2 / c2 + rho_w2 / cw2 + rho_h2 / ch2)  # EIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU


class ComputeLoss:
    """
    Compute loss for CTD-YOLO with WIOU loss
    """
    def __init__(self, model, autobalance=False, use_wiou=True):
        self.sort_obj_iou = False
        self.use_wiou = use_wiou
        
        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=0.0)  # positive, negative BCE targets

        # Focal loss
        g = 1.5  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = model.model[-1] if hasattr(model, 'model') else model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, model.hyp, autobalance
        
        # Initialize WIOU loss
        if self.use_wiou:
            self.wiou_loss = WIOULoss(monotonous=True, inner_iou=True, focaler_gamma=1.5)
        
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = next(model.parameters()).device

    def __call__(self, p, targets):  # predictions, targets, model
        device = targets.device
        lcls = torch.zeros(1, device=device)
        lbox = torch.zeros(1, device=device)
        lobj = torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                # Prediction subset corresponding to targets
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                
                if self.use_wiou:
                    # Use WIOU loss
                    anchor_points = torch.stack([gi, gj], dim=1).float()
                    lbox += self.wiou_loss(pbox, tbox[i], anchor_points)
                else:
                    # Use original IoU loss
                    iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                    lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze().detach().clamp(0)
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= 0.05
        lobj *= 1.0
        lcls *= 0.5
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < 4.0  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps
