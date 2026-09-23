# Copyright (c) OpenMMLab. All rights reserved.
"""MORI-seg: Object-Aware RTMDet-Ins head.

Reference: "Instance Segmentation of Biomedical Images with an
Object-aware Embedding Learned with Local Constraints" (arXiv:2004.09821)

Training-only auxiliary supervision on the mask feature:
  - embedding consistency + local discriminative term
  - distance-to-boundary regression
  - boundary band classification

The detection and segmentation inference path is identical to the stock
RTMDetInsSepBNHead; the auxiliary branches are used by the loss only.
Registered as ``MORIObjectAwareRTMDetInsSepBNHead``; requires stock mmdet 3.3.0.
"""

import math
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.ops import batched_nms
from torch import Tensor

from mmdet.models.dense_heads.rtmdet_ins_head import RTMDetInsSepBNHead
from mmdet.registry import MODELS
from mmdet.structures.bbox import get_box_tensor, get_box_wh, scale_boxes
from mmdet.utils import InstanceList, OptInstanceList


@MODELS.register_module()
class MORIObjectAwareRTMDetInsSepBNHead(RTMDetInsSepBNHead):
    """RTMDet-Ins SepBN head + Object-Aware auxiliary branches.

    The detection and instance mask branches behave exactly as in
    RTMDetInsSepBNHead; the auxiliary branches are training-only and do not
    change the inference output interface.

    Auxiliary losses:
        loss_objaware = warmup * (lambda_reg * L_reg
                                  + lambda_emb * (L_con + lambda_dis * L_dis))
        loss_objaware_boundary = boundary_warmup * boundary_loss_weight * L_bnd

        where
        - L_reg: distance regression (pixel to nearest boundary, normalized)
        - L_con: same-instance embedding consistency (cosine distance)
        - L_dis: local discriminative loss on neighboring instances
        - L_bnd: boundary band BCE / focal BCE

    Args:
        objaware_embedding_dim (int): Embedding dimension D. Default 16.
        objaware_hidden_channels (int): Hidden channels of the auxiliary branches. Default 32.
        lambda_objaware_reg (float): Weight of L_reg. Default 1.0.
        lambda_objaware_emb (float): Overall weight of the embedding terms. Default 5.0.
        lambda_objaware_dis (float): Coefficient of L_dis inside the embedding term. Default 1.0.
        neighbor_distance (float): Neighbourhood threshold d.
        neighbor_distance_is_input (bool): If True, d is given in input-image
            pixels and is rescaled to the mask-feature stride automatically.
        objaware_local_constraint (bool): If True, apply L_dis to neighbouring
            instances only; otherwise apply it globally.
        objaware_same_class_only (bool): If True, apply L_dis only between
            instances of the same class.
        objaware_include_background (bool): Whether background takes part in
            the embedding terms as an extra "object".
        objaware_min_instance_pixels (int): Minimum instance size, in pixels,
            after downsampling to the mask-feature resolution.
        objaware_balance_reg_fg_bg (bool): Whether L_reg balances foreground
            and background frequency.
        objaware_bg_reg_weight (float): Background pixel weight when balancing
            is disabled.
        objaware_dist_target_transform (str): Distance target transform,
            'power' or 'exp'.
        objaware_dist_target_power (float): Exponent a when transform='power'.
        objaware_dist_target_exp_alpha (float): Alpha when transform='exp',
            mapping x to (exp(alpha*x)-1)/(exp(alpha)-1).
        objaware_warmup_iters (int): Linear warmup iterations for the
            distance/embedding terms; 0 disables warmup.
        objaware_boundary_hidden_channels (int): Hidden channels of the boundary branch.
        objaware_boundary_loss_weight (float): Boundary loss weight; 0 disables the branch.
        objaware_boundary_width (int): Boundary band width.
        objaware_boundary_width_is_input (bool): If True, the band width is
            given in input-image pixels.
        objaware_boundary_pos_weight (float): Positive-class weight of the boundary BCE.
        objaware_boundary_warmup_iters (int): Warmup iterations of the boundary branch.
        objaware_boundary_loss_type (str): Boundary loss type, 'bce' or 'focal'.
        objaware_boundary_focal_gamma (float): Boundary focal gamma.
        objaware_boundary_focal_alpha (float): Boundary focal alpha.
        objaware_boundary_suppress_infer (bool): Whether to apply boundary
            suppression at inference time.
        objaware_boundary_suppress_gamma (float): Strength of the boundary
            suppression at inference time.
        objaware_export_map (bool): Whether to export the objaware maps at inference time.
    """

    def __init__(self,
                 *args,
                 objaware_embedding_dim: int = 16,
                 objaware_hidden_channels: int = 32,
                 lambda_objaware_reg: float = 1.0,
                 lambda_objaware_emb: float = 5.0,
                 lambda_objaware_dis: float = 1.0,
                 neighbor_distance: float = 10.0,
                 neighbor_distance_is_input: bool = True,
                 objaware_local_constraint: bool = True,
                 objaware_same_class_only: bool = False,
                 objaware_include_background: bool = False,
                 objaware_min_instance_pixels: int = 4,
                 objaware_balance_reg_fg_bg: bool = True,
                 objaware_bg_reg_weight: float = 1.0,
                 objaware_dist_target_transform: str = 'power',
                 objaware_dist_target_power: float = 1.0,
                 objaware_dist_target_exp_alpha: float = 3.0,
                 objaware_warmup_iters: int = 0,
                 objaware_boundary_hidden_channels: int = 32,
                 objaware_boundary_loss_weight: float = 0.0,
                 objaware_boundary_width: int = 2,
                 objaware_boundary_width_is_input: bool = False,
                 objaware_boundary_pos_weight: float = 1.0,
                 objaware_boundary_warmup_iters: int = 0,
                 objaware_boundary_loss_type: str = 'bce',
                 objaware_boundary_focal_gamma: float = 2.0,
                 objaware_boundary_focal_alpha: float = 0.25,
                 objaware_boundary_suppress_infer: bool = False,
                 objaware_boundary_suppress_gamma: float = 0.5,
                 objaware_export_map: bool = False,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.objaware_embedding_dim = objaware_embedding_dim
        self.objaware_hidden_channels = objaware_hidden_channels
        self.lambda_objaware_reg = lambda_objaware_reg
        self.lambda_objaware_emb = lambda_objaware_emb
        self.lambda_objaware_dis = lambda_objaware_dis
        self.neighbor_distance = neighbor_distance
        self.neighbor_distance_is_input = neighbor_distance_is_input
        self.objaware_local_constraint = objaware_local_constraint
        self.objaware_same_class_only = objaware_same_class_only
        self.objaware_include_background = objaware_include_background
        self.objaware_min_instance_pixels = objaware_min_instance_pixels
        self.objaware_balance_reg_fg_bg = objaware_balance_reg_fg_bg
        self.objaware_bg_reg_weight = objaware_bg_reg_weight
        self.objaware_dist_target_transform = str(objaware_dist_target_transform).lower()
        if self.objaware_dist_target_transform not in {'power', 'exp'}:
            raise ValueError('objaware_dist_target_transform must be "power" or "exp"')
        self.objaware_dist_target_power = float(objaware_dist_target_power)
        if self.objaware_dist_target_power <= 0:
            raise ValueError('objaware_dist_target_power must be > 0')
        self.objaware_dist_target_exp_alpha = float(objaware_dist_target_exp_alpha)
        if self.objaware_dist_target_exp_alpha <= 0:
            raise ValueError('objaware_dist_target_exp_alpha must be > 0')
        self.objaware_warmup_iters = objaware_warmup_iters

        self.objaware_boundary_hidden_channels = int(objaware_boundary_hidden_channels)
        self.objaware_boundary_loss_weight = float(objaware_boundary_loss_weight)
        self.objaware_boundary_width = int(objaware_boundary_width)
        self.objaware_boundary_width_is_input = bool(objaware_boundary_width_is_input)
        self.objaware_boundary_pos_weight = float(objaware_boundary_pos_weight)
        self.objaware_boundary_warmup_iters = int(objaware_boundary_warmup_iters)
        self.objaware_boundary_loss_type = str(objaware_boundary_loss_type).lower()
        self.objaware_boundary_focal_gamma = float(objaware_boundary_focal_gamma)
        self.objaware_boundary_focal_alpha = float(objaware_boundary_focal_alpha)
        self.objaware_boundary_suppress_infer = bool(objaware_boundary_suppress_infer)
        self.objaware_boundary_suppress_gamma = float(objaware_boundary_suppress_gamma)

        if self.objaware_boundary_loss_type not in {'bce', 'focal'}:
            raise ValueError('objaware_boundary_loss_type must be "bce" or "focal"')

        self.objaware_export_map = bool(objaware_export_map)

        # Auxiliary prediction branch from mask feature (B, num_prototypes, H, W)
        self.objaware_proj = nn.Sequential(
            ConvModule(
                self.num_prototypes,
                self.objaware_hidden_channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            ConvModule(
                self.objaware_hidden_channels,
                self.objaware_hidden_channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
        )
        self.objaware_emb_head = nn.Conv2d(self.objaware_hidden_channels,
                                           self.objaware_embedding_dim, 1)
        self.objaware_dist_head = nn.Conv2d(self.objaware_hidden_channels, 1, 1)

        self.objaware_boundary_proj = nn.Sequential(
            ConvModule(
                self.num_prototypes,
                self.objaware_boundary_hidden_channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            ConvModule(
                self.objaware_boundary_hidden_channels,
                self.objaware_boundary_hidden_channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
        )
        self.objaware_boundary_head = nn.Conv2d(self.objaware_boundary_hidden_channels, 1, 1)

        # Iteration counters used by the warmup schedules
        self.register_buffer('_objaware_iter',
                             torch.tensor(0, dtype=torch.long),
                             persistent=True)
        self.register_buffer('_objaware_boundary_iter',
                             torch.tensor(0, dtype=torch.long),
                             persistent=True)

    def loss_by_feat(self,
                     cls_scores: List[Tensor],
                     bbox_preds: List[Tensor],
                     kernel_preds: List[Tensor],
                     mask_feat: Tensor,
                     batch_gt_instances: InstanceList,
                     batch_img_metas: List[dict],
                     batch_gt_instances_ignore: OptInstanceList = None):
        """Compute RTMDet-Ins base losses + object-aware auxiliary loss."""
        losses = super().loss_by_feat(
            cls_scores,
            bbox_preds,
            kernel_preds,
            mask_feat,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore)

        objaware = self._compute_object_aware_loss(mask_feat, batch_gt_instances)
        losses.update(objaware)
        return losses

    def _compute_object_aware_loss(self, mask_feat: Tensor,
                                   batch_gt_instances: InstanceList) -> Dict[str, Tensor]:
        """Compute object-aware auxiliary losses on mask feature map."""
        self._objaware_iter += 1
        if self.objaware_warmup_iters > 0:
            warmup = min(self._objaware_iter.item() / float(self.objaware_warmup_iters),
                         1.0)
        else:
            warmup = 1.0

        self._objaware_boundary_iter += 1
        if self.objaware_boundary_warmup_iters > 0:
            boundary_warmup = min(
                self._objaware_boundary_iter.item()
                / float(self.objaware_boundary_warmup_iters), 1.0)
        else:
            boundary_warmup = 1.0

        # Auxiliary predictions
        aux_feat = self.objaware_proj(mask_feat.float())
        emb_pred = self.objaware_emb_head(aux_feat)
        emb_pred = F.normalize(emb_pred, p=2, dim=1, eps=1e-6)
        dist_pred = F.relu(self.objaware_dist_head(aux_feat).squeeze(1))
        boundary_logits = self.objaware_boundary_head(
            self.objaware_boundary_proj(mask_feat.float())).squeeze(1)

        B, _, H, W = emb_pred.shape
        device = emb_pred.device

        reg_losses: List[Tensor] = []
        con_losses: List[Tensor] = []
        dis_losses: List[Tensor] = []
        boundary_losses: List[Tensor] = []

        # neighbor threshold on feature scale
        stride = float(self.prior_generator.strides[0][0])
        if self.neighbor_distance_is_input:
            neigh_radius = int(round(self.neighbor_distance / max(stride, 1.0)))
        else:
            neigh_radius = int(round(self.neighbor_distance))
        neigh_radius = max(1, neigh_radius)

        if self.objaware_boundary_width_is_input:
            boundary_width = int(round(self.objaware_boundary_width / max(stride, 1.0)))
        else:
            boundary_width = int(round(self.objaware_boundary_width))
        boundary_width = max(0, boundary_width)

        boundary_pos_weight = None
        if self.objaware_boundary_pos_weight != 1.0:
            boundary_pos_weight = boundary_logits.new_tensor(self.objaware_boundary_pos_weight)

        for b in range(B):
            gt_instances = batch_gt_instances[b]
            sample_zero = dist_pred[b].sum() * 0

            if not hasattr(gt_instances, 'masks'):
                reg_losses.append(sample_zero)
                con_losses.append(sample_zero)
                dis_losses.append(sample_zero)
                boundary_losses.append(sample_zero)
                continue

            gt_masks = gt_instances.masks
            if not torch.is_tensor(gt_masks):
                gt_masks = gt_masks.to_tensor(dtype=torch.bool, device=device)
            else:
                gt_masks = gt_masks.to(device=device, dtype=torch.bool)

            if gt_masks.numel() == 0:
                # no foreground: distance target is zero map
                empty_target = torch.zeros_like(dist_pred[b])
                full_bg = torch.ones_like(empty_target, dtype=torch.bool)
                reg_losses.append(
                    self._distance_regression_loss(
                        dist_pred[b], empty_target, ~full_bg, full_bg))
                con_losses.append(sample_zero)
                dis_losses.append(sample_zero)
                boundary_losses.append(sample_zero)
                continue

            if gt_masks.dim() == 2:
                gt_masks = gt_masks.unsqueeze(0)

            # Downsample GT instances to mask feature resolution
            ds_masks = F.interpolate(
                gt_masks.float().unsqueeze(1),
                size=(H, W),
                mode='nearest').squeeze(1) > 0.5

            # filter tiny instances to reduce noisy supervision
            valid = ds_masks.flatten(1).sum(dim=1) >= self.objaware_min_instance_pixels
            all_labels = gt_instances.labels.to(device=device)
            if valid.any():
                ds_masks = ds_masks[valid]
                labels = all_labels[valid]
            else:
                ds_masks = ds_masks[:0]
                labels = all_labels[:0]

            num_inst = ds_masks.shape[0]
            if num_inst == 0:
                empty_target = torch.zeros_like(dist_pred[b])
                full_bg = torch.ones_like(empty_target, dtype=torch.bool)
                reg_losses.append(
                    self._distance_regression_loss(
                        dist_pred[b], empty_target, ~full_bg, full_bg))
                con_losses.append(sample_zero)
                dis_losses.append(sample_zero)
                boundary_losses.append(sample_zero)
                continue

            # build instance id map
            inst_map = torch.full((H, W), -1, dtype=torch.long, device=device)
            for inst_id in range(num_inst):
                inst_map[ds_masks[inst_id]] = inst_id

            fg_mask = inst_map >= 0
            bg_mask = ~fg_mask

            if not fg_mask.any():
                empty_target = torch.zeros_like(dist_pred[b])
                reg_losses.append(
                    self._distance_regression_loss(
                        dist_pred[b], empty_target, fg_mask, bg_mask))
                con_losses.append(sample_zero)
                dis_losses.append(sample_zero)
                boundary_losses.append(sample_zero)
                continue

            # Keep only instances still visible in inst_map after overlap resolution
            active_ids = torch.unique(inst_map[fg_mask])
            active_ids = active_ids[active_ids >= 0]
            if active_ids.numel() == 0:
                empty_target = torch.zeros_like(dist_pred[b])
                reg_losses.append(
                    self._distance_regression_loss(
                        dist_pred[b], empty_target, fg_mask, bg_mask))
                con_losses.append(sample_zero)
                dis_losses.append(sample_zero)
                boundary_losses.append(sample_zero)
                continue

            active_masks = torch.stack([inst_map == old_id for old_id in active_ids], dim=0)
            active_labels = labels[active_ids.long()]
            num_inst = int(active_masks.shape[0])

            # remap instance ids to contiguous range [0, num_inst-1]
            inst_map_compact = torch.full_like(inst_map, -1)
            for new_id, old_id in enumerate(active_ids):
                inst_map_compact[inst_map == old_id] = new_id

            # distance target
            dist_target = torch.zeros((H, W), dtype=dist_pred.dtype, device=device)
            for inst_id in range(num_inst):
                norm_dist = self._normalized_distance_map(active_masks[inst_id])
                dist_target[active_masks[inst_id]] = norm_dist[active_masks[inst_id]]

            reg_losses.append(
                self._distance_regression_loss(
                    dist_pred[b], dist_target, fg_mask, bg_mask))

            if self.objaware_boundary_loss_weight > 0:
                boundary_target = self._build_boundary_target(active_masks, boundary_width)
                boundary_losses.append(
                    self._boundary_supervision_loss(
                        logits=boundary_logits[b],
                        target=boundary_target,
                        pos_weight=boundary_pos_weight))
            else:
                boundary_losses.append(sample_zero)

            # embedding consistency + discriminative
            emb_flat = emb_pred[b].permute(1, 2, 0).reshape(-1, self.objaware_embedding_dim)
            emb_flat = F.normalize(emb_flat, p=2, dim=1, eps=1e-6)
            inst_flat = inst_map_compact.reshape(-1)

            if self.objaware_include_background:
                # background as an additional object id = num_inst
                use_ids = inst_flat.clone()
                use_ids[use_ids < 0] = num_inst
                num_groups = num_inst + 1
                emb_used = emb_flat
            else:
                valid_pix = inst_flat >= 0
                if valid_pix.sum() == 0:
                    con_losses.append(sample_zero)
                    dis_losses.append(sample_zero)
                    continue
                use_ids = inst_flat[valid_pix]
                emb_used = emb_flat[valid_pix]
                num_groups = num_inst

            means: List[Tensor] = []
            obj_con: List[Tensor] = []
            for gid in range(num_groups):
                gid_mask = use_ids == gid
                if gid_mask.sum() == 0:
                    continue
                pixels = emb_used[gid_mask]
                mean_vec = F.normalize(
                    pixels.mean(dim=0, keepdim=True), p=2, dim=1, eps=1e-6).squeeze(0)
                means.append(mean_vec)
                cos_sim = F.cosine_similarity(pixels, mean_vec.unsqueeze(0), dim=1, eps=1e-6)
                obj_con.append((1.0 - cos_sim).mean())

            if len(obj_con) > 0:
                con_losses.append(torch.stack(obj_con).mean())
            else:
                con_losses.append(sample_zero)

            # Local discriminative term is applied on foreground instances
            if num_inst >= 2:
                fg_means = torch.stack(means[:num_inst], dim=0)
                dis_mask = self._build_discriminative_mask(
                    ds_masks=active_masks,
                    labels=active_labels,
                    radius=neigh_radius,
                    local_only=self.objaware_local_constraint,
                    same_class_only=self.objaware_same_class_only)

                dis_added = False

                if dis_mask.any():
                    # |1 - D| with D=1-cos => |cos|
                    abs_cos = torch.abs(fg_means @ fg_means.t())
                    dis_losses.append(abs_cos[dis_mask].mean())
                    dis_added = True

                # optional background-object orthogonality
                if self.objaware_include_background and len(means) == (num_inst + 1):
                    bg_mean = means[-1]
                    bg_abs_cos = torch.abs(fg_means @ bg_mean)
                    if bg_abs_cos.numel() > 0:
                        dis_losses.append(bg_abs_cos.mean())
                        dis_added = True

                if not dis_added:
                    dis_losses.append(sample_zero)
            if num_inst < 2:
                dis_losses.append(sample_zero)

        zero = dist_pred.sum() * 0
        reg_val = torch.stack(reg_losses).mean() if len(reg_losses) > 0 else zero
        con_val = torch.stack(con_losses).mean() if len(con_losses) > 0 else zero
        dis_val = torch.stack(dis_losses).mean() if len(dis_losses) > 0 else zero
        boundary_val = torch.stack(boundary_losses).mean() if len(boundary_losses) > 0 else zero

        total = self.lambda_objaware_reg * reg_val + self.lambda_objaware_emb * (
            con_val + self.lambda_objaware_dis * dis_val)
        total = total * warmup

        loss_objaware_boundary = (
            boundary_val
            * self.objaware_boundary_loss_weight
            * boundary_warmup)
        loss_objaware_boundary = torch.nan_to_num(
            loss_objaware_boundary, nan=0.0, posinf=1.0, neginf=0.0)

        return {
            'loss_objaware': total,
            'loss_objaware_boundary': loss_objaware_boundary,
            'objaware_reg_val': reg_val.detach(),
            'objaware_con_val': con_val.detach(),
            'objaware_dis_val': dis_val.detach(),
            'objaware_boundary_val': boundary_val.detach(),
            'objaware_warmup': reg_val.new_tensor(warmup),
            'objaware_boundary_warmup': reg_val.new_tensor(boundary_warmup),
        }

    def _boundary_supervision_loss(self,
                                   logits: Tensor,
                                   target: Tensor,
                                   pos_weight: Optional[Tensor] = None) -> Tensor:
        """Boundary supervision loss: BCE or focal BCE."""
        if self.objaware_boundary_loss_type == 'bce':
            return F.binary_cross_entropy_with_logits(
                logits,
                target,
                reduction='mean',
                pos_weight=pos_weight)

        # focal BCE
        bce = F.binary_cross_entropy_with_logits(
            logits,
            target,
            reduction='none',
            pos_weight=pos_weight)
        prob = torch.sigmoid(logits)
        p_t = prob * target + (1.0 - prob) * (1.0 - target)
        alpha_t = (
            self.objaware_boundary_focal_alpha * target
            + (1.0 - self.objaware_boundary_focal_alpha) * (1.0 - target)
        )
        focal_weight = alpha_t * (1.0 - p_t).pow(self.objaware_boundary_focal_gamma)
        return (focal_weight * bce).mean()

    @staticmethod
    def _upsample_to_image(feat_map: Tensor, stride: int, img_meta: Optional[dict],
                           rescale: bool) -> Tensor:
        """Upsample a (1, C, h, w) map by stride, optionally back to ori_shape."""
        feat_map = F.interpolate(
            feat_map, scale_factor=stride, mode='bilinear', align_corners=False)
        if rescale and img_meta is not None and img_meta.get('scale_factor') is not None:
            ori_h, ori_w = img_meta['ori_shape'][:2]
            scale_factor = [1 / s for s in img_meta['scale_factor']]
            feat_map = F.interpolate(
                feat_map,
                size=[
                    math.ceil(feat_map.shape[-2] * scale_factor[0]),
                    math.ceil(feat_map.shape[-1] * scale_factor[1])
                ],
                mode='bilinear',
                align_corners=False)
            feat_map = feat_map[..., :ori_h, :ori_w]
        return feat_map

    def _predict_objaware_map(self,
                              mask_feat: Tensor,
                              img_meta: Optional[dict],
                              rescale: bool = False) -> Tensor:
        """Predict distance map for inference/export. Returns (H, W)."""
        aux_feat = self.objaware_proj(mask_feat.unsqueeze(0).float())
        dist_map = F.relu(self.objaware_dist_head(aux_feat))
        stride = self.prior_generator.strides[0][0]
        return self._upsample_to_image(dist_map, stride, img_meta, rescale).squeeze(0).squeeze(0)

    def _predict_objaware_embedding_map(self,
                                        mask_feat: Tensor,
                                        img_meta: Optional[dict],
                                        rescale: bool = False) -> Tensor:
        """Predict embedding map for inference/export. Returns (D, H, W)."""
        aux_feat = self.objaware_proj(mask_feat.unsqueeze(0).float())
        emb_map = F.normalize(self.objaware_emb_head(aux_feat), p=2, dim=1, eps=1e-6)
        stride = self.prior_generator.strides[0][0]
        return self._upsample_to_image(emb_map, stride, img_meta, rescale).squeeze(0)

    def _predict_boundary_map(self,
                              mask_feat: Tensor,
                              img_meta: Optional[dict],
                              rescale: bool = False) -> Tensor:
        """Predict boundary probability map for inference/export. Returns (H, W)."""
        boundary_map = torch.sigmoid(self.objaware_boundary_head(
            self.objaware_boundary_proj(mask_feat.unsqueeze(0).float())))
        stride = self.prior_generator.strides[0][0]
        return self._upsample_to_image(boundary_map, stride, img_meta,
                                       rescale).squeeze(0).squeeze(0)

    def _bbox_mask_post_process(
            self,
            results,
            mask_feat,
            cfg,
            rescale: bool = False,
            with_nms: bool = True,
            img_meta: Optional[dict] = None):
        stride = self.prior_generator.strides[0][0]
        boundary_map: Optional[Tensor] = None

        if rescale:
            assert img_meta.get('scale_factor') is not None
            scale_factor = [1 / s for s in img_meta['scale_factor']]
            results.bboxes = scale_boxes(results.bboxes, scale_factor)

        if hasattr(results, 'score_factors'):
            score_factors = results.pop('score_factors')
            results.scores = results.scores * score_factors

        if cfg.get('min_bbox_size', -1) >= 0:
            w, h = get_box_wh(results.bboxes)
            valid_mask = (w > cfg.min_bbox_size) & (h > cfg.min_bbox_size)
            if not valid_mask.all():
                results = results[valid_mask]

        assert with_nms, 'with_nms must be True for RTMDet-Ins'
        if results.bboxes.numel() > 0:
            bboxes = get_box_tensor(results.bboxes)
            det_bboxes, keep_idxs = batched_nms(bboxes, results.scores,
                                                results.labels, cfg.nms)
            results = results[keep_idxs]
            results.scores = det_bboxes[:, -1]
            results = results[:cfg.max_per_img]

            mask_logits = self._mask_predict_by_feat_single(
                mask_feat, results.kernels, results.priors)

            mask_logits = F.interpolate(
                mask_logits.unsqueeze(0), scale_factor=stride, mode='bilinear')

            if rescale:
                ori_h, ori_w = img_meta['ori_shape'][:2]
                mask_logits = F.interpolate(
                    mask_logits,
                    size=[
                        math.ceil(mask_logits.shape[-2] * scale_factor[0]),
                        math.ceil(mask_logits.shape[-1] * scale_factor[1])
                    ],
                    mode='bilinear',
                    align_corners=False)[..., :ori_h, :ori_w]

            if self.objaware_boundary_suppress_infer and self.objaware_boundary_suppress_gamma > 0:
                if boundary_map is None:
                    boundary_map = self._predict_boundary_map(
                        mask_feat, img_meta, rescale=rescale)
                boundary_prob = boundary_map.to(mask_logits.dtype).unsqueeze(0).unsqueeze(0)
                if boundary_prob.shape[-2:] != mask_logits.shape[-2:]:
                    boundary_prob = F.interpolate(
                        boundary_prob,
                        size=mask_logits.shape[-2:],
                        mode='bilinear',
                        align_corners=False)
                mask_logits = mask_logits - self.objaware_boundary_suppress_gamma * boundary_prob

            masks = mask_logits.sigmoid().squeeze(0)
            masks = masks > cfg.mask_thr_binary
            results.masks = masks
        else:
            h, w = img_meta['ori_shape'][:2] if rescale else img_meta[
                'img_shape'][:2]
            results.masks = torch.zeros(
                size=(results.bboxes.shape[0], h, w),
                dtype=torch.bool,
                device=results.bboxes.device)

        if self.objaware_export_map:
            with torch.no_grad():
                obj_map = self._predict_objaware_map(mask_feat, img_meta, rescale=rescale)
                emb_map = self._predict_objaware_embedding_map(
                    mask_feat, img_meta, rescale=rescale)
                if boundary_map is None:
                    boundary_map = self._predict_boundary_map(mask_feat, img_meta, rescale=rescale)
            export_dict = {
                'objaware_map': obj_map.detach(),
                'objaware_map_kind': 'distance',
                'objaware_embedding_map': emb_map.detach(),
                'objaware_boundary_map': boundary_map.detach(),
            }
            if hasattr(results, 'set_metainfo'):
                results.set_metainfo(export_dict)
            else:
                results.objaware_map = export_dict['objaware_map']
                results.objaware_map_kind = export_dict['objaware_map_kind']
                results.objaware_embedding_map = export_dict['objaware_embedding_map']
                results.objaware_boundary_map = export_dict['objaware_boundary_map']

        return results

    def _distance_regression_loss(self,
                                  pred: Tensor,
                                  target: Tensor,
                                  fg_mask: Tensor,
                                  bg_mask: Tensor) -> Tensor:
        """Weighted MSE for distance regression."""
        sq = (pred - target).pow(2)
        weights = torch.ones_like(sq)

        if self.objaware_balance_reg_fg_bg:
            num_fg = int(fg_mask.sum().item())
            num_bg = int(bg_mask.sum().item())
            if num_fg > 0 and num_bg > 0:
                # foreground and background contribute equally
                weights[fg_mask] = 0.5 / num_fg
                weights[bg_mask] = 0.5 / num_bg
            elif num_fg > 0:
                weights[fg_mask] = 1.0 / num_fg
                weights[bg_mask] = 0.0
            else:
                weights[bg_mask] = 1.0 / max(num_bg, 1)
        else:
            weights[bg_mask] = self.objaware_bg_reg_weight

        return (sq * weights).sum() / (weights.sum() + 1e-6)

    def _normalized_distance_map(self, mask: Tensor) -> Tensor:
        """Distance-to-boundary map normalized to [0, 1] within one instance."""
        if mask.numel() == 0 or mask.sum() == 0:
            return mask.float()

        mask_np = mask.detach().cpu().numpy().astype('uint8')
        dist = cv2.distanceTransform(mask_np, cv2.DIST_L2, 5)
        max_dist = float(dist.max())
        if max_dist > 0:
            dist = dist / max_dist
        dist = np.clip(dist, 0.0, 1.0)
        if self.objaware_dist_target_transform == 'exp':
            alpha = float(self.objaware_dist_target_exp_alpha)
            denom = np.expm1(alpha)
            if np.abs(denom) < 1e-12:
                dist = dist.astype(np.float32, copy=False)
            else:
                dist = (np.expm1(alpha * dist) / denom).astype(np.float32, copy=False)
        else:
            if self.objaware_dist_target_power != 1.0:
                dist = np.power(dist, self.objaware_dist_target_power).astype(np.float32, copy=False)
        return torch.from_numpy(dist).to(mask.device, dtype=torch.float32)

    def _build_discriminative_mask(self,
                                   ds_masks: Tensor,
                                   labels: Tensor,
                                   radius: int,
                                   local_only: bool,
                                   same_class_only: bool) -> Tensor:
        """Build pair mask indicating which instance pairs should be pushed apart."""
        n = ds_masks.shape[0]
        device = ds_masks.device
        pair_mask = torch.zeros((n, n), dtype=torch.bool, device=device)

        if n <= 1:
            return pair_mask

        if local_only:
            # A pair is considered neighbors if mask_j intersects dilate(mask_i, r)
            masks_f = ds_masks.float().unsqueeze(1)
            dilated = F.max_pool2d(
                masks_f, kernel_size=2 * radius + 1, stride=1, padding=radius).squeeze(1) > 0
            flat_masks = ds_masks.reshape(n, -1)
            for i in range(n):
                overlap = (dilated[i].reshape(1, -1) & flat_masks).any(dim=1)
                overlap[i] = False
                pair_mask[i] = overlap
            pair_mask = pair_mask | pair_mask.t()
        else:
            pair_mask[:] = True
            pair_mask.fill_diagonal_(False)

        if same_class_only:
            class_same = labels.view(-1, 1) == labels.view(1, -1)
            pair_mask = pair_mask & class_same
            pair_mask.fill_diagonal_(False)

        return pair_mask

    @staticmethod
    def _build_boundary_target(ds_masks: Tensor, width: int) -> Tensor:
        """Build a thick boundary band target from instance masks."""
        if ds_masks.numel() == 0 or width <= 0:
            return torch.zeros_like(ds_masks[0], dtype=torch.float32)

        masks_f = ds_masks.float().unsqueeze(1)
        kernel = 2 * width + 1
        dilated = F.max_pool2d(
            masks_f, kernel_size=kernel, stride=1, padding=width)
        eroded = 1.0 - F.max_pool2d(
            1.0 - masks_f, kernel_size=kernel, stride=1, padding=width)
        boundary = (dilated - eroded).squeeze(1) > 0.5
        return boundary.any(dim=0).float()
