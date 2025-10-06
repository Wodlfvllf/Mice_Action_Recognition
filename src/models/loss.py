import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class HungarianMatcher:
    """Matches predictions to ground truth using Hungarian algorithm"""
    
    def __init__(self, cost_class: float = 1, cost_span: float = 5, cost_iou: float = 2):
        self.cost_class = cost_class
        self.cost_span = cost_span
        self.cost_iou = cost_iou
    
    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Performs the matching
        
        Args:
            outputs: dict with 'class_logits' and 'spans'
            targets: list of dicts with 'class_id', 'start_frac', 'end_frac'
        
        Returns:
            List of (pred_idx, target_idx) tuples for each batch element
        """
        batch_size = outputs['class_logits'].shape[0]
        num_queries = outputs['class_logits'].shape[1]
        
        # Flatten predictions
        out_prob = outputs['class_logits'].flatten(0, 1).softmax(-1)  # (batch*queries, classes+1)
        out_spans = outputs['spans'].flatten(0, 1)  # (batch*queries, 2)
        
        # Prepare targets
        tgt_ids = torch.cat([
            torch.tensor([t['class_id'] for t in tgt], dtype=torch.long)
            for tgt in targets
        ])
        
        tgt_spans = torch.cat([
            torch.tensor([[t['start_frac'], t['end_frac']] for t in tgt], dtype=torch.float)
            for tgt in targets
        ])
        
        # Compute cost matrices
        cost_class = -out_prob[:, tgt_ids]
        cost_span = torch.cdist(out_spans, tgt_spans, p=1)
        
        # Compute IoU cost
        cost_iou = -self.span_iou(out_spans, tgt_spans)
        
        # Final cost matrix
        C = self.cost_span * cost_span + self.cost_class * cost_class + self.cost_iou * cost_iou
        C = C.view(batch_size, num_queries, -1).cpu()
        
        # Hungarian matching for each batch element
        indices = []
        for i, c in enumerate(C.split([len(t) for t in targets], -1)):
            indices.append(linear_sum_assignment(c[i]))
        
        return [(torch.as_tensor(i, dtype=torch.int64), 
                torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
    
    def span_iou(self, spans1, spans2):
        """Compute IoU between spans"""
        # spans: (N, 2) with [start, end]
        
        # Expand for pairwise computation
        spans1 = spans1.unsqueeze(1)  # (N, 1, 2)
        spans2 = spans2.unsqueeze(0)  # (1, M, 2)
        
        # Intersection
        inter_start = torch.max(spans1[..., 0], spans2[..., 0])
        inter_end = torch.min(spans1[..., 1], spans2[..., 1])
        inter_len = (inter_end - inter_start).clamp(min=0)
        
        # Union
        union_start = torch.min(spans1[..., 0], spans2[..., 0])
        union_end = torch.max(spans1[..., 1], spans2[..., 1])
        union_len = union_end - union_start
        
        # IoU
        iou = inter_len / (union_len + 1e-6)
        
        return iou

class DETRLoss(nn.Module):
    """DETR loss with Hungarian matching"""
    
    def __init__(self, num_classes: int = 13, no_object_weight: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = HungarianMatcher()
        
        # Loss weights
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = no_object_weight
        self.register_buffer('empty_weight', empty_weight)
    
    def forward(self, outputs, targets):
        """Compute the loss"""
        
        # Get matching
        indices = self.matcher.forward(outputs, targets)
        
        # Compute losses
        loss_ce = self.loss_labels(outputs, targets, indices)
        loss_span = self.loss_spans(outputs, targets, indices)
        
        losses = {
            'loss_ce': loss_ce,
            'loss_span': loss_span,
            'loss_total': loss_ce + 5 * loss_span
        }
        
        return losses
    
    def loss_labels(self, outputs, targets, indices):
        """Classification loss"""
        
        src_logits = outputs['class_logits']
        
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([
            torch.tensor([t['class_id'] for t in tgt], dtype=torch.long)
            for tgt in targets
        ])
        
        target_classes = torch.full(
            src_logits.shape[:2], 
            self.num_classes,
            dtype=torch.long, 
            device=src_logits.device
        )
        target_classes[idx] = target_classes_o.to(src_logits.device)
        
        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2), 
            target_classes, 
            self.empty_weight
        )
        
        return loss_ce
    
    def loss_spans(self, outputs, targets, indices):
        """Span regression loss"""
        
        idx = self._get_src_permutation_idx(indices)
        src_spans = outputs['spans'][idx]
        
        target_spans = torch.cat([
            torch.tensor([[t['start_frac'], t['end_frac']] for t in tgt], dtype=torch.float)
            for tgt in targets
        ])
        
        if len(target_spans) == 0:
            return torch.tensor(0.0, device=outputs['spans'].device)
        
        loss_span = F.l1_loss(src_spans, target_spans.to(src_spans.device))
        
        return loss_span
    
    def _get_src_permutation_idx(self, indices):
        """Helper to get permutation indices"""
        batch_idx = torch.cat([
            torch.full_like(src, i) for i, (src, _) in enumerate(indices)
        ])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx