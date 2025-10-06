from typing import Dict, List

import numpy as np
import pandas as pd
import torch


class InferenceEngine:
    """Handles inference and submission generation"""
    
    def __init__(self, model, label_encoder, window_size: int = 64, stride: int = 32):
        self.model = model
        self.label_encoder = label_encoder
        self.window_size = window_size
        self.stride = stride
    
    def predict_video(self, windows: np.ndarray, window_meta: pd.DataFrame):
        """Run inference on a video's windows"""
        
        self.model.eval()
        all_predictions = []
        
        with torch.no_grad():
            # Process in batches
            batch_size = 32
            for i in range(0, len(windows), batch_size):
                batch = torch.FloatTensor(windows[i:i+batch_size])
                outputs = self.model(batch)
                
                # Extract predictions
                for b in range(batch.size(0)):
                    window_info = window_meta.iloc[i + b]
                    preds = self._extract_predictions(outputs, b, window_info)
                    all_predictions.extend(preds)
        
        # Merge overlapping predictions
        merged = self._merge_predictions(all_predictions)
        
        return merged
    
    def _extract_predictions(self, outputs, batch_idx, window_info):
        """Extract predictions from model output"""
        
        predictions = []
        
        # Get predictions for this sample
        class_logits = outputs['class_logits'][batch_idx]
        spans = outputs['spans'][batch_idx]
        agent_logits = outputs['agent_logits'][batch_idx]
        target_logits = outputs['target_logits'][batch_idx]
        
        # Get class predictions
        class_probs = class_logits.softmax(-1)
        class_ids = class_probs.argmax(-1)
        
        # Filter out no-object predictions
        valid_mask = class_ids != self.model.num_classes
        
        if not valid_mask.any():
            return predictions
        
        # Convert relative spans to absolute frames
        start_frames = window_info['start_frame'] + spans[:, 0] * self.window_size
        end_frames = window_info['start_frame'] + spans[:, 1] * self.window_size
        
        # Get agent/target predictions
        agents = agent_logits.argmax(-1) + 1  # Mouse IDs start at 1
        targets = target_logits.argmax(-1) + 1
        
        for i in torch.where(valid_mask)[0]:
            predictions.append({
                'video_id': window_info['video_id'],
                'action': self.label_encoder.inverse_transform([class_ids[i].item()])[0],
                'agent_id': agents[i].item(),
                'target_id': targets[i].item(),
                'start_frame': int(start_frames[i].item()),
                'stop_frame': int(end_frames[i].item()),
                'confidence': class_probs[i, class_ids[i]].item()
            })
        
        return predictions
    
    def _merge_predictions(self, predictions):
        """Merge overlapping predictions using NMS-like approach"""
        
        if not predictions:
            return []
        
        # Group by action and agent-target pair
        grouped = {}
        for pred in predictions:
            key = (pred['action'], pred['agent_id'], pred['target_id'])
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(pred)
        
        merged = []
        
        for key, group in grouped.items():
            # Sort by confidence
            group = sorted(group, key=lambda x: x['confidence'], reverse=True)
            
            # Apply NMS
            keep = []
            for pred in group:
                # Check overlap with already kept predictions
                overlap = False
                for kept in keep:
                    iou = self._compute_iou(
                        (pred['start_frame'], pred['stop_frame']),
                        (kept['start_frame'], kept['stop_frame'])
                    )
                    if iou > 0.5:  # Overlap threshold
                        overlap = True
                        break
                
                if not overlap:
                    keep.append(pred)
            
            merged.extend(keep)
        
        return merged
    
    def _compute_iou(self, span1, span2):
        """Compute IoU between two spans"""
        start1, end1 = span1
        start2, end2 = span2
        
        # Intersection
        inter_start = max(start1, start2)
        inter_end = min(end1, end2)
        inter_len = max(0, inter_end - inter_start)
        
        # Union
        union_start = min(start1, start2)
        union_end = max(end1, end2)
        union_len = union_end - union_start
        
        if union_len == 0:
            return 0
        
        return inter_len / union_len
    
    def create_submission(self, predictions: List[Dict]) -> pd.DataFrame:
        """Create submission dataframe"""
        
        submission = []
        row_id = 0
        
        for pred in predictions:
            submission.append({
                'row_id': row_id,
                'video_id': pred['video_id'],
                'agent_id': pred['agent_id'],
                'target_id': pred['target_id'],
                'action': pred['action'],
                'start_frame': pred['start_frame'],
                'stop_frame': pred['stop_frame']
            })
            row_id += 1
        
        return pd.DataFrame(submission)
