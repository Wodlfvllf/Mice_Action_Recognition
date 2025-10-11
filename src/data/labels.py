from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder


class LabelCreator:
    """Creates DETR-style labels for windows"""
    
    def __init__(self, window_size: int = 64, num_classes: int = 13):
        self.window_size = window_size
        self.num_classes = num_classes
        self.label_encoder = LabelEncoder()

    def fit(self, annotation_paths: list[Path]):
        """Fit label encoder on all behaviors"""
        all_behaviors = set()
        for path in annotation_paths:
            annotations = pd.read_parquet(path)
            all_behaviors.update(annotations['action'].unique())
        self.label_encoder.fit(list(all_behaviors))
    
    def create_labels(self, annotations_path: Path, window_meta: pd.DataFrame):
        """Create DETR-style labels for each window"""
        
        # Load annotations
        annotations = pd.read_parquet(annotations_path)
        
        # Create labels for each window
        window_labels = []
        
        for _, window in window_meta.iterrows():
            # Find annotations overlapping with this window
            overlapping = annotations[
                (annotations['start_frame'] < window['end_frame']) &
                (annotations['stop_frame'] > window['start_frame'])
            ]
            
            # Convert to relative positions within window
            gt_spans = []
            for _, ann in overlapping.iterrows():
                # Clip to window boundaries
                start = max(ann['start_frame'], window['start_frame'])
                end = min(ann['stop_frame'], window['end_frame'])
                
                # Convert to relative position [0, 1]
                start_frac = (start - window['start_frame']) / self.window_size
                end_frac = (end - window['start_frame']) / self.window_size
                
                # Get class label
                class_id = self.label_encoder.transform([ann['action']])[0]
                
                gt_spans.append({
                    'start_frac': start_frac,
                    'end_frac': end_frac,
                    'class_id': class_id,
                    'agent_id': ann['agent_id'],
                    'target_id': ann['target_id']
                })
            
            window_labels.append(gt_spans)
        
        return window_labels
