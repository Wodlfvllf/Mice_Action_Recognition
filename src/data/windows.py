from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


class WindowCreator:
    """Creates sliding windows from features"""
    
    def __init__(self, window_size: int = 64, stride: int = 32, feature_dim: int = 256):
        self.window_size = window_size
        self.stride = stride
        self.feature_dim = feature_dim
    
    def create_windows(self, features_dict: Dict, video_id: str, output_dir: Path):
        """Create sliding windows from features"""
        
        # Combine all features into single representation
        combined_features = self._combine_features(features_dict)
        
        # Create sliding windows
        windows, window_meta = self._create_sliding_windows(
            combined_features, video_id
        )
        
        # Save windows and metadata
        np.save(output_dir / f'windows/{video_id}_windows.npy', windows)
        window_meta.to_csv(
            output_dir / f'windows/{video_id}_meta.csv', 
            index=False
        )
        
        return windows, window_meta
    
    def _combine_features(self, features_dict: Dict):
        """Combine different feature types into single frame representation"""
        
        mouse_center = features_dict['mouse_center']
        pairwise = features_dict['pairwise']
        
        # Get unique frames
        frames = sorted(mouse_center['video_frame'].unique())
        n_frames = len(frames)
        
        # Initialize combined feature array
        combined = np.zeros((n_frames, self.feature_dim), dtype=np.float32)
        
        for i, frame in enumerate(frames):
            features = []
            
            # Mouse center features (4 mice × 6 features = 24)
            frame_mice = mouse_center[mouse_center['video_frame'] == frame]
            for _, mouse in frame_mice.iterrows():
                features.extend([
                    mouse['cx'], mouse['cy'], 
                    mouse['vx'], mouse['vy'],
                    mouse['speed'], mouse['vis_frac']
                ])
            
            # Pairwise features (6 pairs × 6 features = 36)
            frame_pairs = pairwise[pairwise['video_frame'] == frame]
            for _, pair in frame_pairs.iterrows():
                features.extend([
                    pair['dist'], pair['rel_x'], pair['rel_y'],
                    pair['rel_vx'], pair['rel_vy'], pair['rel_speed']
                ])
            
            # Pad or truncate to feature_dim
            features = np.array(features)
            if len(features) < self.feature_dim:
                features = np.pad(features, (0, self.feature_dim - len(features)))
            else:
                features = features[:self.feature_dim]
            
            combined[i] = features
        
        # Replace NaN with 0
        combined = np.nan_to_num(combined, 0)
        
        return combined, frames
    
    def _create_sliding_windows(self, combined_features, video_id: str):
        """Create sliding windows with metadata"""
        
        features, frames = combined_features
        n_frames = len(frames)
        
        windows = []
        window_meta = []
        
        for start_idx in range(0, n_frames - self.window_size + 1, self.stride):
            end_idx = start_idx + self.window_size
            
            # Extract window
            window = features[start_idx:end_idx]
            windows.append(window)
            
            # Create metadata
            window_meta.append({
                'video_id': video_id,
                'window_idx': len(windows) - 1,
                'start_frame': frames[start_idx],
                'end_frame': frames[end_idx - 1],
                'start_idx': start_idx,
                'end_idx': end_idx
            })
        
        windows = np.array(windows, dtype=np.float32)
        window_meta = pd.DataFrame(window_meta)
        
        return windows, window_meta