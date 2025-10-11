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
        """Combine different feature types into single frame representation (vectorized)"""

        mouse_center = features_dict['mouse_center']
        pairwise = features_dict['pairwise']

        # Pivot mouse_center features
        mc_pivot = mouse_center.pivot_table(index='video_frame', columns='mouse_id',
                                            values=['cx', 'cy', 'vx', 'vy', 'speed', 'vis_frac'])
        mc_pivot.columns = ['_'.join(map(str, col)) for col in mc_pivot.columns]

        # Pivot pairwise features
        pw_pivot = pairwise.pivot_table(index='video_frame', columns='pair_id',
                                        values=['dist', 'rel_x', 'rel_y', 'rel_vx', 'rel_vy', 'rel_speed'])
        pw_pivot.columns = ['_'.join(map(str, col)) for col in pw_pivot.columns]

        # Merge features
        combined = pd.concat([mc_pivot, pw_pivot], axis=1)

        # Get unique frames
        frames = sorted(mouse_center['video_frame'].unique())
        combined = combined.reindex(frames)

        # Pad or truncate to feature_dim
        if combined.shape[1] < self.feature_dim:
            padding = pd.DataFrame(np.zeros((combined.shape[0], self.feature_dim - combined.shape[1])),
                                   index=combined.index)
            combined = pd.concat([combined, padding], axis=1)
        elif combined.shape[1] > self.feature_dim:
            combined = combined.iloc[:, :self.feature_dim]

        # Replace NaN with 0 and convert to numpy
        combined_np = np.nan_to_num(combined.values, 0).astype(np.float32)

        return combined_np, frames
    
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