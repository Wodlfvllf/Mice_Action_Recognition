from pathlib import Path

import numpy as np
import pandas as pd


class DataPreprocessor:
    """Preprocesses tracking parquet files into feature representations"""
    
    def __init__(self, data_dir: Path, output_dir: Path):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create feature subdirectories
        self.dirs = {
            'raw_pose': self.output_dir / 'features/raw_pose',
            'mouse_center': self.output_dir / 'features/mouse_center',
            'pairwise': self.output_dir / 'features/pairwise',
            'windows': self.output_dir / 'windows',
            'labels': self.output_dir / 'labels'
        }
        for d in self.dirs.values():
            d.mkdir(exist_ok=True, parents=True)
    
    def process_video(self, video_id: str, lab_id: str, metadata: pd.Series):
        """Process a single video's tracking data"""
        
        # Load tracking data
        tracking_path = self.data_dir / f'train_tracking/{lab_id}/{video_id}.parquet'
        if not tracking_path.exists():
            return None
            
        tracking_df = pd.read_parquet(tracking_path)
        
        # 1. Create raw pose features (per-frame, per-mouse, all keypoints)
        raw_pose = self._create_raw_pose_features(tracking_df, metadata)
        raw_pose.to_parquet(self.dirs['raw_pose'] / f'{video_id}.parquet')
        
        # 2. Create mouse center features (aggregated per mouse)
        mouse_center = self._create_mouse_center_features(raw_pose, metadata)
        mouse_center.to_parquet(self.dirs['mouse_center'] / f'{video_id}.parquet')
        
        # 3. Create pairwise features
        pairwise = self._create_pairwise_features(mouse_center)
        pairwise.to_parquet(self.dirs['pairwise'] / f'{video_id}.parquet')
        
        return {
            'raw_pose': raw_pose,
            'mouse_center': mouse_center,
            'pairwise': pairwise
        }
    
    def _create_raw_pose_features(self, tracking_df: pd.DataFrame, metadata: pd.Series):
        """Reshape tracking data to wide format with all keypoints per mouse"""
        
        # Pivot to get all bodyparts as columns
        pivot = tracking_df.pivot_table(
            index=['video_frame', 'mouse_id'],
            columns='bodypart',
            values=['x', 'y'],
            fill_value=np.nan
        )
        
        # Flatten column names
        pivot.columns = ['_'.join(col).strip() for col in pivot.columns.values]
        pivot = pivot.reset_index()
        
        # Add visibility flags (1 if tracked, 0 if missing)
        for col in pivot.columns:
            if col.startswith('x_') or col.startswith('y_'):
                bodypart = col[2:]
                vis_col = f'vis_{bodypart}'
                if vis_col not in pivot.columns:
                    pivot[vis_col] = (~pivot[col].isna()).astype(float)
        
        return pivot
    
    def _create_mouse_center_features(self, raw_pose: pd.DataFrame, metadata: pd.Series):
        """Create aggregated center features per mouse"""
        
        features = []
        
        for (frame, mouse_id), group in raw_pose.groupby(['video_frame', 'mouse_id']):
            x_cols = [c for c in group.columns if c.startswith('x_')]
            y_cols = [c for c in group.columns if c.startswith('y_')]
            
            # Center as mean of visible keypoints
            x_vals = group[x_cols].values[0]
            y_vals = group[y_cols].values[0]
            
            x_valid = x_vals[~np.isnan(x_vals)]
            y_valid = y_vals[~np.isnan(y_vals)]
            
            if len(x_valid) > 0:
                cx = np.mean(x_valid)
                cy = np.mean(y_valid)
                vis_frac = len(x_valid) / len(x_cols)
            else:
                cx = cy = np.nan
                vis_frac = 0.0
            
            features.append({
                'video_frame': frame,
                'mouse_id': mouse_id,
                'cx': cx,
                'cy': cy,
                'vis_frac': vis_frac
            })
        
        df = pd.DataFrame(features).sort_values(['video_frame', 'mouse_id'])
        
        # Add velocity features
        for mouse_id in df['mouse_id'].unique():
            mask = df['mouse_id'] == mouse_id
            mouse_df = df[mask].copy()
            
            # Compute velocities (frame-to-frame differences)
            mouse_df['vx'] = mouse_df['cx'].diff()
            mouse_df['vy'] = mouse_df['cy'].diff()
            mouse_df['speed'] = np.sqrt(mouse_df['vx']**2 + mouse_df['vy']**2)
            
            df.loc[mask, ['vx', 'vy', 'speed']] = mouse_df[['vx', 'vy', 'speed']].values
        
        # Fill NaN velocities with 0
        df[['vx', 'vy', 'speed']] = df[['vx', 'vy', 'speed']].fillna(0)
        
        return df
    
    def _create_pairwise_features(self, mouse_center: pd.DataFrame):
        """Create pairwise distance and relative motion features"""
        
        features = []
        
        for frame, group in mouse_center.groupby('video_frame'):
            mice = group['mouse_id'].values
            
            for i, m1 in enumerate(mice):
                for j, m2 in enumerate(mice):
                    if i >= j:  # Include self-pairs
                        m1_data = group[group['mouse_id'] == m1].iloc[0]
                        m2_data = group[group['mouse_id'] == m2].iloc[0]
                        
                        # Distance and relative position
                        dist = np.sqrt((m1_data['cx'] - m2_data['cx'])**2 + 
                                      (m1_data['cy'] - m2_data['cy'])**2)
                        rel_x = m1_data['cx'] - m2_data['cx']
                        rel_y = m1_data['cy'] - m2_data['cy']
                        
                        # Relative velocity
                        rel_vx = m1_data['vx'] - m2_data['vx']
                        rel_vy = m1_data['vy'] - m2_data['vy']
                        rel_speed = np.sqrt(rel_vx**2 + rel_vy**2)
                        
                        features.append({
                            'video_frame': frame,
                            'pair_id': f'{m1}_{m2}',
                            'agent_id': m1,
                            'target_id': m2,
                            'dist': dist,
                            'rel_x': rel_x,
                            'rel_y': rel_y,
                            'rel_vx': rel_vx,
                            'rel_vy': rel_vy,
                            'rel_speed': rel_speed
                        })
        
        return pd.DataFrame(features)
