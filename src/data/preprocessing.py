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
        """Create aggregated center features per mouse (vectorized)"""

        # Get x and y columns
        x_cols = [c for c in raw_pose.columns if c.startswith('x_')]
        y_cols = [c for c in raw_pose.columns if c.startswith('y_')]

        # Calculate center of mass and visibility fraction
        df = raw_pose.copy()
        df['cx'] = df[x_cols].mean(axis=1)
        df['cy'] = df[y_cols].mean(axis=1)
        df['vis_frac'] = df[x_cols].notna().sum(axis=1) / len(x_cols)

        # Select and sort columns
        df = df[['video_frame', 'mouse_id', 'cx', 'cy', 'vis_frac']].sort_values(['video_frame', 'mouse_id'])

        # Calculate velocities using groupby().diff()
        df[['vx', 'vy']] = df.groupby('mouse_id')[['cx', 'cy']].diff()
        df['speed'] = np.sqrt(df['vx']**2 + df['vy']**2)

        # Fill NaN velocities with 0
        df[['vx', 'vy', 'speed']] = df[['vx', 'vy', 'speed']].fillna(0)

        return df
    
    def _create_pairwise_features(self, mouse_center: pd.DataFrame):
        """Create pairwise distance and relative motion features (vectorized)"""

        # Merge to create all pairs
        pairs = pd.merge(mouse_center, mouse_center, on='video_frame', suffixes=('_agent', '_target'))

        # Filter for valid pairs (agent_id >= target_id)
        pairs = pairs[pairs['mouse_id_agent'] >= pairs['mouse_id_target']].copy()

        # Calculate pairwise features
        pairs['dist'] = np.sqrt((pairs['cx_agent'] - pairs['cx_target'])**2 +
                                (pairs['cy_agent'] - pairs['cy_target'])**2)
        pairs['rel_x'] = pairs['cx_agent'] - pairs['cx_target']
        pairs['rel_y'] = pairs['cy_agent'] - pairs['cy_target']
        pairs['rel_vx'] = pairs['vx_agent'] - pairs['vx_target']
        pairs['rel_vy'] = pairs['vy_agent'] - pairs['vy_target']
        pairs['rel_speed'] = np.sqrt(pairs['rel_vx']**2 + pairs['rel_vy']**2)

        # Create pair_id
        pairs['pair_id'] = pairs['mouse_id_agent'].astype(str) + '_' + pairs['mouse_id_target'].astype(str)

        # Select and rename columns
        pairs = pairs.rename(columns={'mouse_id_agent': 'agent_id', 'mouse_id_target': 'target_id'})
        
        feature_cols = [
            'video_frame', 'pair_id', 'agent_id', 'target_id',
            'dist', 'rel_x', 'rel_y', 'rel_vx', 'rel_vy', 'rel_speed'
        ]
        
        return pairs[feature_cols]
