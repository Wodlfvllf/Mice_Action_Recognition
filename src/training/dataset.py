from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class MouseBehaviorDataset(Dataset):
    """Dataset for mouse behavior detection"""
    
    def __init__(self, windows_dir: Path, labels_dir: Path, video_ids: List[str]):
        self.windows_dir = Path(windows_dir)
        self.labels_dir = Path(labels_dir)
        self.video_ids = video_ids
        
        # Load all data into memory
        self.data = []
        for video_id in tqdm(video_ids, desc="Loading dataset"):
            windows = np.load(self.windows_dir / f'{video_id}_windows.npy')
            labels = np.load(self.labels_dir / f'{video_id}_labels.npy', allow_pickle=True)
            
            for i in range(len(windows)):
                self.data.append({
                    'window': windows[i],
                    'labels': labels[i] if i < len(labels) else []
                })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return torch.FloatTensor(item['window']), item['labels']
