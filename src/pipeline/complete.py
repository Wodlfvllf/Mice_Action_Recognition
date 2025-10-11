from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data import DataPreprocessor, WindowCreator, LabelCreator
from ..models import MouseBehaviorDETR, DETRLoss
from ..training import MouseBehaviorDataset, train_epoch, evaluate
from ..inference import InferenceEngine


class CompletePipeline:
    """Complete end-to-end training pipeline"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.preprocessor = DataPreprocessor(
            config['data_dir'],
            config['output_dir']
        )
        
        self.window_creator = WindowCreator(
            window_size=config['window_size'],
            stride=config['stride'],
            feature_dim=config['feature_dim']
        )
        
        self.label_creator = LabelCreator(
            window_size=config['window_size'],
            num_classes=config['num_classes']
        )
        
        # Initialize model
        self.model = MouseBehaviorDETR(
            feature_dim=config['feature_dim'],
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_encoder_layers=config['num_encoder_layers'],
            num_decoder_layers=config['num_decoder_layers'],
            num_queries=config['num_queries'],
            num_classes=config['num_classes'],
            dropout=config['dropout']
        ).to(self.device)
        
        self.criterion = DETRLoss(num_classes=config['num_classes'])
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['num_epochs']
        )
    
    def process_data(self):
        """Process all training data"""
        
        print("=" * 50)
        print("STEP 1: Processing Training Data")
        print("=" * 50)
        
        # Load metadata
        train_df = pd.read_csv(self.config['data_dir'] / 'train.csv')

        # Fit label encoder
        annotation_paths = []
        for idx, row in train_df.iterrows():
            ann_path = self.config['data_dir'] / f"train_annotation/{row['lab_id']}/{row['video_id']}.parquet"
            if ann_path.exists():
                annotation_paths.append(ann_path)
        self.label_creator.fit(annotation_paths)
        
        # Process each video
        for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Processing videos"):
            video_id = row['video_id']
            lab_id = row['lab_id']
            
            # Skip if already processed
            if (self.preprocessor.dirs['windows'] / f'{video_id}_windows.npy').exists():
                continue
            
            # Process features
            features = self.preprocessor.process_video(video_id, lab_id, row)
            if features is None:
                continue
            
            # Create windows
            windows, window_meta = self.window_creator.create_windows(
                features, video_id, self.preprocessor.output_dir
            )
            
            # Create labels if annotations exist
            ann_path = self.config['data_dir'] / f'train_annotation/{lab_id}/{video_id}.parquet'
            if ann_path.exists():
                labels = self.label_creator.create_labels(ann_path, window_meta)
                np.save(
                    self.preprocessor.dirs['labels'] / f'{video_id}_labels.npy',
                    labels,
                    allow_pickle=True
                )
    
    def train(self):
        """Train the model"""
        
        print("=" * 50)
        print("STEP 2: Training Model")
        print("=" * 50)
        
        # Get all processed videos with labels
        label_files = list(self.preprocessor.dirs['labels'].glob('*.npy'))
        video_ids = [f.stem.replace('_labels', '') for f in label_files]
        
        # Shuffle video_ids for random split
        video_ids = np.random.permutation(video_ids)

        # Split train/val
        n_train = int(0.8 * len(video_ids))
        train_ids = video_ids[:n_train]
        val_ids = video_ids[n_train:]
        
        print(f"Training videos: {len(train_ids)}")
        print(f"Validation videos: {len(val_ids)}")
        
        # Create datasets
        train_dataset = MouseBehaviorDataset(
            self.preprocessor.dirs['windows'],
            self.preprocessor.dirs['labels'],
            train_ids
        )
        
        val_dataset = MouseBehaviorDataset(
            self.preprocessor.dirs['windows'],
            self.preprocessor.dirs['labels'],
            val_ids
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=4,
            collate_fn=self._collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=4,
            collate_fn=self._collate_fn
        )
        
        # Training loop
        best_loss = float('inf')
        
        for epoch in range(self.config['num_epochs']):
            print(f"\nEpoch {epoch + 1}/{self.config['num_epochs']}")
            print("-" * 40)
            
            # Train
            train_loss = train_epoch(
                self.model, train_loader, self.criterion, self.optimizer, self.device
            )
            
            # Validate
            val_predictions = evaluate(self.model, val_loader, self.device)
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Validation Predictions: {len(val_predictions)} windows with detections")
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Save best model
            if train_loss < best_loss:
                best_loss = train_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': train_loss,
                    'config': self.config
                }, self.config['output_dir'] / 'best_model.pth')
                print(f"Saved best model with loss {train_loss:.4f}")
    
    def inference(self, test_df: pd.DataFrame):
        """Run inference on test data"""
        
        print("=" * 50)
        print("STEP 3: Running Inference")
        print("=" * 50)
        
        # Load best model
        checkpoint = torch.load(self.config['output_dir'] / 'best_model.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Initialize inference engine
        inference_engine = InferenceEngine(
            self.model,
            self.label_creator.label_encoder,
            window_size=self.config['window_size'],
            stride=self.config['stride']
        )
        
        all_predictions = []
        
        # Process each test video
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Processing test videos"):
            video_id = row['video_id']
            lab_id = row['lab_id']
            
            # Process features
            features = self.preprocessor.process_video(video_id, lab_id, row)
            if features is None:
                continue
            
            # Create windows
            windows, window_meta = self.window_creator.create_windows(
                features, video_id, self.preprocessor.output_dir
            )
            
            # Run inference
            predictions = inference_engine.predict_video(windows, window_meta)
            all_predictions.extend(predictions)
        
        # Create submission
        submission = inference_engine.create_submission(all_predictions)
        submission.to_csv(self.config['output_dir'] / 'submission.csv', index=False)
        
        print(f"Generated submission with {len(submission)} predictions")
        
        return submission
    
    def _collate_fn(self, batch):
        """Custom collate function for variable-length labels"""
        windows = torch.stack([item[0] for item in batch])
        labels = [item[1] for item in batch]
        return windows, labels