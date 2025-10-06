

    def train(self):
        """Train the model"""
        
        print("=" * 50)
        print("STEP 2: Training Model")
        print("=" * 50)
        
        # Get all processed videos with labels
        label_files = list(self.preprocessor.dirs['labels'].glob('*.npy'))
        video_ids = [f.stem.replace('_labels', '') for f in label_files]
        
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