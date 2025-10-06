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