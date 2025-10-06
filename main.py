import pandas as pd
from mice_recog.configs import get_default_config
from mice_recog.src.pipeline import CompletePipeline

def main():
    """Main execution function"""
    
    print("=" * 60)
    print("MOUSE BEHAVIOR DETECTION TRAINING PIPELINE")
    print("DETR-Style Transformer Architecture")
    print("=" * 60)
    
    # Get configuration
    config = get_default_config()
    
    # Initialize pipeline
    pipeline = CompletePipeline(config)
    
    # Process training data
    pipeline.process_data()
    
    # Train model
    pipeline.train()
    
    # Run inference on test data
    test_df = pd.read_csv(config['data_dir'] / 'test.csv')
    submission = pipeline.inference(test_df)
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Submission saved to: {config['output_dir'] / 'submission.csv'}")
    print(f"Model saved to: {config['output_dir'] / 'best_model.pth'}")
    
    return submission

if __name__ == "__main__":
    main()
