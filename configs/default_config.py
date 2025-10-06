from pathlib import Path

def get_default_config():
    """Get default configuration"""
    
    config = {
        # Data paths
        'data_dir': Path('./dataset/'),
        'output_dir': Path('.'),
        
        # Window parameters
        'window_size': 64,
        'stride': 32,
        'feature_dim': 256,
        
        # Model parameters
        'd_model': 256,
        'nhead': 8,
        'num_encoder_layers': 6,
        'num_decoder_layers': 6,
        'num_queries': 100,
        'num_classes': 13,
        'dropout': 0.1,
        
        # Training parameters
        'batch_size': 16,
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4
    }
    
    return config
