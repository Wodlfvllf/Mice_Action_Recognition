import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    

class MouseBehaviorDETR(nn.Module):
    """DETR-style model for mouse behavior detection"""
    
    def __init__(
        self,
        feature_dim: int = 256,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        num_queries: int = 100,
        num_classes: int = 13,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_queries = num_queries
        self.num_classes = num_classes
        
        # Input projection
        self.input_proj = nn.Linear(feature_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
            batch_first=True
        )
        
        # Query embeddings
        self.query_embed = nn.Embedding(num_queries, d_model)
        
        # Detection heads
        self.class_head = nn.Linear(d_model, num_classes + 1)  # +1 for no-object
        self.span_head = nn.Linear(d_model, 2)  # start_frac, end_frac
        self.agent_head = nn.Linear(d_model, 4)  # 4 mice
        self.target_head = nn.Linear(d_model, 4)  # 4 mice
    
    def forward(self, x):
        """
        Args:
            x: (batch, window_size, feature_dim)
        Returns:
            dict with predictions for each query
        """
        batch_size = x.size(0)
        
        # Project input features
        x = self.input_proj(x)  # (batch, window_size, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Generate query embeddings
        queries = self.query_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Transformer forward pass
        decoder_out = self.transformer(x, queries)
        
        # Apply detection heads
        outputs = {
            'class_logits': self.class_head(decoder_out),  # (batch, queries, classes+1)
            'spans': torch.sigmoid(self.span_head(decoder_out)),  # (batch, queries, 2)
            'agent_logits': self.agent_head(decoder_out),  # (batch, queries, 4)
            'target_logits': self.target_head(decoder_out)  # (batch, queries, 4)
        }
        
        return outputs
