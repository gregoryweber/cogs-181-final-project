import torch
import torch.nn as nn
from .modules import PMA, ISAB, SAB, PISAB

class HAITiterPredictor(nn.Module):
    """
    Modified Set Transformer for predicting H1, H3, and B values from flow cytometry data
    
    Args:
        num_markers: Number of markers in the FCS file
        dim_hidden: Dimension of hidden representation
        num_heads: Number of attention heads
        num_inds: Number of induced points for ISAB
        hidden_layers: Number of hidden layers
        layer_norm: Whether to use layer normalization
    """
    def __init__(self, 
                 num_markers, 
                 dim_hidden=128, 
                 num_heads=4, 
                 num_inds=32,
                 hidden_layers=2,
                 layer_norm=True,
                 dropout=0.1):
        super(HAITiterPredictor, self).__init__()
        
        # Store configuration
        self.num_markers = num_markers
        
        # Encoder - processes each cell as part of a set
        enc_layers = [ISAB(num_markers, dim_hidden, num_heads, num_inds, ln=layer_norm)]
        for _ in range(1, hidden_layers):
            enc_layers.append(ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=layer_norm))
        self.enc = nn.Sequential(*enc_layers)
        
        # Pooling Multi-head Attention to aggregate cell information
        self.pma = PMA(dim_hidden, num_heads, 1)  # Single output representation
        
        # MLP for predicting the three titer values
        self.mlp = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_hidden, dim_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_hidden // 2, 3)  # Output: H1, H3, B
        )
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, num_cells, num_markers]
                where each sample contains a variable number of cells
                
        Returns:
            Tensor of shape [batch_size, 3] with predicted H1, H3, B values
        """
        # Apply set transformer to process all cells
        h = self.enc(x)  # [batch_size, num_cells, dim_hidden]
        
        # Aggregate cell information using PMA
        h = self.pma(h)  # [batch_size, 1, dim_hidden]
        
        # Predict titer values
        titer_preds = self.mlp(h.squeeze(1))  # [batch_size, 3]
        
        return titer_preds