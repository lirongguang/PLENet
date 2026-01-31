import sys
import numpy as np
import math

import torch
from torch import nn
import torch.nn.functional as F

sys.path.append("../")
from config import DefaultConfig
configs = DefaultConfig()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EGRETLayer(nn.Module):
    """Equivariant graph convolution layer - single-head version"""
    def __init__(self, in_dim, out_dim, edge_dim, use_bias=True, config_dict=None):
        super(EGRETLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_dim = edge_dim
        self.use_bias = use_bias
        
        # Configuration parameters
        if config_dict is None:
            config_dict = {}
        self.feat_drop = config_dict.get('feat_drop', 0.1)
        self.edge_feat_drop = config_dict.get('edge_feat_drop', 0.1)
        self.attn_drop = config_dict.get('attn_drop', 0.1)
        
        # Node feature transformation
        self.node_transform = nn.Linear(in_dim, out_dim, bias=use_bias).to(device)
        
        # Equivariant geometric feature processing
        self.coord_mlp = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, out_dim)
        ).to(device)
        
        # Distance encoding
        self.distance_expansion = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, out_dim)
        ).to(device)
        
        # Attention mechanism
        self.attention_mlp = nn.Sequential(
            nn.Linear(out_dim * 3 + edge_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, 1)
        ).to(device)
        
        # Message passing network
        self.message_mlp = nn.Sequential(
            nn.Linear(out_dim * 3 + edge_dim, out_dim * 2),
            nn.ReLU(),
            nn.Linear(out_dim * 2, out_dim)
        ).to(device)
        
        # Coordinate update network (equivariant)
        self.coord_update = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, 1)
        ).to(device)
        
        # Dropout layers
        self.feat_dropout = nn.Dropout(self.feat_drop)
        self.edge_dropout = nn.Dropout(self.edge_feat_drop)
        self.attn_dropout = nn.Dropout(self.attn_drop)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters"""
        gain = nn.init.calculate_gain('relu')
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=gain)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
    
    def forward(self, graph, node_feats, edge_feats=None, coordinates=None):
        """
        Forward pass
        Args:
            graph: DGL graph object
            node_feats: node features [N, in_dim]
            edge_feats: edge features [E, edge_dim]
            coordinates: node coordinates [N, 3]
        Returns:
            updated_feats: updated node features [N, out_dim]
            attention_weights: attention weights
            updated_coords: updated coordinates (if coordinates are provided)
        """
        with graph.local_scope():
            # Get edge indices
            src, dst = graph.edges()
            
            # Feature dropout
            node_feats = self.feat_dropout(node_feats)
            if edge_feats is not None:
                edge_feats = self.edge_dropout(edge_feats)
            
            # Node feature transformation
            h = self.node_transform(node_feats)
            
            # Compute geometric features (if coordinates are provided)
            coord_feats = None
            distance_feats = None
            relative_pos = None
            
            if coordinates is not None:
                # Compute relative position vectors
                relative_pos = coordinates[dst] - coordinates[src]  # [E, 3]
                
                # Compute distances
                distances = torch.norm(relative_pos, dim=1, keepdim=True)  # [E, 1]
                distance_feats = self.distance_expansion(distances)  # [E, out_dim]
                
                # Coordinate features (equivariant)
                coord_feats = self.coord_mlp(relative_pos)  # [E, out_dim]
            
            # Build edge input features
            edge_input_feats = []
            edge_input_feats.append(h[src])  # Source node features
            edge_input_feats.append(h[dst])  # Destination node features
            
            if distance_feats is not None:
                edge_input_feats.append(distance_feats)
            else:
                edge_input_feats.append(torch.zeros(len(src), self.out_dim).to(h.device))
            
            if edge_feats is not None:
                edge_input_feats.append(edge_feats)
            else:
                edge_input_feats.append(torch.zeros(len(src), self.edge_dim).to(h.device))
            
            edge_input = torch.cat(edge_input_feats, dim=1)  # [E, out_dim*3 + edge_dim]
            
            # Compute attention weights
            attention_logits = self.attention_mlp(edge_input)  # [E, 1]
            attention_weights = self.attn_dropout(F.softmax(attention_logits, dim=0))
            
            # Compute messages
            messages = self.message_mlp(edge_input)  # [E, out_dim]
            
            # Add geometric information
            if coord_feats is not None:
                messages = messages + coord_feats
            
            # Weight messages
            weighted_messages = messages * attention_weights
            
            # Aggregate messages
            graph.edata['m'] = weighted_messages
            graph.update_all(fn.copy_e('m', 'm'), fn.sum('m', 'h'))
            updated_feats = graph.ndata['h']
            
            # Residual connection
            if updated_feats.shape[1] == h.shape[1]:
                updated_feats = updated_feats + h
            
            # Coordinate update (equivariant)
            updated_coords = coordinates
            if coordinates is not None and relative_pos is not None:
                # Compute scalar coefficients for coordinate updates
                coord_weights = self.coord_update(messages)  # [E, 1]
                coord_weights = coord_weights * attention_weights
                
                # Equivariant coordinate update: use the direction of relative position vectors
                coord_updates = coord_weights * F.normalize(relative_pos, dim=1)  # [E, 3]
                
                # Aggregate coordinate updates
                coord_update_sum = torch.zeros_like(coordinates)
                coord_update_sum.index_add_(0, dst, coord_updates)
                
                updated_coords = coordinates + coord_update_sum * 0.1  # Small step size
            
            return updated_feats, attention_weights, updated_coords

class MultiHeadEGRETLayer(nn.Module):
    """Multi-head equivariant graph convolution layer"""
    def __init__(self, in_dim, out_dim, edge_dim, num_heads, use_bias=True, merge='avg', config_dict=None):
        super(MultiHeadEGRETLayer, self).__init__()
        self.num_heads = num_heads
        self.merge = merge
        self.out_dim = out_dim
        
        # Ensure the output dimension is divisible by the number of heads
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
        self.head_dim = out_dim // num_heads
        
        # Create multiple heads
        self.heads = nn.ModuleList()
        for _ in range(num_heads):
            self.heads.append(
                EGRETLayer(
                    in_dim=in_dim,
                    out_dim=self.head_dim if merge == 'cat' else out_dim,
                    edge_dim=edge_dim,
                    use_bias=use_bias,
                    config_dict=config_dict
                )
            )
        
        # Output projection layer (for both avg and cat merges)
        if merge == 'avg':
            self.output_proj = nn.Linear(out_dim, out_dim, bias=use_bias).to(device)
        elif merge == 'cat':
            self.output_proj = nn.Linear(out_dim, out_dim, bias=use_bias).to(device)
    
    def forward(self, graph, node_feats, edge_feats=None, coordinates=None):
        """
        Forward pass
        Args:
            graph: DGL graph object
            node_feats: node features [N, in_dim]
            edge_feats: edge features [E, edge_dim]
            coordinates: node coordinates [N, 3]
        Returns:
            output_feats: output node features [N, out_dim]
            attention_weights: list of attention weights for all heads
        """
        head_outputs = []
        head_attentions = []
        updated_coordinates = coordinates
        
        # Compute outputs for each head
        for i, head in enumerate(self.heads):
            head_out, head_attn, head_coords = head(graph, node_feats, edge_feats, coordinates)
            head_outputs.append(head_out)
            head_attentions.append(head_attn)
            
            # Use coordinate updates only from the first head (to avoid multiple updates)
            if i == 0:
                updated_coordinates = head_coords
        
        # Merge multi-head outputs
        if self.merge == 'cat':
            # Concatenate outputs from all heads
            output_feats = torch.cat(head_outputs, dim=1)
        elif self.merge == 'avg':
            # Average outputs from all heads
            output_feats = torch.stack(head_outputs, dim=0).mean(dim=0)
        else:
            raise ValueError(f"Unknown merge method: {self.merge}")
        
        # Apply output projection
        output_feats = self.output_proj(output_feats)
        
        return output_feats, head_attentions

# Import DGL functions
try:
    import dgl
    from dgl import function as fn
except ImportError:
    print("Warning: DGL not found. Please install DGL for full functionality.")
    # Provide a simple fallback implementation
    class fn:
        @staticmethod
        def copy_e(edge_feat, out_name):
            return lambda edges: {out_name: edges.data[edge_feat]}
        
        @staticmethod
        def sum(msg, out_name):
            return lambda nodes: {out_name: torch.sum(nodes.mailbox[msg], dim=1)}

# Configuration dictionary
config_dict = {
    'feat_drop': 0.2,
    'edge_feat_drop': 0.1,
    'attn_drop': 0.2
}
