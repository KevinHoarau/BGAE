import torch
from torch.nn import LayerNorm
from torch_geometric.nn import GAE, VGAE, GCNConv, ResGatedGraphConv, DeepGCNLayer

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, layers):
        super().__init__()
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, out_channels))
        
        for l in range(layers-1):
            self.convs.append(GCNConv(out_channels, out_channels))

    def forward(self, x, edge_index):
        
        for conv in self.convs:
            x = conv(x, edge_index).relu()

        return x