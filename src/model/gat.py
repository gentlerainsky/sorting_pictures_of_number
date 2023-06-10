# This project is a Pytorch implementation of sorting algorithm representation learning.
# The implementation is a imitation from the baseline model of https://github.com/deepmind/clrs

from torch import nn
import torch_geometric.nn as geom_nn

class SortGAT(nn.Module):
    def __init__(
            self,
            in_channel,
            gat_head,
            hidden_dim,
            dropout
        ):
        super().__init__()
        self.in_channel = in_channel
        self.hidden_dim = hidden_dim
        self.gat_heads = gat_head
        self.dropout = dropout
        self.gat = geom_nn.GAT(
            num_layers=1,
            in_channels=self.in_channel,
            hidden_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            heads=self.gat_heads,
            dropout=self.dropout,
        )

    def forward(self, data):
        x = self.gat(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr
        )
        return x
