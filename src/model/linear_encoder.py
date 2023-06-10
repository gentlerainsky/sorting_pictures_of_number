# This project is a Pytorch implementation of sorting algorithm representation learning.
# The implementation is a imitation from the baseline model of https://github.com/deepmind/clrs

from torch import nn

class FeatureEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.pos_encoder = nn.Linear(1, hidden_dim)
        self.key_encoder = nn.Linear(1, hidden_dim)
        self.i_encoder = nn.Linear(1, hidden_dim)
        self.j_encoder = nn.Linear(1, hidden_dim)
        self.pred_h_encoder = nn.Linear(1, hidden_dim)

    def forward(self, key, pos, pred_h_edge_features, i, j):
        node_features = self.pos_encoder(pos)
        node_features += self.key_encoder(key)
        node_features += self.i_encoder(i)
        node_features += self.j_encoder(j)
        edge_features = self.pred_h_encoder(pred_h_edge_features)
        return node_features, edge_features
