# This project is a Pytorch implementation of sorting algorithm representation learning.
# The implementation is a imitation from the baseline model of https://github.com/deepmind/clrs

from torch import nn
from torch_geometric.data import Data
import torch.nn.functional as F
import torch
from collections import namedtuple


DecoderOutput = namedtuple('DecoderOutput', ['i', 'j', 'pred', 'pred_h', 'pred_mask'])


def log_sinkhorn(x, steps, temperature, zero_diagonal, with_noise):
    if with_noise:
        # Add standard Gumbel noise (see https://arxiv.org/abs/1802.08665)
        noise = -torch.log(-torch.log(torch.rand(x.shape) + 1e-12) + 1e-12).to(x.device)
        x = x + noise
    x = x / temperature
    if zero_diagonal:
        iden = torch.eye(x.shape[-1]).to(x.device)
        x = x - 1e6 * iden
    for _ in range(steps):
        x = F.log_softmax(x, dim=-1)
        x = F.log_softmax(x, dim=-2)
    return x


def postprocess(decoder_output):
    # i mask_one
    i = torch.softmax(decoder_output.i, dim=-1)
    # j mask_one
    j = torch.softmax(decoder_output.j, dim=-1)
    # pred_mask mask_one
    pred_mask = torch.softmax(decoder_output.pred_mask, dim=-1)
    # pred_h pointer
    pred_h = F.softmax(decoder_output.pred_h, dim=-1)
    
    # pred permutation_pointer
    pred = log_sinkhorn(
        decoder_output.pred,
        steps=25,
        temperature=0.1,
        zero_diagonal=True,
        with_noise=False
    )

    pred = torch.exp(pred)
    return DecoderOutput(**{
        'i': i,
        'j': j,
        'pred_mask': pred_mask,
        'pred_h': pred_h,
        'pred': pred
    })



class FeatureDecoder(nn.Module):
    def __init__(self, node_hidden_dim, input_dim, hidden_dim):
        super().__init__()
        self.pred_decoder = nn.ModuleList([
            nn.Linear(node_hidden_dim, hidden_dim),
            nn.Linear(node_hidden_dim, hidden_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(input_dim, 1)
        ])
        self.pred_h_decoder = nn.ModuleList([
            nn.Linear(node_hidden_dim, hidden_dim),
            nn.Linear(node_hidden_dim, hidden_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(input_dim, 1)
        ])
        self.pred_mask_decoder = nn.Linear(node_hidden_dim, 1)
        self.i_decoder = nn.Linear(node_hidden_dim, 1)
        self.j_decoder = nn.Linear(node_hidden_dim, 1)

    def decode_pred_h(self, decoders, h_t, edge_fts):
        p_1 = decoders[0](h_t)
        p_2 = decoders[1](h_t)
        p_3 = decoders[2](edge_fts)

        p_e = p_2.unsqueeze(-2) + p_3
        p_m = torch.maximum(
            p_1.unsqueeze(-2),
            p_e.permute((0, 2, 1, 3))
        )

        preds = decoders[3](p_m).squeeze(-1)
        return preds


    def decode_pred(self, decoders, h_t, edge_fts):
        x = self.decode_pred_h(decoders, h_t, edge_fts)
        if self.training:
            preds = log_sinkhorn(
                x=x, steps=10, temperature=0.1,
                zero_diagonal=True, with_noise=True)
        else:
            preds = log_sinkhorn(
                x=x, steps=10, temperature=0.1,
                zero_diagonal=True, with_noise=False)
        return preds

    def forward(self, node_features, edge_features):
        i = self.i_decoder(node_features).squeeze(-1)
        j = self.j_decoder(node_features).squeeze(-1)
        pred_mask = self.pred_mask_decoder(node_features).squeeze(-1)
        pred = self.decode_pred(self.pred_decoder, node_features, edge_features)
        pred_h = self.decode_pred_h(self.pred_h_decoder, node_features, edge_features)
        return DecoderOutput(i=i, j=j, pred_mask=pred_mask, pred_h=pred_h, pred=pred)