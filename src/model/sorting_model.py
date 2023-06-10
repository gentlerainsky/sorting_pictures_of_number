# This project is a Pytorch implementation of sorting algorithm representation learning.
# The implementation is a imitation from the baseline model of https://github.com/deepmind/clrs

from torch import nn
import torch
import torch_geometric
import torch_geometric.data as geom_data

from src.model.linear_encoder import FeatureEncoder
from src.model.gat import SortGAT
from src.model.linear_decoder import FeatureDecoder, postprocess
import numpy as np


class SortingModel(nn.Module):
    def __init__(
            self,
            gat_head,
            feature_encoded_dim,
            dropout,
            num_node
    ):
        super().__init__()
        self.hidden_dim = feature_encoded_dim
        self.gat_heads = gat_head
        self.dropout = dropout
        self.num_node = num_node
        self.encoder = FeatureEncoder(
            hidden_dim=self.hidden_dim
        ).to(torch.double)
        self.gat = SortGAT(
            in_channel=self.hidden_dim,
            gat_head=gat_head,
            hidden_dim=self.hidden_dim,
            dropout=dropout
        ).to(torch.double)
        self.decoder = FeatureDecoder(
            node_hidden_dim=self.hidden_dim * 3,
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim
        ).to(torch.double)

    def forward(self, x, num_execution_steps):
        input_key = x.input.key.unsqueeze(-1).double()
        input_pos = x.input.pos.unsqueeze(-1).double()

        model_results = []
        # FIRST PASS
        node_features, edge_features = self.encoder(
            key=input_key,
            pos=input_pos,
            i=x.hint.i_list[0].unsqueeze(-1).double(),
            j=x.hint.j_list[0].unsqueeze(-1).double(),
            pred_h_edge_features=x.hint.pred_h_edge_features[0].double()
        )
        sample = geom_data.Data(
            x=node_features,
            edge_index=x.hint.pred_h_edge_indices[0],
            edge_attr=edge_features
        )
        output = self.gat(sample)
        previous_hidden = torch.zeros_like(output)
        h_t = torch.concat([output, previous_hidden, node_features], axis=-1)
        previous_hidden = output
        decoded_features = self.decoder(
            node_features=h_t,
            edge_features=torch_geometric.utils.to_dense_adj(
                edge_index=sample.edge_index,
                edge_attr=sample.edge_attr
            )
        )
        processed_decoded_features = postprocess(decoded_features)
        
        model_results.append(processed_decoded_features)

        # as we have already run the first step.
        steps = np.max([1, num_execution_steps - 1])
        for _ in range(steps):
            feature_mat = processed_decoded_features.pred_h.squeeze(0)
            adj_mat = ((feature_mat + feature_mat.transpose(1, 0)) > 0.5).double()
            iden = torch.eye(self.num_node).to(adj_mat.device)
            adj_mat_i = adj_mat + iden
            adj_mat_i = (adj_mat_i > 0).long()
            edge_index = adj_mat_i.nonzero().t().contiguous()
            edge_features = feature_mat[adj_mat_i > 0].view(edge_index.shape[1], -1)

            node_features, edge_features = self.encoder(
                key=input_key,
                pos=input_pos,
                i=processed_decoded_features.i.unsqueeze(-1),
                j=processed_decoded_features.j.unsqueeze(-1),
                pred_h_edge_features=edge_features
            )
            sample = geom_data.Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_features
            )
            output = self.gat(sample)
            h_t = torch.concat([output, previous_hidden, node_features], axis=-1)
            previous_hidden = output
            decoded_features = self.decoder(
                node_features=h_t,
                edge_features=torch_geometric.utils.to_dense_adj(
                    edge_index=sample.edge_index,
                    edge_attr=sample.edge_attr
                )
            )
            processed_decoded_features = postprocess(decoded_features)
            model_results.append(processed_decoded_features)
        return model_results
