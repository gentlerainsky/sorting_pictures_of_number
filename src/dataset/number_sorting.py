# This project is a Pytorch implementation of sorting algorithm representation learning.
# The implementation is a imitation from the baseline model of https://github.com/deepmind/clrs

import torch
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
from src.dataset.number_generator import AlgorithmName, generate_data
from collections import namedtuple

SortingSamples = namedtuple('SortingSample', ['input', 'hint', 'target'])
SortingInput = namedtuple('SortingInput', ['key', 'input', 'pos'])
SortingHint = namedtuple('SortingFeature', ['num_step', 'pred_h', 'pred_h_edge_indices', 'pred_h_edge_features', 'i_list', 'j_list'])
SortingTarget = namedtuple('SortingTarget', ['pred', 'mask', 'original_pred'])

def convert_pred_h(num_node, _pred_h):
    pred_h = _pred_h
    if not torch.is_tensor(_pred_h):
        pred_h = torch.tensor(_pred_h)
    adj_mat = F.one_hot(pred_h, num_classes=num_node)
    adj_mat_i = adj_mat + torch.eye(num_node)
    adj_mat_i = (adj_mat_i > 0).long()
    edge_index = adj_mat_i.nonzero().t().contiguous()
    edge_features = adj_mat[adj_mat_i > 0].view(edge_index.shape[1], -1)
    return edge_index, edge_features

def convert_target(pointers, num_items):
    """convert target into cyclic predecessor + first node mask"""
    pointers_one_hot = F.one_hot(pointers, num_items)
    last = pointers_one_hot.sum(-2).argmin()
    first = pointers_one_hot.diagonal().argmax()
    mask = F.one_hot(first, num_items)
    pointers_one_hot += mask[..., None] * F.one_hot(last, num_items)
    pointers_one_hot -= mask[..., None] * mask
    return pointers_one_hot, mask

class NumberSortingDataset(data.Dataset):
    def __init__(self, n, num_items, range_min, range_max, random_state=None, device='cpu'):
        super().__init__()
        if random_state is not None:
            np.random.seed(random_state)
        self.range_min = range_min
        self.range_max = range_max
        self.num_items = num_items
        self.results, self.data = generate_data(AlgorithmName.INSERTION_SORT, n, num_items, range_min, range_max)
        self.size = n
        self.processed_data = []
        self.device = device
        self.prepare_data()

    def __len__(self):
        return self.size

    def convert_pred_h(self, num_node, pred_h):
        """convert pred_h to edge_index and edge_features"""
        return convert_pred_h(num_node, pred_h)

    def convert_target(self, pointers):
        """convert target into cyclic predecessor + first node mask"""
        return convert_target(pointers, self.num_items)

    def prepare_data(self):
        for idx in range(self.size):
            data_input = self.data[idx]['input']['node']
            num_node = data_input['key']['data'].shape[0]
            data_hint = self.data[idx]['hint']
            data_output = self.data[idx]['output']
            inputs = {
                'key': (
                torch.tensor(data_input['key']['data'])
                    / (torch.tensor(self.range_max) - torch.tensor(self.range_min))
                ).to(self.device),
                'pos': torch.tensor(data_input['pos']['data']).to(self.device),
                'input': torch.tensor(data_input['key']['data']).to(self.device)
            }
            sorting_inputs = SortingInput(**inputs)

            hints = {
                'num_step': len(data_hint['node']['pred_h']['data']),
                'pred_h': [],
                'pred_h_edge_indices': [],
                'pred_h_edge_features': [],
                'i_list': torch.tensor(data_hint['node']['i']['data']).to(self.device),
                'j_list': torch.tensor(data_hint['node']['j']['data']).to(self.device)
            }

            for pred_h in data_hint['node']['pred_h']['data']:
                edge_index, edge_feature = self.convert_pred_h(num_node, pred_h)
                hints['pred_h_edge_indices'].append(edge_index)
                hints['pred_h_edge_features'].append(edge_feature)
                hints['pred_h'].append(torch.tensor(pred_h))
            sorting_hints = SortingHint(**hints)
            pred, mask = self.convert_target(
                torch.tensor(data_output['node']['pred']['data']).to(self.device)
            )
            targets = SortingTarget(
                original_pred=torch.tensor(data_output['node']['pred']['data']).to(self.device),
                pred=pred,
                mask=mask
            )
            self.processed_data.append(SortingSamples(sorting_inputs, sorting_hints, targets))

    def __getitem__(self, idx):        
        return self.processed_data[idx]
