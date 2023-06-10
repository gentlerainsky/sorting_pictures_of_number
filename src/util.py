# This project is a Pytorch implementation of sorting algorithm representation learning.
# The implementation is a imitation from the baseline model of https://github.com/deepmind/clrs

import numpy as np
import torch
import torch.nn.functional as F


def pred_list_to_permutation_matrix(pred_list):
    start = F.one_hot(torch.tensor(pred_list)).sum(-2).argmin().numpy().tolist()
    is_end = False
    curr = start
    l = [curr]
    previous = None
    while not is_end:
        curr = pred_list[curr]
        if curr == previous:
            break
        l.append(curr)
        previous = curr
    permute_matrix = torch.flip(F.one_hot(torch.tensor(np.array(l))), dims=[0])
    return permute_matrix
