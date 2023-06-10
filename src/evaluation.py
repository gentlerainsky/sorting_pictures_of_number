# This project is a Pytorch implementation of sorting algorithm representation learning.
# The implementation is a imitation from the baseline model of https://github.com/deepmind/clrs

import numpy as np


def fuse_pointer_and_mask(pred, pred_mask):
    np_pred = pred.cpu()
    np_pred_mask = pred_mask.cpu()
    data = np.where(
        np_pred_mask.data > 0.5,
        np.arange(np_pred.data.shape[-1]),
        np.argmax(np_pred.data, axis=-1)
    )
    return data

def calculate_accuracy(pred, truth):
    return np.mean((pred == truth) * 1.0)

def evaluate(truth_pred, pred, truth_pred_mask, pred_mask):
    truth = fuse_pointer_and_mask(truth_pred, truth_pred_mask)
    pred = fuse_pointer_and_mask(pred, pred_mask)
    measure = calculate_accuracy(pred, truth)
    return measure
