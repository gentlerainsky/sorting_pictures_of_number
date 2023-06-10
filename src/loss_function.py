# This project is a Pytorch implementation of sorting algorithm representation learning.
# The implementation is a imitation from the baseline model of https://github.com/deepmind/clrs

import torch
import torch.nn.functional as F


def pred_mask_loss(truth, pred):
    loss = (
        -torch.sum(truth * F.log_softmax(pred, dim=-1)) /
        torch.sum(truth.data == 1)
    )
    return loss


def pred_loss(truth, pred):
    loss = torch.mean(-torch.sum(truth.data * pred, dim=-1))
    return loss


def pred_h_loss(num_node, truth, pred):
    loss = torch.mean(-torch.sum(
        F.one_hot(truth, num_node) * F.log_softmax(pred, dim=-1),
        dim=-1
    ))
    return loss

def ij_loss(truth, pred):
    loss = -torch.sum(
        truth * F.log_softmax(pred, dim=-1), dim=-1,
        keepdims=True
    )
    return loss


def accumulate_loss(truth, preds, use_hint_loss):
    # output
    output_loss = pred_mask_loss(truth.target.mask, preds[-1].pred_mask)
    output_loss = output_loss + pred_loss(truth.target.pred, preds[-1].pred)
    # hints
    num_item = truth.input.key.shape[0]
    hint_loss_pred_h = []
    hint_loss_i = []
    hint_loss_j = []

    loss = output_loss
    if use_hint_loss:
        for step in range(len(preds)):
            # First hint is the input
            hint_step = step + 1
            tmp_loss = pred_h_loss(num_item, truth.hint.pred_h[hint_step], preds[step].pred_h)
            hint_loss_pred_h.append(tmp_loss)
            tmp_loss = ij_loss(truth.hint.i_list[hint_step], preds[step].i)
            hint_loss_i.append(tmp_loss)
            tmp_loss = ij_loss(truth.hint.j_list[hint_step], preds[step].j)
            hint_loss_j.append(tmp_loss)
        loss = loss + (
            torch.mean(torch.tensor(hint_loss_pred_h))
            + torch.mean(torch.tensor(hint_loss_i))
            + torch.mean(torch.tensor(hint_loss_j))
        )
    return loss
