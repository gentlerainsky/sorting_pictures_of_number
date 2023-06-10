from pytorch_lightning import LightningModule
from src.model.multidigit_mnist_conv import MultiDigitMNISTConv
from src.evaluation import fuse_pointer_and_mask, calculate_accuracy
import torch
import clrs
from src.dataset.number_sorting import (
    SortingTarget, SortingSamples, SortingHint, SortingInput, convert_pred_h, convert_target
)
import numpy as np


class LitMultiDigitMNISTSortingModel(LightningModule):
    def __init__(
        self,
        pretrained_sorting_model,
        num_digits = 4,
        num_execution_step = 4,
        learning_rate = 0.001,
        batch_size = 25,
    ):
        super().__init__()
        
        self.pretrained_sorting_model = pretrained_sorting_model
        for p in self.pretrained_sorting_model.parameters():
            p.requires_grad = False

        self.cnn_model = MultiDigitMNISTConv(num_digits)
        self.num_execution_step = num_execution_step
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.train_loss_log = []
        self.val_rank_log = []
        self.validation_step_outputs = []
        self.training_step_outputs = []
        self.automatic_optimization = False

    def forward(self, x):
        pass

    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.cnn_model.parameters(), lr=self.learning_rate)
    #     sch = torch.optim.lr_scheduler.StepLR(
    #         optimizer, step_size = 1, gamma = 0.75
    #     )
    #     #learning rate scheduler
    #     return {
    #         "optimizer":optimizer,
    #         "lr_scheduler" : {
    #             "scheduler" : sch,
    #             "monitor" : "train_loss",
                
    #         }
    #     }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.cnn_model.parameters(), lr=self.learning_rate)
        return optimizer

    def prepare_sample_and_groundtruth(self, ground_truth_sequence, input_sequence):
        ## FOR GROUNDTRUTH PERMUTATION
        # We can generate the ground truth without knowing the actual value of the sequence.
        # But I'm too lazy so I'm just use the CLRS to generate the ground truth
        # sorted predecessor pointer using the CLRS.
        clrs_hint = clrs._src.algorithms.sorting.insertion_sort(ground_truth_sequence.detach().numpy())
        original_pred = clrs_hint[1]['output']['node']['pred']['data']
        pred, mask = convert_target(torch.tensor(original_pred), 4)

        target = SortingTarget(
            original_pred=original_pred,
            pred=pred,
            mask=mask
        )

        ## FOR INPUT FEATURES
        # We need to preprocess the key to be a value within [0 - 1].
        input = SortingInput(
            input=input_sequence,
            pos=torch.tensor(np.arange(input_sequence.shape[0]) / input_sequence.shape[0]),
            key=(
                (input_sequence - input_sequence.min())
                / (input_sequence.max() - input_sequence.min())
            ),
        )

        ## PREPARE INITIAL STATE FOR HINTS
        # This only contain the initial state of each hint.
        # Note We don't use hint for calculating loss so we don't need to fill any other detail here.
        initial_pointer = ([0] * 2) + list(range(int(np.max([0, input_sequence.shape[0] - 2]))))
        initial_i = [1] + [0] * int(np.max([0, input_sequence.shape[0] - 1]))
        initial_j = [1] + [0] * int(np.max([0, input_sequence.shape[0] - 1]))
        edge_indices, edge_features = convert_pred_h(
            torch.tensor(input_sequence.shape[0]),
            torch.tensor(initial_pointer)
        )
        hint = SortingHint(
            num_step=4,
            pred_h=[torch.tensor(initial_pointer)],
            pred_h_edge_indices=[edge_indices],
            pred_h_edge_features=[edge_features],
            i_list=[torch.tensor(initial_i)],
            j_list=[torch.tensor(initial_j)],
        )

        new_sample = SortingSamples(
            input=input, hint=hint, target=target
        )
        return new_sample

    def remove_batch_dim(self, sample):
        image, ground_truth = sample
        image = image[0]
        ground_truth = ground_truth[0]
        return image, ground_truth

    def training_step(self, batch, sample_idx):
        image, ground_truth = self.remove_batch_dim(batch)
        encoded_image = torch.flatten(self.cnn_model(image))
        sample = self.prepare_sample_and_groundtruth(ground_truth, encoded_image)
        loss = self.pretrained_sorting_model.forward_for_loss(sample, self.num_execution_step)
        self.training_step_outputs.append(loss.item())
        self.manual_backward(loss)
        # accumulate gradients of N batches
        if (sample_idx + 1) % self.batch_size == 0:
            opt = self.optimizers()
            self.clip_gradients(opt, gradient_clip_val=1, gradient_clip_algorithm="norm")
            opt.step()
            opt.zero_grad()
        return loss

    def validation_step(self, batch, sample_idx):
        image, ground_truth = self.remove_batch_dim(batch)
        # print('image', image.shape)
        encoded_image = torch.flatten(self.cnn_model(image))
        # print('encoded_image', encoded_image.shape)
        sample = self.prepare_sample_and_groundtruth(ground_truth, encoded_image)
        loss = self.pretrained_sorting_model.forward_for_loss(sample, self.num_execution_step)
        gt = ground_truth.detach().numpy()
        accuracy = calculate_accuracy(
            np.argsort(encoded_image.detach().numpy()), np.argsort(gt)
        )
        self.validation_step_outputs.append((loss.item(), accuracy))
        self.log('val_loss', loss)
        return loss, accuracy

    def on_validation_epoch_end(self):
        outs = self.validation_step_outputs
        losses = []
        accuracies = []
        for out in outs:
            loss, accuracy = out
            losses.append(loss)
            accuracies.append(accuracy)
        results = {
            "loss": torch.mean(torch.tensor(losses, dtype=torch.float)).item(),
            "accuracy": torch.mean(torch.tensor(accuracies, dtype=torch.float)).item(),
        }
        self.val_rank_log.append(results)

        outs = self.training_step_outputs
        mean_loss = torch.mean(torch.tensor(outs)).item()
        self.train_loss_log.append(mean_loss)
        self.training_step_outputs.clear()
        avg_train_loss = self.train_loss_log[-1]
        print(
            f"Train Set Loss = {avg_train_loss} / "
            + f"Validation Set (@epoch:{self.current_epoch}): "
            + f"loss={self.val_rank_log[-1]['loss']}, "
            + f"accuracy={self.val_rank_log[-1]['accuracy']}"
        )
        self.validation_step_outputs.clear()
        return results['loss'], results['accuracy']
