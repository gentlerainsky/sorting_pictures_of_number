from pytorch_lightning import LightningModule
from src.dataset.number_sorting import NumberSortingDataset, SortingTarget, SortingSamples, SortingHint, SortingInput
from functools import partial
import torch.utils.data as data
from src.model.sorting_model import SortingModel
from src.loss_function import accumulate_loss
from src.evaluation import evaluate, fuse_pointer_and_mask
from src.util import pred_list_to_permutation_matrix
import torch


class LitSortingModel(LightningModule):
    def __init__(
        self,
        gat_head=1,
        feature_encoded_dim=128,
        dropout=0.0,
        num_node=4,
        num_train=1000,
        num_val=100,
        num_test=100,
        learning_rate=0.001,
        batch_size=25,
        hint_loss=True
    ):
        super().__init__()
        self.model = SortingModel(
            gat_head=gat_head,
            feature_encoded_dim=feature_encoded_dim,
            dropout=dropout,
            num_node=num_node
        ).to(self.device)
        self.dataset_generator = partial(
            NumberSortingDataset,
            device=self.device,
            num_items=4,
            range_min=-10000000,
            range_max=10000000
        )
        self.loss_criterion = accumulate_loss
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
        self.train_loss_log = []
        self.val_loss_log = []
        self.validation_step_outputs = []
        self.training_step_outputs = []
        self.hint_loss = hint_loss
        self.automatic_optimization = False

    def forward(self, sample, num_step):
        results = self.model(sample, num_step)
        pred = fuse_pointer_and_mask(
            pred_mask=results[-1].pred_mask,
            pred=results[-1].pred
        )
        permutation_matrix = pred_list_to_permutation_matrix(pred)
        return permutation_matrix

    def forward_for_loss(self, sample, num_step):
        results = self.model(sample, num_step)
        loss = self.loss_criterion(sample, results, use_hint_loss=False)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        sch = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size = 1, gamma = 0.9
        )
        #learning rate scheduler
        return {
            "optimizer":optimizer,
            "lr_scheduler" : {
                "scheduler" : sch,
                "monitor" : "train_loss",
            }
        }

    def remove_batch_dim(self, sample):
        target = SortingTarget(
            original_pred=sample.target.original_pred.squeeze(0).to(self.device),
            pred=sample.target.pred.squeeze(0).to(self.device),
            mask=sample.target.mask.squeeze(0).to(self.device)
        )
        hint = SortingHint(
            num_step=sample.hint.num_step.squeeze(0).to(self.device),
            pred_h=[pred_h.squeeze(0).to(self.device) for pred_h in sample.hint.pred_h],
            pred_h_edge_indices=[item.squeeze(0).to(self.device) for item in sample.hint.pred_h_edge_indices],
            pred_h_edge_features=[item.squeeze(0).to(self.device) for item in sample.hint.pred_h_edge_features],
            i_list=sample.hint.i_list.squeeze(0).to(self.device),
            j_list=sample.hint.j_list.squeeze(0).to(self.device),
        )
        input = SortingInput(
            key=sample.input.key.squeeze(0).to(self.device),
            pos=sample.input.pos.squeeze(0).to(self.device),
            input=sample.input.input.squeeze(0).to(self.device),
        )
        new_sample = SortingSamples(
            input=input, hint=hint, target=target
        )
        return new_sample

    def training_step(self, batch, sample_idx):
        sample = self.remove_batch_dim(batch)
        results = self.model(sample, len(sample.hint.pred_h) - 1)
        loss = self.loss_criterion(sample, results, use_hint_loss=self.hint_loss)
        self.training_step_outputs.append(loss)
        self.manual_backward(loss)
        # accumulate gradients of N batches
        if (sample_idx + 1) % self.batch_size == 0:
            opt = self.optimizers()
            self.clip_gradients(opt, gradient_clip_val=1, gradient_clip_algorithm="norm")
            opt.step()
            opt.zero_grad()
        return loss

    def validation_step(self, batch, sample_idx):
        sample = self.remove_batch_dim(batch)
        results = self.model(sample, len(sample.hint.pred_h) - 1)
        loss = self.loss_criterion(sample, results, use_hint_loss=self.hint_loss)
        accuracy = evaluate(
            truth_pred=sample.target.pred,
            truth_pred_mask=sample.target.mask,
            pred_mask=results[-1].pred_mask,
            pred=results[-1].pred
        )
        self.validation_step_outputs.append((loss, accuracy))
        self.log('val_loss', loss)
        return loss, accuracy

    def on_validation_epoch_end(self):
        train_outs = self.training_step_outputs
        val_outs = self.validation_step_outputs
        losses = []
        accuracies = []
        for val_out in val_outs:
            loss, accuracy = val_out
            losses.append(loss)
            accuracies.append(accuracy)
        results = {
            "loss": torch.mean(torch.tensor(losses, dtype=torch.float)).item(),
            "accuracy": torch.mean(torch.tensor(accuracies, dtype=torch.float)).item(),
        }
        self.val_loss_log.append(results)
        self.train_loss_log.append(torch.mean(torch.tensor(train_outs)).item())
        print(f"training loss {len(train_outs)} samples: {self.train_loss_log[-1]}")
        print(
            f"Validation Set {len(val_outs)} samples (@epoch:{self.current_epoch}): "
            + f"loss={self.val_loss_log[-1]['loss']}, "
            + f"accuracy={self.val_loss_log[-1]['accuracy']}"
        )
        self.validation_step_outputs.clear()
        self.training_step_outputs.clear()
        
        return results['loss'], results['accuracy']

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = self.dataset_generator(self.num_train)
            self.val_dataset = self.dataset_generator(self.num_val)
        if stage == "test" or stage is None:
            self.test_dataset = self.dataset_generator(self.num_test)

    def train_dataloader(self):
        return data.DataLoader(
            self.train_dataset, batch_size=1, shuffle=True, drop_last=True, pin_memory=True
        )

    def val_dataloader(self):
        return data.DataLoader(self.val_dataset, batch_size=1)
    
