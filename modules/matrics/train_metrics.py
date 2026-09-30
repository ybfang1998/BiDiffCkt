import torch
from torch import Tensor
import torch.nn as nn
import wandb
from modules.matrics.abstract_metrics import CrossEntropyMetric, BinaryCrossEntropyMetric

class TrainLossDiscrete(nn.Module):
    """ Train with Cross entropy"""
    def __init__(self, lambda_train, edge_loss_fn, edge_activation, pos_weight=None, device=None):
        super().__init__()
        self.node_loss = CrossEntropyMetric()
        self.net_loss = CrossEntropyMetric()
        if edge_loss_fn == 'CE':
            self.edge_loss = CrossEntropyMetric()
        elif edge_loss_fn == 'BCE':
            self.edge_loss = BinaryCrossEntropyMetric(pos_weight)
        # Keep torchmetrics metrics on the same device as the model/inputs.
        # (In Lightning the whole module is moved automatically, but this prevents mismatches
        # when `device` is provided and metrics are used early.)
        if device is not None:
            self.node_loss.to(device)
            self.net_loss.to(device)
            self.edge_loss.to(device)
        self.lambda_train = lambda_train
        self.edge_loss_fn = edge_loss_fn
        self.edge_activation = edge_activation
        self.pos_weight = pos_weight
    def forward(self, masked_pred_U, masked_pred_V, masked_pred_E, true_U, true_V, true_E, log: bool, U_mask=None, V_mask=None):
        """ Compute train metrics
        masked_pred_U : tensor -- (bs, n_u, dx)
        masked_pred_V : tensor -- (bs, n_v, de)
        masked_pred_E : tensor -- (bs, n_u, n_n, de)
        pred_y : tensor -- (bs, )
        true_U : tensor -- (bs, n_u, 10)
        true_V : tensor -- (bs, n_v, 10)
        true_E : tensor -- (bs, n_u, n_v, n_edge_types)
        log : boolean. """

        true_U = torch.reshape(true_U, (-1, true_U.size(-1)))  # (bs * n_u, 10)
        true_V = torch.reshape(true_V, (-1, true_V.size(-1)))  # (bs * n_v, 10)
        true_E = torch.reshape(true_E, (-1, true_E.size(-1)))  # (bs * n_u * n_v, n_edge_types)
        masked_pred_U = torch.reshape(masked_pred_U, (-1, masked_pred_U.size(-1)))  # (bs * n_u, 10)
        masked_pred_V = torch.reshape(masked_pred_V, (-1, masked_pred_V.size(-1)))  # (bs * n_v, 10)
        masked_pred_E = torch.reshape(masked_pred_E, (-1, masked_pred_E.size(-1)))  # (bs * n_u * n_v, n_edge_types)

        # Remove masked rows
        mask_U_flatten = torch.reshape(U_mask, (-1,))
        mask_V_flatten = torch.reshape(V_mask, (-1,))
        if self.edge_loss_fn == 'CE':
            mask_E_flatten = (true_E != 0.).any(dim=-1)
        elif self.edge_loss_fn == 'BCE':
            mask_E = U_mask.unsqueeze(2) * V_mask.unsqueeze(1) 
            mask_E_flatten = torch.reshape(mask_E, (-1,))

        flat_true_U = true_U[mask_U_flatten, :]
        flat_true_V = true_V[mask_V_flatten, :]
        flat_pred_U = masked_pred_U[mask_U_flatten, :]
        flat_pred_V = masked_pred_V[mask_V_flatten, :]

        flat_true_E = true_E[mask_E_flatten, :]
        flat_pred_E = masked_pred_E[mask_E_flatten, :]

        loss_X = self.node_loss(flat_pred_U, flat_true_U) if true_U.numel() > 0 else 0.0
        loss_net = self.net_loss(flat_pred_V, flat_true_V) if true_V.numel() > 0 else 0.0
        loss_E = self.edge_loss(flat_pred_E, flat_true_E) if true_E.numel() > 0 else 0.0

        loss_total = self.lambda_train[0] * loss_X + self.lambda_train[1] * loss_net + self.lambda_train[2] * loss_E

        if log:
            to_log = {"train_loss/batch_CE": (loss_total).detach(),
                      "train_loss/X_CE": self.node_loss.compute() if true_U.numel() > 0 else -1,
                      "train_loss/Net_CE": self.net_loss.compute() if true_V.numel() > 0 else -1,
                      "train_loss/E_CE": self.edge_loss.compute() if true_E.numel() > 0 else -1}

            if wandb.run:
                wandb.log(to_log, commit=True)

        return loss_total

    def reset(self):
        for metric in [self.node_loss, self.net_loss, self.edge_loss]:
            metric.reset()

    def log_epoch_metrics(self):
        epoch_node_loss = self.node_loss.compute() if self.node_loss.total_samples > 0 else -1
        epoch_net_loss = self.net_loss.compute() if self.net_loss.total_samples > 0 else -1
        epoch_edge_loss = self.edge_loss.compute() if self.edge_loss.total_samples > 0 else -1

        to_log = {"train_epoch/total_CE": epoch_node_loss + epoch_net_loss + epoch_edge_loss,
                  "train_epoch/X_CE": epoch_node_loss,
                  "train_epoch/Net_CE": epoch_net_loss,
                  "train_epoch/E_CE": epoch_edge_loss}
        if wandb.run:
            wandb.log(to_log, commit=True)

        return to_log
