import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.diffusion import diffusion_utils
from modules.diffusion.transformer import GraphTransformer
from modules.diffusion.digress_transformer import GraphTransformer as DigressGraphTransformer
import utils
from modules.matrics.train_metrics import TrainLossDiscrete

import pytorch_lightning as pl
import time
import os

class DiscreteDiffusion(pl.LightningModule):
    def __init__(self,
                 cfg,
                 marginal_X,
                 marginal_E,
                 node_dist,
                 log_every_steps,
                 pos_weight=None):
        super().__init__()

        self.cfg = cfg
        self.model_name = cfg.experiments.general.model_name
        self.U_marginal = marginal_X[0]
        self.V_marginal = marginal_X[1]
        self.E_marginal = marginal_E
        self.T = cfg.model.T

        self.output_dims = cfg.datasets.BiDiffckt.output_dims
        self.input_dims = cfg.datasets.BiDiffckt.input_dims

        self.conditional = cfg.model.conditional

        # if not conditional, only t as condition
        if self.conditional:
            self.input_dims['y'] += 1
        else:
            self.input_dims['y'] = 1

        self.U_dist = node_dist[0]
        self.V_dist = node_dist[1]
        self.log_every_steps = log_every_steps

        self.noise_schedule = PredefinedNoiseScheduleDiscrete('cosine',
                                                              timesteps=self.T)
        self.transition_model = MarginalUniformTransition(self.U_marginal, self.V_marginal, self.E_marginal)


        cfg_graph_transformer = cfg.model.GraphTransformer
        graph_transformer_type = cfg.experiments.general.get('transformer_type', 'BiDiffCkt')

        if graph_transformer_type == 'digress':
            self.input_dims = cfg.datasets.Digress.input_dims
            self.input_dims['y'] = 1
            cfg_graph_transformer = cfg.model.DigressGraphTransformer
            self.output_dims = cfg.datasets.Digress.output_dims
            self.model = DigressGraphTransformer(cfg_graph_transformer.n_layers,
                                                 self.input_dims,
                                                 cfg_graph_transformer.hidden_mlp_dims,
                                                 cfg_graph_transformer.hidden_dims,
                                                 self.output_dims,
                                                 act_fn_in=nn.ReLU(),
                                                 act_fn_out=nn.ReLU())
        else:
            self.model = GraphTransformer(cfg_graph_transformer.n_layers,
                                        self.input_dims,
                                        cfg_graph_transformer.hidden_mlp_dims,
                                        cfg_graph_transformer.hidden_dims,
                                        self.output_dims,
                                        act_fn_in=nn.ReLU(),
                                        act_fn_out=nn.ReLU())

        # Loss and Metrics
        self.edge_loss_fn = cfg.experiments.train.edge_loss
        self.edge_activation = cfg.experiments.train.edge_activation
        if pos_weight is not None:
            self.pos_weight = utils.to_tf_device(pos_weight, self.device)
        else:
            self.pos_weight = None
        self.train_loss = TrainLossDiscrete(cfg.experiments.train.lambda_train, self.edge_loss_fn, self.edge_activation, pos_weight=self.pos_weight, device=self.device)
        self.save_hyperparameters()

    def training_step(self, data, i):
        if self.edge_loss_fn == 'CE':
            data = {k: utils.to_tf_device(v, self.device, dtype=torch.float32) for k, v in data.items()}
        elif self.edge_loss_fn == 'BCE':
            data = {k: utils.to_tf_device(v, self.device, dtype=torch.float32) if k != 'E'
                    else utils.to_tf_device(v, self.device, dtype=torch.long) for k, v in data.items()}
        
        U_mask, V_mask = data['U_mask'].to(dtype=torch.bool), data['V_mask'].to(dtype=torch.bool)
        data = utils.PlaceHolder(data['U'], data['V'], data['E'], data['y']).mask(U_mask, V_mask, edge_loss_fn=self.edge_loss_fn)
        U, V, E, y = data.U, data.V, data.E, data.y
        noisy_data = self.apply_noise(U, V, E, U_mask, V_mask, y)
        if self.conditional:
            y = torch.cat((y, noisy_data['t']), dim=1)
        else:
            y = noisy_data['t']

        pred = self.model(noisy_data['U_t'], noisy_data['V_t'], noisy_data['E_t'], y, U_mask, V_mask, edge_loss_fn=self.edge_loss_fn)
        # Loss
        loss = self.train_loss(pred.U, pred.V, pred.E, U, V, utils.to_tf_device(E, self.device), log=i % self.log_every_steps == 0, U_mask=U_mask, V_mask=V_mask)
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.experiments.train.lr, amsgrad=True,
                                 weight_decay=self.cfg.experiments.train.weight_decay)

    def on_fit_start(self) -> None:
        if self.local_rank == 0:
            utils.setup_wandb(self.cfg)

    def on_train_epoch_start(self) -> None:
        self.print("Starting train epoch...")
        self.start_epoch_time = time.time()
        self.train_loss.reset()
        # self.train_metrics.reset()

    def on_train_epoch_end(self) -> None:
        to_log = self.train_loss.log_epoch_metrics()
        self.log("train_epoch/total_CE", to_log['train_epoch/total_CE'], sync_dist=True)
        self.print(f"Epoch {self.current_epoch}: total_CE: {to_log['train_epoch/total_CE'] :.3f}")
        self.print(f" -- X_CE: {to_log['train_epoch/X_CE'] :.3f} --"
                   f" -- Net_CE: {to_log['train_epoch/Net_CE'] :.3f} --"
                   f" -- E_CE: {to_log['train_epoch/E_CE'] :.3f} --"
                   f" -- {time.time() - self.start_epoch_time:.1f}s ")

    def predict_step(self, data, i, num_nodes=None):
        if len(data) == 2:
            num_nodes = data
        G = self.sample_batch(num_nodes)
        return {'G': G}

    def on_predict_start(self):
        # available after trainer has setup
        self.total_batches = self.trainer.num_predict_batches[0]  # first dataloader
        self.processed_batches = 0
        print(f"Total prediction batches = {self.total_batches}")

    def on_predict_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        self.processed_batches += 1
        proportion = self.processed_batches / self.total_batches
        print(f"Predicted batch {self.processed_batches}/{self.total_batches} "
              f"({proportion:.2%} done)")


    def apply_noise(self, U, V, E, U_mask, V_mask, y, t_int=None):
        """ Sample noise and apply it to the data. """

        # Sample a timestep t.
        # When evaluating, the loss for t=0 is computed separately
        lowest_t = 0 if self.training else 1
        if t_int is None:
            t_int = torch.randint(lowest_t, self.T + 1, size=(E.size(0), 1), device=self.device).float()  # (bs, 1)
        s_int = t_int - 1

        t_float = t_int / self.T
        s_float = s_int / self.T

        # beta_t and alpha_s_bar are used for denoising/loss computation
        beta_t = self.noise_schedule(t_normalized=t_float)                         # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)      # (bs, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)      # (bs, 1)

        Q_U_bar, Q_V_bar, Q_E_bar = self.transition_model.get_Qt_bar(alpha_t_bar, device=self.device)  # (bs, dx_in, dx_out), (bs, dx_in, dx_out), (bs, de_in, de_out)

        # Compute transition probabilities
        probU = U @ Q_U_bar  # (bs, n_d, dx_d_out)
        probV = V @ Q_V_bar  # (bs, n_n, dx_n_out)
        if self.edge_loss_fn == 'CE':
            probE = E @ Q_E_bar.unsqueeze(1)  # (bs, n_d, n_n, de_out)
        elif self.edge_loss_fn == 'BCE':
            probE = utils.to_tf_device(F.one_hot(E, num_classes=2), self.device) @ Q_E_bar.unsqueeze(1).unsqueeze(1)  # (bs, n_d, n_n, de_out, 2)

        sampled_t = diffusion_utils.sample_discrete_features(probU, probV, probE, U_mask, V_mask, edge_loss_fn=self.edge_loss_fn)

        #One-hot Embedding
        U_t = F.one_hot(sampled_t.U, num_classes=self.output_dims['U'])
        V_t = F.one_hot(sampled_t.V, num_classes=self.output_dims['V'])
        assert U.shape == U_t.shape
        assert V.shape == V_t.shape

        if self.edge_loss_fn == 'CE':
            E_t = F.one_hot(sampled_t.E, num_classes=self.output_dims['E']).type_as(E)
        elif self.edge_loss_fn == 'BCE':
            E_t = sampled_t.E
        assert E.shape == E_t.shape

        z_t = utils.PlaceHolder(U=U_t, V=V_t, E=E_t, y=y).type_as(U).mask(U_mask, V_mask, edge_loss_fn=self.edge_loss_fn)

        noisy_data = {'t_int': t_int, 't': t_float, 'beta_t': beta_t, 'alpha_s_bar': alpha_s_bar,
                      'alpha_t_bar': alpha_t_bar, 'U_t': z_t.U, 'V_t': z_t.V, 'E_t': z_t.E,
                      'U_mask': U_mask, 'V_mask': V_mask}
        return noisy_data

    @torch.no_grad()
    def sample_batch(self, num_nodes=None):
        """
        :param num_nodes: int, <int>tensor (batch_size) (optional) for specifying number of nodes
        :param keep_chain: int: number of chains to save to file
        :param keep_chain_steps: number of timesteps to save for each chain
        :return: molecule_list. Each element of this list is a tuple (atom_types, charges, positions)
        """
        batch_size = self.cfg.experiments.test.batch_size

        if num_nodes is None:
            n_devices = self.U_dist.sample((batch_size,)).to(self.device)
            n_nets = []
            for n_device in n_devices:
                n_net = self.V_dist[n_device].sample((1,))
                n_nets.append(n_net)
            n_nets = torch.tensor(n_nets).to(self.device)
        else:
            assert isinstance(num_nodes[0], torch.Tensor)
            assert isinstance(num_nodes[1], torch.Tensor)
            n_devices = num_nodes[0]
            n_nets = num_nodes[1]

        n_U_max = torch.max(n_devices).item()
        n_V_max = torch.max(n_nets).item()
        # Build the masks
        arange = torch.arange(n_U_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        U_mask = arange < n_devices.unsqueeze(1)

        arange = torch.arange(n_V_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        V_mask = arange < n_nets.unsqueeze(1)

        # Sample noise  -- z has size (n_samples, n_nodes, n_features)
        z_T = diffusion_utils.sample_discrete_feature_noise(self.U_marginal, self.V_marginal, self.E_marginal, U_mask, V_mask, edge_loss_fn=self.edge_loss_fn, edge_type_dim=self.output_dims['E'])
        U, V, E = z_T.U, z_T.V, z_T.E

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s_int in reversed(range(0, self.T)):
            s_array = s_int * torch.ones((batch_size, 1)).type_as(U)
            t_array = s_array + 1
            s_norm = s_array / self.T
            t_norm = t_array / self.T

            # Sample z_s
            sampled_s, discrete_sampled_s = self.sample_p_zs_given_zt(s_norm, t_norm, U, V, E, U_mask, V_mask)
            U, V, E = sampled_s.U, sampled_s.V, sampled_s.E

        # Sample
        sampled_s = sampled_s.mask(U_mask, V_mask, collapse=True, edge_loss_fn=self.edge_loss_fn)
        U, V, E = sampled_s.U, sampled_s.V, sampled_s.E

        G = []
        for i in range(batch_size):
            g = {}
            n_U = n_devices[i]
            n_V = n_nets[i]
            g['U'] = U[i, :n_U].cpu()
            g['V'] = V[i, :n_V].cpu()
            g['E'] = E[i, :n_U, :n_V].cpu()
            G.append(g)
        return G

    def sample_p_zs_given_zt(self, s, t, U_t, V_t, E_t, U_mask, V_mask, y=None):
        """Samples from zs ~ p(zs | zt). Only used during sampling.
           if last_step, return the graph prediction as well"""
        bs, n_U, _ = U_t.shape
        _, n_V, _ = V_t.shape
        beta_t = self.noise_schedule(t_normalized=t)  # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)

        # Retrieve transitions matrix
        Qt_U_bar, Qt_V_bar, Qt_E_bar = self.transition_model.get_Qt_bar(alpha_t_bar, device=self.device)
        Qs_U_bar, Qs_V_bar, Qs_E_bar = self.transition_model.get_Qt_bar(alpha_s_bar, device=self.device)
        Qt_U, Qt_V, Qt_E = self.transition_model.get_Qt(beta_t, self.device)

        # Neural net predictions
        if self.conditional:
            y = torch.cat((y, t), dim=1)
        else:
            y = t

        pred = self.model(U_t, V_t, E_t, y, U_mask, V_mask)

        # Normalize predictions
        pred_U = F.softmax(pred.U, dim=-1)
        pred_V = F.softmax(pred.V, dim=-1)
        if self.edge_loss_fn == 'CE':
            pred_E = F.softmax(pred.E, dim=-1)
        elif self.edge_loss_fn == 'BCE':
            pred_E = F.sigmoid(pred.E)


        p_s_and_t_given_0_U = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=U_t,
                                                                                           Qt=Qt_U,
                                                                                           Qsb=Qs_U_bar,
                                                                                           Qtb=Qt_U_bar)
        # Dim of these two tensors: bs, N, d0, d_t-1
        weighted_U = pred_U.unsqueeze(-1) * p_s_and_t_given_0_U  # bs, n, d0, d_t-1
        unnormalized_prob_U = weighted_U.sum(dim=2)  # bs, n, d_t-1
        unnormalized_prob_U[torch.sum(unnormalized_prob_U, dim=-1) == 0] = 1e-5
        prob_U = unnormalized_prob_U / torch.sum(unnormalized_prob_U, dim=-1, keepdim=True)  # bs, n, d_t-1

        assert ((prob_U.sum(dim=-1) - 1).abs() < 1e-4).all()

        p_s_and_t_given_0_V = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=V_t,
                                                                                             Qt=Qt_V,
                                                                                             Qsb=Qs_V_bar,
                                                                                             Qtb=Qt_V_bar)

        # Dim of these two tensors: bs, N, d0, d_t-1
        weighted_V = pred_V.unsqueeze(-1) * p_s_and_t_given_0_V  # bs, n, d0, d_t-1
        unnormalized_prob_V = weighted_V.sum(dim=2)  # bs, n, d_t-1
        unnormalized_prob_V[torch.sum(unnormalized_prob_V, dim=-1) == 0] = 1e-5
        prob_V = unnormalized_prob_V / torch.sum(unnormalized_prob_V, dim=-1, keepdim=True)  # bs, n, d_t-1

        assert ((prob_V.sum(dim=-1) - 1).abs() < 1e-4).all()

        p_s_and_t_given_0_E = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=E_t,
                                                                                           Qt=Qt_E,
                                                                                           Qsb=Qs_E_bar,
                                                                                           Qtb=Qt_E_bar,
                                                                                           edge_loss_fn=self.edge_loss_fn)
        if self.edge_loss_fn == 'CE':
            # Dim of these two tensors: bs, N, d0, d_t-1
            pred_E = pred_E.reshape((bs, -1, pred_E.shape[-1]))
            weighted_E = pred_E.unsqueeze(-1) * p_s_and_t_given_0_E        # bs, N, d0, d_t-1
            unnormalized_prob_E = weighted_E.sum(dim=-2)
            unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
            prob_E = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
            prob_E = prob_E.reshape(bs, n_U, n_V, pred_E.shape[-1])
            assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()
        
        elif self.edge_loss_fn == 'BCE':
            pred_E_two_hot = torch.stack([1.0 - pred_E, pred_E], dim=-1)
            pred_E_flat = pred_E_two_hot.flatten(start_dim=1, end_dim=-2)
            weighted_E = pred_E_flat.unsqueeze(-1) * p_s_and_t_given_0_E
            unnormalized_prob_E = weighted_E.sum(dim=-2)
            unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
            prob_E = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
            prob_E = prob_E.reshape(bs, n_U, n_V, pred_E.shape[-1], 2)

        sampled_s = diffusion_utils.sample_discrete_features(prob_U, prob_V, prob_E, U_mask, V_mask, edge_loss_fn=self.edge_loss_fn)


        U_s = F.one_hot(sampled_s.U, num_classes=self.output_dims['U']).float()
        assert U_s.shape == U_t.shape
        V_s = F.one_hot(sampled_s.V, num_classes=self.output_dims['V']).float()
        assert V_s.shape == V_t.shape

        if self.edge_loss_fn == 'CE':
            E_s = F.one_hot(sampled_s.E, num_classes=self.output_dims['E']).float()
            assert E_t.shape == E_s.shape
        elif self.edge_loss_fn == 'BCE':
            E_s = sampled_s.E
            assert E_t.shape == E_s.shape

        out_one_hot = utils.PlaceHolder(U_s, V_s, E_s, y=torch.zeros(y.shape[0], 0))
        out_discrete = utils.PlaceHolder(U_s, V_s, E_s, y=torch.zeros(y.shape[0], 0))

        return out_one_hot.mask(U_mask, V_mask, edge_loss_fn=self.edge_loss_fn).type_as(y), out_discrete.mask(U_mask, V_mask, collapse=True, edge_loss_fn=self.edge_loss_fn).type_as(y)


class MarginalUniformTransition:
    def __init__(self, U_marginals, V_marginals, E_marginals):
        self.U_classes = len(U_marginals)
        self.V_classes = len(V_marginals)
        self.E_classes = len(E_marginals)

        self.U_marginals = U_marginals
        self.V_marginals = V_marginals
        self.E_marginals = E_marginals

        self.U_T = U_marginals.unsqueeze(0).expand(self.U_classes, -1).unsqueeze(0)
        self.V_T = V_marginals.unsqueeze(0).expand(self.V_classes, -1).unsqueeze(0)
        self.E_T = E_marginals.unsqueeze(0).expand(self.E_classes, -1).unsqueeze(0)

    def get_Qt(self, beta_t, device):
        """ Returns one-step transition matrices for X_d, X_n and E, from step t - 1 to step t.
        Qt = (1 - beta_t) * I + beta_t / K

        beta_t: (bs)                         noise level between 0 and 1
        returns: qx_d (bs, dx_d, dx_d), qx_n (bs, dx_n, dx_n), qe (bs, de, de). """
        beta_t = beta_t.unsqueeze(1)
        beta_t = beta_t.to(device)
        self.U_T = self.U_T.to(device)
        self.V_T = self.V_T.to(device)
        self.E_T = self.E_T.to(device)

        q_U = beta_t * self.U_T + (1 - beta_t) * torch.eye(self.U_classes, device=device).unsqueeze(0)
        q_V = beta_t * self.V_T + (1 - beta_t) * torch.eye(self.V_classes, device=device).unsqueeze(0)
        q_E = beta_t * self.E_T + (1 - beta_t) * torch.eye(self.E_classes, device=device).unsqueeze(0)

        return q_U, q_V, q_E

    def get_Qt_bar(self, alpha_bar_t, device):
        """ Returns t-step transition matrices for X_d, X_n and E, from step 0 to step t.
        Qt = prod(1 - beta_t) * I + (1 - prod(1 - beta_t)) * K

        alpha_bar_t: (bs)         Product of the (1 - beta_t) for each time step from 0 to t.
        returns: qx_d (bs, dx_d, dx_d), qx (bs, dx_n, dx_n), qe (bs, de, de).
        """
        alpha_bar_t = alpha_bar_t.unsqueeze(1)
        alpha_bar_t = alpha_bar_t.to(device)
        self.U_T = self.U_T.to(device)
        self.V_T = self.V_T.to(device)
        self.E_T = self.E_T.to(device)

        q_U = alpha_bar_t * torch.eye(self.U_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.U_T
        q_V = alpha_bar_t * torch.eye(self.V_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.V_T
        q_E = alpha_bar_t * torch.eye(self.E_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.E_T

        return q_U, q_V, q_E

class PredefinedNoiseScheduleDiscrete(torch.nn.Module):
    """
    Predefined noise schedule. Essentially creates a lookup array for predefined (non-learned) noise schedules.
    """

    def __init__(self, noise_schedule, timesteps):
        super(PredefinedNoiseScheduleDiscrete, self).__init__()
        self.timesteps = timesteps

        if noise_schedule == 'cosine':
            betas = diffusion_utils.cosine_beta_schedule_discrete(timesteps)
        elif noise_schedule == 'custom':
            betas = diffusion_utils.custom_beta_schedule_discrete(timesteps)
        else:
            raise NotImplementedError(noise_schedule)

        self.register_buffer('betas', torch.from_numpy(betas).float())

        self.alphas = 1 - torch.clamp(self.betas, min=0, max=0.9999)

        log_alpha = torch.log(self.alphas)
        log_alpha_bar = torch.cumsum(log_alpha, dim=0)
        self.alphas_bar = torch.exp(log_alpha_bar)
        # print(f"[Noise schedule: {noise_schedule}] alpha_bar:", self.alphas_bar)

    def forward(self, t_normalized=None, t_int=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.timesteps)
        return self.betas[t_int.long()]

    def get_alpha_bar(self, t_normalized=None, t_int=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.timesteps)
        return self.alphas_bar.to(t_int.device)[t_int.long()]