import numpy as np
import torch
from torch.nn import functional as F

import utils


def cosine_beta_schedule_discrete(timesteps, s=0.008):
    """ Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ. """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)

    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = 1 - alphas
    return betas.squeeze()

def custom_beta_schedule_discrete(timesteps, average_num_nodes=50, s=0.008):
    """ Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ. """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)

    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = 1 - alphas

    assert timesteps >= 100

    p = 4 / 5       # 1 - 1 / num_edge_classes
    num_edges = average_num_nodes * (average_num_nodes - 1) / 2

    # First 100 steps: only a few updates per graph
    updates_per_graph = 1.2
    beta_first = updates_per_graph / (p * num_edges)

    betas[betas < beta_first] = beta_first
    return np.array(betas)

def sample_discrete_features(probU, probV, probE, U_mask, V_mask, edge_loss_fn='CE'):
    ''' Sample features from multinomial distribution with given probabilities (probU, probV, probE)
        :param probU: bs, F, n, du_out        device features
        :param probV: bs, F, n, dv_out        net features
        :param probE: bs, n, n, de_out     edge features
    '''
    if edge_loss_fn == 'CE':
        bs, n_U, n_V, _ = probE.shape
    elif edge_loss_fn == 'BCE':
        bs, n_U, n_V, n_type, _ = probE.shape
    # Noise X
    # The masked rows should define probability distributions as well
    probU[~U_mask] = 1 / probU.shape[-1]
    probV[~V_mask] = 1 / probV.shape[-1]
    # Flatten the probability tensor to sample with multinomial
    probU = probU.reshape(bs * n_U, -1)       # (bs * n_U, du_out)
    probV = probV.reshape(bs * n_V, -1)       # (bs * n_V, dv_out)

    # Sample U_t, V_t
    U_t = probU.multinomial(1)                                  # (bs * n_U, 1)
    U_t = U_t.reshape(bs, n_U)     # (bs, n_U)
    V_t = probV.multinomial(1)                                  # (bs * n_V, 1)
    V_t = V_t.reshape(bs, n_V)     # (bs, n_V)

    # Noise E
    if edge_loss_fn == 'CE':
        # The masked rows should define probability distributions as well
        inverse_edge_mask = ~(V_mask.unsqueeze(1) * U_mask.unsqueeze(2))
        probE[inverse_edge_mask] = 1 / probE.shape[-1]

        probE = probE.reshape(bs * n_U * n_V, -1)    # (bs * n_U * n_V, de_out)

        # Sample E
        E_t = probE.multinomial(1).reshape(bs, n_U, n_V)   # (bs, n_U, n_V)
    elif edge_loss_fn == 'BCE':
        inverse_edge_mask = ~(V_mask.unsqueeze(1) * U_mask.unsqueeze(2))
        probE[inverse_edge_mask] = 1 / probE.shape[-1]

        # sample E
        E_t = torch.bernoulli(probE[..., 1])

    return utils.PlaceHolder(U=U_t, V=V_t, E=E_t, y=torch.zeros(bs, 0).type_as(U_t))

def sample_discrete_feature_noise(U_marginal, V_marginal, E_marginal, U_mask, V_mask, edge_loss_fn='CE', edge_type_dim=None):
    """ Sample from the limit distribution of the diffusion process"""
    bs, n_U_max = U_mask.shape
    _, n_V_max = V_mask.shape
    long_mask = U_mask.long()

    U_limit = U_marginal[None, None, :].expand(bs, n_U_max, -1)
    U = U_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_U_max)
    U = U.type_as(long_mask)
    U = F.one_hot(U, num_classes=U_marginal.shape[-1]).float()

    V_limit = V_marginal[None, None, :].expand(bs, n_V_max, -1)
    V = V_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_V_max)
    V = V.type_as(long_mask)
    V = F.one_hot(V, num_classes=V_marginal.shape[-1]).float()

    if edge_loss_fn == 'CE':
        e_limit = E_marginal[None, None, None, :].expand(bs, n_U_max, n_V_max, -1)
        E = e_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_U_max, n_V_max)
        E = E.type_as(long_mask)
        E = F.one_hot(E, num_classes=e_limit.shape[-1]).float()
    elif edge_loss_fn == 'BCE':
        e_limit = E_marginal[None, None, None, None, :].expand(bs, n_U_max, n_V_max, edge_type_dim, -1)
        E = torch.bernoulli(e_limit[..., 1]).type_as(long_mask).float()

    return utils.PlaceHolder(U=U, V=V, E=E, y=torch.zeros(bs, 0).type_as(U)).mask(U_mask, V_mask, edge_loss_fn=edge_loss_fn)

def compute_batched_over0_posterior_distribution(X_t, Qt, Qsb, Qtb, edge_loss_fn='CE'):
    """ M: X or E
        Compute xt @ Qt.T * x0 @ Qsb / x0 @ Qtb @ xt.T for each possible value of x0
        X_t: bs, n, dt          or bs, n, n, dt  or bs, n, n, n_edge_types, dt
        Qt: bs, d_t-1, dt
        Qsb: bs, d0, d_t-1
        Qtb: bs, d0, dt.
    """
    # Flatten feature tensors
    # Careful with this line. It does nothing if X is a node feature. If X is an edge features it maps to
    # bs x (n ** 2) x d for the case of CE edge loss, and bs x (n ** 2 * n_edge_types) x d for the case of BCE edge loss. The transition matrices should be computed accordingly.
    if edge_loss_fn == 'BCE':
        X_t = X_t.to(torch.float32)
        X_t = torch.stack([1.0 - X_t, X_t], dim=-1)
    X_t = X_t.flatten(start_dim=1, end_dim=-2).to(torch.float32)            # bs x N x dt

    Qt_T = Qt.transpose(-1, -2)                 # bs, dt, d_t-1
    left_term = X_t @ Qt_T                      # bs, N, d_t-1
    left_term = left_term.unsqueeze(dim=2)      # bs, N, 1, d_t-1

    right_term = Qsb.unsqueeze(1)               # bs, 1, d0, d_t-1
    numerator = left_term * right_term          # bs, N, d0, d_t-1

    X_t_transposed = X_t.transpose(-1, -2)      # bs, dt, N

    prod = Qtb @ X_t_transposed                 # bs, d0, N
    prod = prod.transpose(-1, -2)               # bs, N, d0
    denominator = prod.unsqueeze(-1)            # bs, N, d0, 1
    denominator[denominator == 0] = 1e-6

    out = numerator / denominator
    return out