import numpy as np
import torch
import random
import wandb
import omegaconf
import time
import subprocess

def set_seed(seed=0):
    if seed is None:
        return
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_wandb(cfg):
    project_name = cfg.experiments.general.project_name
    if cfg.experiments.general.experiment_type == 'dummy_test':
        project_name = f'{project_name}_dummy_test'
    ts = time.strftime('%b%d-%H:%M:%S', time.gmtime())
    config_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    if cfg.experiments.general.wandb == 'None':
        return
    kwargs = {'name': f'{ts}', 'project': project_name, 'config': config_dict,
              'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': cfg.experiments.general.wandb}
    wandb.init(**kwargs)
    wandb.save('*.txt')

def one_hot_encoding(X, n_bins):
    batch_encode = []
    for x in X:
        one_hot_emb = np.zeros(n_bins)
        one_hot_emb[x] = 1
        batch_encode.append(one_hot_emb)
    return np.array(batch_encode)

def get_g_data(g):
    x_A = g.get_adjacency()

    N = g.vcount()
    x_e = []
    for i in range(0, N):
        current_e = [0]*(N-i-1)
        for j in range(i, N):
            if i == j:
                continue
            if x_A[i, j] == 1:
                current_e[j-i-1] = 1
        x_e += current_e

    x_t = g.vs['type']
    x_p = g.vs['pos']
    x_b = [g.vs['r'], g.vs['c'], g.vs['gm']]
    x_ef = g.get_edgelist()
    return x_A, x_e, x_t, x_p, x_b, x_ef

# Remove Invalid Circuit from Circuit Graph and Spec
def invalid_circuit_remove(G, spec):
    # A circuit is valid if the gain_0db > 0 and gain_0db > max(gain)
    N = len(spec)

    G_new = []
    row_to_drop = []
    i = 0
    while i < N:
        if spec["valid"][i] == 1:
            G_new.append(G[i])
        else:
            row_to_drop.append(i)
        i+=1
    spec = spec.drop(row_to_drop).reset_index(drop=True)
    return G_new, spec

# Fractional parts discarding of specification
def fraction_discarding(specifications):
    specifications["gain"] = specifications["gain"].astype(int)
    specifications["bw"] = specifications["bw"].astype(int)
    specifications["pm"] = specifications["pm"].astype(int)
    return specifications

def spec_scaling(spec):
    spec_scaled = spec.copy()
    spec_scaled['gain'] = (spec_scaled['gain'] - spec_scaled['gain'].min()) / (spec_scaled['gain'].max() - spec_scaled['gain'].min())
    spec_scaled['bw'] = (spec_scaled['bw'] - spec_scaled['bw'].min()) / (
                spec_scaled['bw'].max() - spec_scaled['bw'].min())
    spec_scaled['pm'] = (spec_scaled['pm'] - spec_scaled['pm'].min()) / (
                spec_scaled['pm'].max() - spec_scaled['pm'].min())
    return spec_scaled

def bipartite_edge_to_full(E_uv):
    """
    E_uv: [B, Nu, Nv, De]

    return:
        E_full:   [B, Nu+Nv, Nu+Nv, De]
        edge_mask:[Nu+Nv, Nu+Nv]
    """
    B, Nu, Nv, De = E_uv.shape

    N = Nu + Nv

    E_full = E_uv.new_zeros(B, N, N, De)

    # U -> V
    E_full[:, :Nu, Nu:, :] = E_uv

    # V -> U
    E_full[:, Nu:, :Nu, :] = E_uv.transpose(1, 2)

    # # True = this location is allowed to have an edge
    # edge_mask = torch.zeros(
    #     N, N,
    #     dtype=torch.bool,
    #     device=E_uv.device
    # )

    # edge_mask[:Nu, Nu:] = True   # U -> V
    # edge_mask[Nu:, :Nu] = True   # V -> U

    return E_full

def full_edge_to_bipartite(E_full, Nu, Nv):
    """
    E_full: [B, Nu+Nv, Nu+Nv, De]
    """
    E_uv = E_full[:, :Nu, Nu:Nu+Nv, :]

    return E_uv

def to_tf_device(x, device, dtype=torch.float32):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)  # convert from numpy or list
    if device is not None:
        return x.to(dtype=dtype, device=device)
    else:
        return x.to(dtype=dtype)

class Digress_PlaceHolder:
    def __init__(self, X, E, y):
        self.X = X
        self.E = E
        self.y = y

    def type_as(self, x: torch.Tensor):
        """ Changes the device and dtype of X, E, y. """
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        self.y = self.y.type_as(x)
        return self

    def mask(self, node_mask, collapse=False):
        x_mask = node_mask.unsqueeze(-1)          # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)             # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)             # bs, 1, n, 1

        if collapse:
            self.X = torch.argmax(self.X, dim=-1)
            self.E = torch.argmax(self.E, dim=-1)

            self.X[node_mask == 0] = - 1
            self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = - 1
        else:
            self.X = self.X * x_mask
            self.E = self.E * e_mask1 * e_mask2
            assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))
        return self

class PlaceHolder:
    def __init__(self, U, V, E, y):
        self.U = U
        self.V = V
        self.E = E
        self.y = y

    def type_as(self, x: torch.Tensor):
        """ Changes the device and dtype of U, V, E, y. """
        self.U = self.U.type_as(x)
        self.V = self.V.type_as(x)
        self.E = self.E.type_as(x)
        self.y = self.y.type_as(x)
        return self

    def mask(self, U_mask, V_mask, collapse=False, edge_loss_fn='CE'):
        U_mask = U_mask.unsqueeze(-1)          # bs, u, 1
        V_mask = V_mask.unsqueeze(-1)          # bs, v, 1
        e_mask1 = U_mask.unsqueeze(2)             # bs, u, 1, 1
        e_mask2 = V_mask.unsqueeze(1)             # bs, 1, v, 1

        if collapse:
            self.U = torch.argmax(self.U, dim=-1)
            self.V = torch.argmax(self.V, dim=-1)

            self.U[U_mask.squeeze(-1) == 0] = -1
            self.V[V_mask.squeeze(-1) == 0] = -1
            if edge_loss_fn == 'CE':
                self.E = torch.argmax(self.E, dim=-1)
                self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = - 1
            elif edge_loss_fn == 'BCE':
                self.E = self.E * e_mask1 * e_mask2

        else:
            self.U = self.U * U_mask
            self.V = self.V * V_mask
            self.E = self.E * e_mask1 * e_mask2
        return self

def assert_correctly_masked(variable, node_mask):
    assert (variable * (1 - node_mask.long())).abs().max().item() < 1e-4, \
        'Variables not masked properly.'

def bipartite_get_batch_edge_mask(n_v_list, batch_size, max_n_v, max_n_net):
    edge_mask = np.ones((batch_size, max_n_v, max_n_net), dtype=bool)
    for i, n_v in enumerate(n_v_list):
        edge_mask[i][n_v: max_n_v] = False
    return edge_mask


def get_best_gpu():
    try:
        cmd = "nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader"
        output = subprocess.check_output(cmd, shell=True).decode("utf-8")
        free_memory = [int(x) for x in output.strip().split("\n")]

        best_gpu_index = free_memory.index(max(free_memory))
        return best_gpu_index
    except Exception as e:
        print(f"no GPU found, using CPU. Error: {e}")
        return 0