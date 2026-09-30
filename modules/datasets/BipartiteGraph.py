import numpy as np
from torch.utils.data import Dataset
from utils import *
import torch.distributions as dist

class BipartiteGraphDataset(Dataset):
    def __init__(self, U, V, E, y):
        self.U = U
        self.V = V
        self.E = E
        self.y = y

    def __len__(self):
        return len(self.U)

class OCB_BipartiteGraphDataset(Dataset):
    def __init__(self, G, N_TYPES=24, E_TYPES=3, conditional=False):
        self.N_TYPES = N_TYPES
        self.E_TYPES = E_TYPES
        self.n_U_max = 6
        self.n_V_max = 4

        self.G = G
        self.conditional = conditional

        # Marginal Distribution
        U_types = np.array([])
        V_types = np.array([])
        E_types = np.array([])

        # Count Number of Nodes Distribution
        U_dist =  [0] * (self.n_U_max + 1)
        ## Distribution of number of nets/ports for each category of number of device (to ensure there are enough nets)
        V_dist = [[0] * (self.n_V_max + 1) for _ in range(self.n_U_max + 1)]

        for g in G:
            U_types = np.concatenate((U_types, g['U']))
            V_types = np.concatenate((V_types, g['V']))
            E_types = np.concatenate((E_types, g['E'][0: len(g['U']), 0: len(g['V'])].flatten()))

            U_dist[len(g['U'])] += 1
            V_dist[len(g['U'])][len(g['V'])] += 1


        marginal_U_type = np.unique(U_types, return_counts=True)[1]
        marginal_V_type = np.unique(V_types, return_counts=True)[1]
        marginal_E_type = np.unique(E_types, return_counts=True)[1]

        # Device,Net and edges types Marginal Distribution
        self.marginal_U_type = marginal_U_type/marginal_U_type.sum()
        self.marginal_V_type = marginal_V_type/marginal_V_type.sum()
        self.marginal_E_type = marginal_E_type/marginal_E_type.sum()

        # Distributions for number of nodes construction
        U_dist = torch.tensor(U_dist, dtype=torch.float)
        U_dist = dist.Categorical(probs=U_dist)

        V_dists = []
        for probs in V_dist:
            try:
                V_dists.append(dist.Categorical(probs=torch.tensor(probs, dtype=torch.float)))
            except ValueError:
                V_dists.append(None)
        self.nodes_dist = [U_dist, V_dists]
    
    def __len__(self):
        return len(self.G)

    def __getitem__(self, idx):
        g = self.G[idx]

        U = g['U']
        V = g['V']
        E = g['E']
        if self.conditional:
            y = g['Y']
        else:
            y = []

        U_mask = g['U_mask']
        V_mask = g['V_mask']

        # Nodes One-hot embedding
        U_one_hot = one_hot_encoding(U, self.N_TYPES)
        V_one_hot = one_hot_encoding(V, 3)
        # Nodes Padding
        U_one_hot = np.vstack((U_one_hot, np.zeros((self.n_U_max - len(U), self.N_TYPES))))
        V_one_hot = np.vstack((V_one_hot, np.zeros((self.n_V_max - len(V), 3))))

        #bi-matrix one-hot encoding
        E_one_hot = np.eye(self.E_TYPES)[E.astype(int)]

        return {
            'U': U_one_hot,
            'V': V_one_hot,
            'E': E_one_hot,
            'y': y,
            'U_mask': U_mask,
            'V_mask': V_mask,
        }

class AnalogGenie_BipartiteGraphDataset(BipartiteGraphDataset):
    def __init__(self, X_d, X_n, E, y=None, E_TYPES=7):
        BipartiteGraphDataset.__init__(self, X_d, X_n, E, y)
        self.E_TYPES = E_TYPES
        self.max_n_device = 72
        self.max_n_net = 59

        # Marginal Distribution
        X_d_types = []
        X_n_types = []
        E_types = []
        for x_d, x_n, bi_mat in zip(self.X_d, self.X_n, self.E):
            X_d_types += x_d
            X_n_types += x_n
            E_types += bi_mat[0: len(x_d), 0: len(x_n)].flatten().tolist()

        marginal_device_type = np.unique(X_d_types, return_counts=True)[1]
        marginal_net_type = np.unique(X_n_types, return_counts=True)[1]

        # Device/Net types Marginal Distribution
        self.marginal_device_type = marginal_device_type/marginal_device_type.sum()
        self.marginal_net_type = marginal_net_type/marginal_net_type.sum()

        # Edge Marginal Distribution
        marginal_edge = np.unique(np.array(E_types), return_counts=True)[1]
        self.marginal_edge = marginal_edge/marginal_edge.sum()

    def __getitem__(self, idx):
        x_d = self.X_d[idx]
        x_n = self.X_n[idx]
        E = self.E[idx]

        ## Node Masks
        x_d_mask = [True] * len(x_d) + [False] * (self.max_n_device - len(x_d))
        x_n_mask = [True] * len(x_n) + [False] * (self.max_n_net - len(x_n))
        # edge_mask = np.ones((self.max_n_device, self.max_n_net), dtype=bool)
        # edge_mask[len(x_d): self.max_n_device] = False
        # edge_mask[:, len(x_n): self.max_n_net] = False

        # Nodes One-hot embedding
        x_d_one_hot = one_hot_encoding(x_d, 12)
        x_n_one_hot = one_hot_encoding(x_n, 30)
        # Nodes Padding
        x_d_one_hot = np.vstack((x_d_one_hot, np.zeros((self.max_n_device - len(x_d), 12))))
        x_n_one_hot = np.vstack((x_n_one_hot, np.zeros((self.max_n_net - len(x_n), 30))))

        #bi-matrix one-hot encoding
        E_one_hot = np.eye(self.E_TYPES)[E.astype(int)]

        return {
            'x_d': x_d_one_hot,
            'x_n': x_n_one_hot,
            'E': E_one_hot,
            'y': [],
            'x_d_mask': np.array(x_d_mask),
            'x_n_mask': np.array(x_n_mask),
        }

class Diffckt_BipartiteGraphDataset(Dataset):
    def __init__(self, G, conditional=False, E_TYPES=5):
        self.E_TYPES = E_TYPES
        self.n_U_max = 13
        self.n_V_max = 18
        self.G = G
        self.conditional = conditional

        # Marginal Distribution for nodes
        U_types = np.array([])
        V_types = np.array([])

        # Count Number of Nodes Distribution
        U_dist =  [0] * (self.n_U_max + 1)
        ## Distribution of number of nets/ports for each category of number of device (to ensure there are enough nets)
        V_dist = [[0] * (self.n_V_max + 1) for _ in range(self.n_U_max + 1)]

        positive_edge_count = np.zeros(self.E_TYPES)
        posible_edge_count = 0

        for g in G:
            U_types = np.concatenate((U_types, g['U']))
            V_types = np.concatenate((V_types, g['V']))
            U_dist[len(g['U'])] += 1
            V_dist[len(g['U'])][len(g['V'])] += 1

            # Calculate edge pos_weight
            U = g['U']
            V = g['V']
            posible_edge_count += len(U) * len(V)
            E = g['E']
            positive_edge_count += E.sum(axis=(0, 1))

        self.pos_weight = (posible_edge_count - positive_edge_count) / positive_edge_count

        marginal_U_type = np.unique(U_types, return_counts=True)[1]
        marginal_V_type = np.unique(V_types, return_counts=True)[1]

        # Device/Net types Marginal Distribution
        self.marginal_U_type = marginal_U_type/marginal_U_type.sum()
        self.marginal_V_type = marginal_V_type/marginal_V_type.sum()

        # Uniform Distribution for each edge type (each label in multi-label edge representation has the same probability of 0/1)
        self.marginal_E_type = np.ones(2) * 0.5

    
        # Distributions for number of nodes construction
        U_dist = torch.tensor(U_dist, dtype=torch.float)
        U_dist = dist.Categorical(probs=U_dist)

        V_dists = []
        for probs in V_dist:
            try:
                V_dists.append(dist.Categorical(probs=torch.tensor(probs, dtype=torch.float)))
            except ValueError:
                V_dists.append(None)
        self.nodes_dist = [U_dist, V_dists]


    
    def __len__(self):
        return len(self.G)

    def __getitem__(self, idx):
        g = self.G[idx]

        U = g['U']
        V = g['V']
        E = g['E']
        U_params = g['node_param'][:U.shape[0], :]
        if self.conditional:
            y = g['Y']
        else:
            y = []

        ## Node Masks
        U_mask = g['U_mask']
        V_mask = g['V_mask']
        
        # Nodes One-hot embedding
        U_one_hot = one_hot_encoding(U, 10)
        V_one_hot = one_hot_encoding(V, 10)

        # Nodes Padding
        U_one_hot = np.vstack((U_one_hot, np.zeros((self.n_U_max - len(U), 10))))
        V_one_hot = np.vstack((V_one_hot, np.zeros((self.n_V_max - len(V), 10))))
        U_params = np.vstack((U_params, np.zeros((self.n_U_max - len(U), 2))))

        return {
            'U': U_one_hot,
            'V': V_one_hot,
            'E': E,
            'y': y,
            'U_mask': U_mask,
            'V_mask': V_mask,
            'U_params': U_params
        }

class Diffckt_Multiclass_BipartiteGraphDataset(Dataset):
    def __init__(self, G, conditional=False, E_TYPES=21):
        self.E_TYPES = E_TYPES
        self.n_U_max = 13
        self.n_V_max = 18
        self.G = G
        self.conditional = conditional

        # Marginal Distribution for nodes
        U_types_list = []
        V_types_list = []
        E_types_list = []

        # Count Number of Nodes Distribution
        U_dist =  [0] * (self.n_U_max + 1)
        # Distribution of number of nets/ports for each category of number of device (to ensure there are enough nets)
        V_dist = [[0] * (self.n_V_max + 1) for _ in range(self.n_U_max + 1)]

        for g in G:
            U_types_list.append(g['U'])
            V_types_list.append(g['V'])

            n_U = len(g['U'])
            n_V = len(g['V'])
            E_types_list.append(g['E'][:n_U, :n_V].flatten())

            # Construct the distribution of number of nodes from dataset
            U_dist[len(g['U'])] += 1
            V_dist[len(g['U'])][len(g['V'])] += 1

        U_types = np.concatenate(U_types_list)
        V_types = np.concatenate(V_types_list)
        E_types = np.concatenate(E_types_list)
        marginal_U_type = np.unique(U_types, return_counts=True)[1]
        marginal_V_type = np.unique(V_types, return_counts=True)[1]
        marginal_E_type = np.unique(E_types, return_counts=True)[1]

        # Device/Net types Marginal Distribution
        self.marginal_U_type = marginal_U_type/marginal_U_type.sum()
        self.marginal_V_type = marginal_V_type/marginal_V_type.sum()
        self.marginal_E_type = marginal_E_type/marginal_E_type.sum()

    
        # Distributions for number of nodes construction
        U_dist = torch.tensor(U_dist, dtype=torch.float)
        U_dist = dist.Categorical(probs=U_dist)

        V_dists = []
        for probs in V_dist:
            try:
                V_dists.append(dist.Categorical(probs=torch.tensor(probs, dtype=torch.float)))
            except ValueError:
                V_dists.append(None)
        self.nodes_dist = [U_dist, V_dists]
    
    def __len__(self):
        return len(self.G)
    
    def __getitem__(self, idx):
        g = self.G[idx]

        U = g['U']
        V = g['V']
        E = g['E']
        U_params = g['node_param'][:U.shape[0], :]
        if self.conditional:
            y = g['Y']
        else:
            y = []

        ## Node Masks
        U_mask = g['U_mask']
        V_mask = g['V_mask']

        # One-hot embedding
        U_one_hot = one_hot_encoding(U, 10)
        V_one_hot = one_hot_encoding(V, 10)
        E_one_hot = np.eye(self.E_TYPES)[E.astype(int)]

        # Nodes Padding
        U_one_hot = np.vstack((U_one_hot, np.zeros((self.n_U_max - len(U), 10))))
        V_one_hot = np.vstack((V_one_hot, np.zeros((self.n_V_max - len(V), 10))))
        U_params = np.vstack((U_params, np.zeros((self.n_U_max - len(U), 2))))

        return {
            'U': U_one_hot,
            'V': V_one_hot,
            'E': E_one_hot,
            'y': y,
            'U_mask': U_mask,
            'V_mask': V_mask,
            'U_params': U_params,
        }

class LaMAGIC_BipartiteGraphDataset(Dataset):
    def __init__(self, data, conditional=False):
        self.E_TYPES = 2
        self.n_U_max = data['max_U']
        self.n_V_max = data['max_V']
        self.G = data['G']
        self.conditional = conditional
        self.n_U_types = len(data['U_NAME_TO_ID'])
        self.n_V_types = len(data['V_NAME_TO_ID'])

        # Marginal Distribution for nodes
        U_types_list = []
        V_types_list = []
        E_types_list = []

        # Count Number of Nodes Distribution
        U_dist =  [0] * (self.n_U_max + 1)
        # Distribution of number of nets/ports for each category of number of device (to ensure there are enough nets)
        V_dist = [[0] * (self.n_V_max + 1) for _ in range(self.n_U_max + 1)]

        for g in self.G:
            U_types_list.append(g['U'])
            V_types_list.append(g['V'])

            n_U = len(g['U'])
            n_V = len(g['V'])
            E_types_list.append(g['E'][:n_U, :n_V].flatten())

            # Construct the distribution of number of nodes from dataset
            U_dist[len(g['U'])] += 1
            V_dist[len(g['U'])][len(g['V'])] += 1

        U_types = np.concatenate(U_types_list)
        V_types = np.concatenate(V_types_list)
        E_types = np.concatenate(E_types_list)
        marginal_U_type = np.unique(U_types, return_counts=True)[1]
        marginal_V_type = np.unique(V_types, return_counts=True)[1]
        marginal_E_type = np.unique(E_types, return_counts=True)[1]

        # Device/Net types Marginal Distribution
        self.marginal_U_type = marginal_U_type/marginal_U_type.sum()
        self.marginal_V_type = marginal_V_type/marginal_V_type.sum()
        self.marginal_E_type = marginal_E_type/marginal_E_type.sum()

    
        # Distributions for number of nodes construction
        U_dist = torch.tensor(U_dist, dtype=torch.float)
        U_dist = dist.Categorical(probs=U_dist)

        V_dists = []
        for probs in V_dist:
            try:
                V_dists.append(dist.Categorical(probs=torch.tensor(probs, dtype=torch.float)))
            except ValueError:
                V_dists.append(None)
        self.nodes_dist = [U_dist, V_dists]
    
    def __len__(self):
        return len(self.G)
    
    def __getitem__(self, idx):
        g = self.G[idx]

        U = g['U']
        V = g['V']
        E = g['E']
        if self.conditional:
            y = []
        else:
            y = []

        ## Node Masks
        U_mask = g['U_mask']
        V_mask = g['V_mask']

        # One-hot embedding
        U_one_hot = one_hot_encoding(U, self.n_U_types)
        V_one_hot = one_hot_encoding(V, self.n_V_types)
        E_one_hot = np.eye(self.E_TYPES)[E.astype(int)]

        # Nodes Padding
        U_one_hot = np.vstack((U_one_hot, np.zeros((self.n_U_max - len(U), self.n_U_types))))
        V_one_hot = np.vstack((V_one_hot, np.zeros((self.n_V_max - len(V), self.n_V_types))))

        return {
            'U': U_one_hot,
            'V': V_one_hot,
            'E': E_one_hot,
            'y': y,
            'U_mask': U_mask,
            'V_mask': V_mask,
        }



