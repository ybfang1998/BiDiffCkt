import math

import torch
import torch.nn as nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from torch.nn import functional as F
from torch import Tensor

import utils
from modules.diffusion.layers import Xtoy, Etoy, masked_softmax


class XEyTransformerLayer(nn.Module):
    """ Transformer that updates node, edge and global features
        d_U: device features
        d_V: net features
        d_e: edge features
        dz : global features
        n_head: the number of heads in the multi_head_attention
        dim_feedforward: the dimension of the feedforward network model after self-attention
        dropout: dropout probablility. 0 to disable
        layer_norm_eps: eps value in layer normalizations.
    """
    def __init__(self, dx: int, de: int, dy: int, n_head: int, dim_ffU: int = 2048, dim_ffV: int = 2048,
                 dim_ffE: int = 128, dim_ffy: int = 2048, dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5, device=None, dtype=None) -> None:
        kw = {'device': device, 'dtype': dtype}
        super().__init__()


        self.self_attn = NodeEdgeBlock(dx, de, dy, n_head, **kw)

        self.linX1 = Linear(dx, dim_ffU, **kw)
        self.linX2 = Linear(dim_ffU, dx, **kw)
        self.normX1 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.normX2 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.dropoutX1 = Dropout(dropout)
        self.dropoutX2 = Dropout(dropout)
        self.dropoutX3 = Dropout(dropout)

        self.linNet1 = Linear(dx, dim_ffV, **kw)
        self.linNet2 = Linear(dim_ffV, dx, **kw)
        self.normNet1 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.normNet2 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.dropoutNet1 = Dropout(dropout)
        self.dropoutNet2 = Dropout(dropout)
        self.dropoutNet3 = Dropout(dropout)

        self.linE1 = Linear(de, dim_ffE, **kw)
        self.linE2 = Linear(dim_ffE, de, **kw)
        self.normE1 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.normE2 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.dropoutE1 = Dropout(dropout)
        self.dropoutE2 = Dropout(dropout)
        self.dropoutE3 = Dropout(dropout)

        self.lin_y1 = Linear(dy, dim_ffy, **kw)
        self.lin_y2 = Linear(dim_ffy, dy, **kw)
        self.norm_y1 = LayerNorm(dy, eps=layer_norm_eps, **kw)
        self.norm_y2 = LayerNorm(dy, eps=layer_norm_eps, **kw)
        self.dropout_y1 = Dropout(dropout)
        self.dropout_y2 = Dropout(dropout)
        self.dropout_y3 = Dropout(dropout)

        self.activation = F.relu

    def forward(self, U: Tensor, V:Tensor, E: Tensor, y, U_mask: Tensor, V_mask: Tensor):
        """ Pass the input through the encoder layer.
            X: (bs, n, d)
            E: (bs, n, n, d)
            y: (bs, dy)
            node_mask: (bs, n) Mask for the src keys per batch (optional)
            Output: newX, newE, new_y with the same shape.
        """
        # print(X.shape, E.shape, nets.shape, y.shape, node_mask.shape, edge_mask.shape)
        
        newX, newNet, newE = self.self_attn(U, V, E, y, U_mask=U_mask, V_mask=V_mask)

        newU = self.dropoutX1(newX)
        X = self.normX1(U + newU)

        newNet_d = self.dropoutNet1(newNet)
        nets = self.normNet1(V + newNet_d)

        newE_d = self.dropoutE1(newE)
        E = self.normE1(E + newE_d)

        # new_y_d = self.dropout_y1(new_y)
        # y = self.norm_y1(y + new_y_d)

        ff_outputX = self.linX2(self.dropoutX2(self.activation(self.linX1(X))))
        ff_outputX = self.dropoutX3(ff_outputX)
        X = self.normX2(X + ff_outputX)

        ff_outputNet = self.linNet2(self.dropoutNet2(self.activation(self.linNet1(nets))))
        ff_outputNet = self.dropoutNet3(ff_outputNet)
        nets = self.normNet2(nets + ff_outputNet)

        ff_outputE = self.linE2(self.dropoutE2(self.activation(self.linE1(E))))
        ff_outputE = self.dropoutE3(ff_outputE)
        E = self.normE2(E + ff_outputE)

        # ff_output_y = self.lin_y2(self.dropout_y2(self.activation(self.lin_y1(y))))
        # ff_output_y = self.dropout_y3(ff_output_y)
        # y = self.norm_y2(y + ff_output_y)

        # print(X.shape, E.shape, nets.shape)
        return X, nets, E


class NodeEdgeBlock(nn.Module):
    """ Cross attention layer that also updates the representations on the edges. """
    def __init__(self, dx, de, dy, n_head, **kwargs):
        super().__init__()
        assert dx % n_head == 0, f"dx: {dx} -- nhead: {n_head}"
        self.dx = dx
        self.de = de
        self.dy = dy
        self.df = int(dx / n_head)
        self.n_head = n_head

        # Attention
        self.q_x = Linear(dx, dx)
        self.k_x = Linear(dx, dx)
        self.v_x = Linear(dx, dx)
        self.q_net = Linear(dx, dx)
        self.k_net = Linear(dx, dx)
        self.v_net = Linear(dx, dx)

        # FiLM E to X
        self.e_add = Linear(de, dx)
        self.e_mul = Linear(de, dx)

        # FiLM y to E
        self.y_e_mul = Linear(dy, dx)           # Warning: here it's dx and not de
        self.y_e_add = Linear(dy, dx)

        # FiLM y to X
        self.y_x_mul = Linear(dy, dx)
        self.y_x_add = Linear(dy, dx)

        # Process y
        # self.y_y = Linear(dy, dy)
        # self.x_y = Xtoy(dx, dy)
        # self.e_y = Etoy(de, dy)

        # Output layers
        self.x_out = Linear(dx, dx)
        self.net_out = Linear(dx, dx)
        self.e_out = Linear(dx, de)
        # self.y_out = nn.Sequential(nn.Linear(dy, dy), nn.ReLU(), nn.Linear(dy, dy))

    def forward(self, U, V, E, y, U_mask, V_mask):
        """
        :param U: bs, n_d, dx        device features
        :param V: bs, n_n, dx        nets features
        :param E: bs, n_d, n_n, de     edge features
        :param y: bs, dz           global features
        :param U_mask: bs, n_d
        :param V_mask: bs, n_n
        :return: newX, newE, new_y with the same shape.
        """
        bs, n_d, _ = U.shape
        bs, n_n, _ = V.shape
        U_mask = U_mask.unsqueeze(-1)        # bs, n_d, 1
        V_mask = V_mask.unsqueeze(-1)        # bs, n_n, 1
        e_mask1 = U_mask.unsqueeze(2)             # bs, n_d, 1, 1
        e_mask2 = V_mask.unsqueeze(1)             # bs, 1, n_n, 1

        # 1. Map x/net to Q, K, V
        Q_x = self.q_x(U) * U_mask        # (bs, n_d, dx)
        K_x = self.k_x(U) * U_mask
        V_x = self.v_x(U) * U_mask
        utils.assert_correctly_masked(Q_x, U_mask)

        Q_net = self.q_net(V) * V_mask     # (bs, n_n, dx)
        K_net = self.k_net(V) * V_mask
        V_net = self.v_net(V) * V_mask
        utils.assert_correctly_masked(Q_net, V_mask)


        # 2. Reshape to (bs, n_node, n_head, df) with dx = n_head * df
        Q_x = Q_x.reshape((Q_x.size(0), Q_x.size(1), self.n_head, self.df))
        K_x = K_x.reshape((K_x.size(0), K_x.size(1), self.n_head, self.df))
        V_x = V_x.reshape((V_x.size(0), V_x.size(1), self.n_head, self.df))
        Q_x = Q_x.unsqueeze(2)                              # (bs, n_d, 1, n_head, df)
        K_x = K_x.unsqueeze(1)                              # (bs, 1, n_d, n_head, df)
        V_x = V_x.unsqueeze(1)                              # (bs, 1, n_d, n_head, df)

        Q_net = Q_net.reshape((Q_net.size(0), Q_net.size(1), self.n_head, self.df))
        K_net = K_net.reshape((K_net.size(0), K_net.size(1), self.n_head, self.df))
        V_net = V_net.reshape((V_net.size(0), V_net.size(1), self.n_head, self.df))
        Q_net = Q_net.unsqueeze(2)                              # (bs, n_n, 1, n_head, df)
        K_net = K_net.unsqueeze(1)                              # (bs, 1, n_n, n_head, df)
        V_net = V_net.unsqueeze(1)                              # (bs, 1, n_n, n_head, df)


        # Compute unnormalized cross attentions. Y is (bs, n_a, n_b, n_head, df)
        Y_x = Q_x * K_net                                       # (bs, n_d, n_n, n_head, df)
        Y_x = Y_x / math.sqrt(Y_x.size(-1))

        Y_net = Q_net * K_x                                       # (bs, n_n, n_d, n_head, df)
        Y_net = Y_net / math.sqrt(Y_net.size(-1))

        # 3. E FiLM to Y_x and Y_net
        E1 = self.e_mul(E) * e_mask1 * e_mask2                    # bs, n_d, n_n, dx
        E1 = E1.reshape((E.size(0), E.size(1), E.size(2), self.n_head, self.df))   # (bs, n_d, n_n, n_head, df)

        E2 = self.e_add(E) * e_mask1 * e_mask2                    # bs, n_d, n_n, dx
        E2 = E2.reshape((E.size(0), E.size(1), E.size(2), self.n_head, self.df))   # (bs, n_d, n_n, n_head, df)

        # Incorporate edge features to the self attention scores.
        Y_x = Y_x * (E1 + 1) + E2                  # (bs, n_d, n_n, n_head, df)
        Y_net = Y_net * (E1.transpose(1, 2) + 1) + E2.transpose(1, 2)          # (bs, n_n, n_d, n_head, df)
        Y = (Y_x + Y_net.transpose(1, 2))/2    # (bs, n_d, n_n, n_head, df)

        # 4. Incorporate y to E
        newE = Y.flatten(start_dim=3)                      # bs, n_d, n_n, dx
        ye1 = self.y_e_add(y).unsqueeze(1).unsqueeze(1)  # bs, 1, 1, de
        ye2 = self.y_e_mul(y).unsqueeze(1).unsqueeze(1)
        newE = ye1 + (ye2 + 1) * newE

        # Output E
        newE = self.e_out(newE) * e_mask1 * e_mask2     # bs, n_n, n_d, de
        utils.assert_correctly_masked(newE, e_mask1 * e_mask2)

        # 5. Compute attentions for X and Nets.
        softmax_mask = e_mask2.expand(-1, n_d, -1, self.n_head)  # bs, n_d, n_n, n_head
        attn_x = masked_softmax(Y_x, softmax_mask, dim=2)

        softmax_mask = U_mask.unsqueeze(1).expand(-1, n_n, -1, self.n_head) # bs, n_n, n_d, n_head
        attn_net = masked_softmax(Y_net, softmax_mask, dim=2)

        # Compute weighted values
        weighted_V_x = attn_x * V_net
        weighted_V_x = weighted_V_x.sum(dim=2)

        weighted_V_net = attn_net * V_x
        weighted_V_net = weighted_V_net.sum(dim=2)

        # Send output to input dim
        weighted_V_x = weighted_V_x.flatten(start_dim=2)            # bs, n_x, dx
        weighted_V_net = weighted_V_net.flatten(start_dim=2)  # bs, n_net, dx
        # Be Carefull above here, probably problem here !!!!!!!!!!!, nodes representation are same in a batch

        # 6. Incorporate y to X
        yx1 = self.y_x_add(y).unsqueeze(1)
        yx2 = self.y_x_mul(y).unsqueeze(1)
        newX = yx1 + (yx2 + 1) * weighted_V_x
        newNet = yx1 + (yx2 + 1) * weighted_V_net

        # Output X net Net
        newX = self.x_out(newX) * U_mask
        utils.assert_correctly_masked(newX, U_mask)
        newNet = self.net_out(newNet) * V_mask
        utils.assert_correctly_masked(newNet, V_mask)

        # # 5. Process y based on X axnd E
        # y = self.y_y(y)
        # e_y = self.e_y(E)
        # x_y = self.x_y(X)
        # new_y = y + x_y + e_y
        # new_y = self.y_out(new_y)               # bs, dy

        return newX, newNet, newE


class GraphTransformer(nn.Module):
    """
    n_layers : int -- number of layers
    dims : dict -- contains dimensions for each feature type
    """
    def __init__(self, n_layers: int, input_dims: dict, hidden_mlp_dims: dict, hidden_dims: dict,
                 output_dims: dict, act_fn_in: nn.ReLU(), act_fn_out: nn.ReLU()):
        super().__init__()

        self.n_layers = n_layers
        self.out_dim_U = output_dims['U']
        self.out_dim_V = output_dims['V']
        self.out_dim_E = output_dims['E']

        self.mlp_in_U = nn.Sequential(nn.Linear(input_dims['U'], hidden_mlp_dims['U']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['U'], hidden_dims['dx']), act_fn_in)

        self.mlp_in_V = nn.Sequential(nn.Linear(input_dims['V'], hidden_mlp_dims['V']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['V'], hidden_dims['dx']), act_fn_in)

        self.mlp_in_E = nn.Sequential(nn.Linear(input_dims['E'], hidden_mlp_dims['E']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['E'], hidden_dims['de']), act_fn_in)

        self.mlp_in_y = nn.Sequential(nn.Linear(input_dims['y'], hidden_mlp_dims['y']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['y'], hidden_dims['dy']), act_fn_in)
        self.tf_layers = nn.ModuleList([XEyTransformerLayer(dx=hidden_dims['dx'],
                                                            de=hidden_dims['de'],
                                                            dy=hidden_dims['dy'],
                                                            n_head=hidden_dims['n_head'],
                                                            dim_ffU=hidden_dims['dim_ffU'],
                                                            dim_ffV=hidden_dims['dim_ffV'],
                                                            dim_ffE=hidden_dims['dim_ffE'],)
                                        for i in range(n_layers)])

        self.mlp_out_U = nn.Sequential(nn.Linear(hidden_dims['dx'], hidden_mlp_dims['U']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['U'], output_dims['U']))

        self.mlp_out_V = nn.Sequential(nn.Linear(hidden_dims['dx'], hidden_mlp_dims['V']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['V'], output_dims['V']))

        self.mlp_out_E = nn.Sequential(nn.Linear(hidden_dims['de'], hidden_mlp_dims['E']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['E'], output_dims['E']))


    def forward(self, U, V, E, y, U_mask, V_mask, edge_loss_fn='CE', probe_t = None):

        # B, N, _ = U.shape

        U_to_out = U[..., :self.out_dim_U]
        V_to_out = V[..., :self.out_dim_V]
        E_to_out = E[..., :self.out_dim_E]

        after_in = utils.PlaceHolder(U=self.mlp_in_U(U), V=self.mlp_in_V(V), E=self.mlp_in_E(E), y=self.mlp_in_y(y)).mask(U_mask, V_mask, edge_loss_fn=edge_loss_fn)
        U, V, E, y = after_in.U, after_in.V, after_in.E, after_in.y

        for i, layer in enumerate(self.tf_layers):
            U, V, E = layer(U, V, E, y, U_mask, V_mask)
            if probe_t is not None and i == probe_t:
                return U, V, E

        U = self.mlp_out_U(U)
        V = self.mlp_out_V(V)
        E = self.mlp_out_E(E)

        U = (U + U_to_out)
        V = (V + V_to_out)
        E = (E + E_to_out)

        return utils.PlaceHolder(U, V, E, y).mask(U_mask, V_mask, edge_loss_fn=edge_loss_fn)