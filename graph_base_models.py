import torch
import torch.nn
from torch.nn import Sequential, ReLU, Linear, LayerNorm, BatchNorm1d
from torch_scatter import scatter_mean, scatter_sum


#+++++++++++++++++++++++++#
#### helper functions #####
#+++++++++++++++++++++++++#
def make_mlp_model(n_input, latent_size, n_output, activate_final=False, normalize=True):
    mlp = [Linear(n_input, latent_size),
           ReLU(),
           Linear(latent_size, latent_size),
           ReLU(),
           Linear(latent_size, latent_size),
           ReLU(),
           Linear(latent_size, n_output)]
    if activate_final:
        mlp.append(ReLU())
    if normalize:
        # mlp.append(BatchNorm1d(n_output))
        mlp.append(LayerNorm(n_output))
    mlp = Sequential(*mlp)
    return mlp


def pad_to_size(x, desired_size, mode='pad0'):
    xsize = x.size(0)
    if xsize >= desired_size:
        return x

    if mode == 'pad0':
        zero_pad = torch.zeros(desired_size - xsize, *x.size()[1:])
        new_x = torch.cat([x, zero_pad], dim=0)
    elif mode == 'repeat':
        assert desired_size % xsize == 0, "repeat mode only valid when desired size is a multipe of xsize"
        n_repeats = desired_size // xsize
        new_x = torch.cat([x] * n_repeats, dim=0)
    else:
        raise(ValueError("only pad0 and repeat accepted."))
    return new_x


#+++++++++++++++++++++++++#
#### block models #####
#+++++++++++++++++++++++++#
class IndependentEdgeMode(torch.nn.Module):
    def __init__(self,
                 n_edge_feats_in,  # number of input edge features
                 n_edge_feats_out,  # number of output edge features
                 latent_size=128,  # latent size of mlp
                 activate_final=False,  # use activate for the last layer or not?
                 normalize=True  # batch normalize the output
                 ):
        super(IndependentEdgeMode, self).__init__()
        self.params = [n_edge_feats_in, n_edge_feats_out]  # useful for debugging
        self.edge_mlp = make_mlp_model(n_edge_feats_in,
                                       latent_size,
                                       n_edge_feats_out,
                                       activate_final=activate_final,
                                       normalize=normalize)

    def forward(self, edge_attr):
        return self.edge_mlp(edge_attr)


class IndependentNodeModel(torch.nn.Module):
    def __init__(self,
                 n_node_feats_in,    # number of input node features
                 n_node_feats_out,   # number of output node features
                 latent_size=128,    # latent size of mlp
                 activate_final=False,  # use activate for the last layer or not?
                 normalize=True         # batch normalize the output
                 ):
        super(IndependentNodeModel, self).__init__()
        self.params = [n_node_feats_in, n_node_feats_out]
        self.node_mlp = make_mlp_model(n_node_feats_in,
                                       latent_size,
                                       n_node_feats_out,
                                       activate_final=activate_final,
                                       normalize=normalize)

    def forward(self, x):
        return self.node_mlp(x)


class IndependentGlobalModel(torch.nn.Module):
    def __init__(self,
                 n_global_in,
                 n_global_out,
                 latent_size=128,
                 activate_final=False
                 ):

        super(IndependentGlobalModel, self).__init__()
        self.params = [n_global_in, n_global_out]
        self.global_mlp = make_mlp_model(n_global_in,
                                         latent_size,
                                         n_global_out,
                                         activate_final=activate_final,
                                         normalize=False  # batch normalization does not work when batch size = 1;
                                                          # https://github.com/pytorch/pytorch/issues/7716
                                         )

    def forward(self, global_attr):
        return self.global_mlp(global_attr)


class EdgeModel(torch.nn.Module):
    def __init__(self,
                 n_edge_feats_in,    # number of input edge features
                 n_edge_feats_out,   # number of output edge features
                 n_node_feats,       # number of input node features
                 n_global_feats,     # number of global (graph) features
                 latent_size=128,    # latent size of mlp
                 activate_final=False,  # use activate for the last layer or not?
                 normalize=True         # batch normalize the output
                 ):
        super(EdgeModel, self).__init__()
        self.params = [n_edge_feats_in, n_edge_feats_out, n_node_feats, n_global_feats]  # useful for debugging
        self.edge_mlp = make_mlp_model(n_edge_feats_in + n_node_feats * 2 + n_global_feats,
                                       latent_size,
                                       n_edge_feats_out,
                                       activate_final=activate_final,
                                       normalize=normalize)

    def forward(self, receiver, sender, edge_attr, global_attr):
        global_attr = pad_to_size(global_attr, edge_attr.size(0), mode='repeat')
        out = torch.cat([receiver, sender, edge_attr, global_attr], dim=1)
        return self.edge_mlp(out)


#  model similar to that introduced in pytorch_geometric
class NodeModel(torch.nn.Module):
    def __init__(self,
                 n_node_feats_in,    # number of input node features
                 n_nodes_feats_out,  # number of output node features
                 n_edge_feats,     # number of input edge features
                 n_global_feats,   # number of global (graph) features
                 latent_size=128,  # latent size of mlp
                 activate_final=False,  # use activate for the last layer or not?
                 normalize=True  # batch normalize the output
                 ):
        super(NodeModel, self).__init__()
        self.params = [n_node_feats_in, n_nodes_feats_out, n_edge_feats, n_global_feats]
        self.node_mlp_1 = make_mlp_model(n_node_feats_in + n_edge_feats,
                                         latent_size,
                                         n_node_feats_in + n_edge_feats,
                                         activate_final=True,
                                         normalize=normalize)
        self.node_mlp_2 = make_mlp_model(n_node_feats_in * 2 + n_edge_feats + n_global_feats,
                                         latent_size,
                                         n_nodes_feats_out,
                                         activate_final=activate_final,
                                         normalize=normalize)

    def forward(self, x, edge_index, edge_attr, global_attr):
        row, col = edge_index
        receiver = x[row]
        out = torch.cat([receiver, edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        global_attr = pad_to_size(global_attr, out.size(0), mode='repeat')
        out = torch.cat([x, out, global_attr], dim=1)
        return self.node_mlp_2(out)


#  model similar to that introduced in pytorch geometric
class GlobalModel(torch.nn.Module):
    def __init__(self,
                 n_global_in,
                 n_global_out,
                 n_node_feats,
                 latent_size=128,
                 activate_final=False
                 ):
        super(GlobalModel, self).__init__()
        self.global_mlp = make_mlp_model(n_global_in + n_node_feats,
                                         latent_size,
                                         n_global_out,
                                         activate_final=activate_final,
                                         normalize=False  # batch normalization does not work when batch size = 1;
                                                          # https://github.com/pytorch/pytorch/issues/7716
                                         )

    def forward(self, x, global_attr):
        xmean = torch.mean(x, dim=0, keepdim=True)
        out = torch.cat([global_attr, xmean], dim=1)
        return self.global_mlp(out)


# similar to GN paper
# https://github.com/fxia22/gn.pytorch/blob/master/gn_models.py
class NodeModel_GN(torch.nn.Module):
    def __init__(self,
                 n_node_feats_in,    # number of input node features
                 n_node_feats_out,  # number of output node features
                 n_edge_feats,     # number of input edge features
                 n_global_feats,   # number of global (graph) features
                 latent_size=128,  # latent size of mlp
                 activate_final=False,  # use activate for the last layer or not?
                 agg_func=scatter_sum,  # function to aggregation edges to nodes
                 normalize=True         # batch normalize the output
                 ):
        super(NodeModel_GN, self).__init__()
        self.agg_func = agg_func
        self.params = [n_node_feats_in, n_node_feats_out, n_edge_feats, n_global_feats]
        self.node_mlp = make_mlp_model(n_node_feats_in + n_edge_feats * 2 + n_global_feats,
                                       latent_size,
                                       n_node_feats_out,
                                       activate_final=activate_final,
                                       normalize=normalize)

    def forward(self, x, edge_index, edge_attr, global_attr):
        row, col = edge_index
        send_edge_attr_agg = self.agg_func(edge_attr, row, dim=0, dim_size=x.size(0))
        recv_edge_attr_agg = self.agg_func(edge_attr, col, dim=0, dim_size=x.size(0))
        global_attr = pad_to_size(global_attr, x.size(0), mode='repeat')
        out = torch.cat([x, send_edge_attr_agg, recv_edge_attr_agg, global_attr], dim=1)
        return self.node_mlp(out)


# similar to GN paper
# https://github.com/fxia22/gn.pytorch/blob/master/gn_models.py
class GlobalModel_GN(torch.nn.Module):
    def __init__(self,
                 n_global_in,
                 n_global_out,
                 n_node_feats,
                 n_edge_feats,
                 latent_size=128,
                 activate_final=False
                 ):

        super(GlobalModel_GN, self).__init__()
        self.params = [n_global_in, n_global_out, n_node_feats, n_edge_feats]
        self.global_mlp = make_mlp_model(n_global_in + n_edge_feats + n_node_feats,
                                         latent_size,
                                         n_global_out,
                                         activate_final=activate_final,
                                         normalize=False  # batch normalization does not work when batch size = 1;
                                                          # https://github.com/pytorch/pytorch/issues/7716
                                         )

    def forward(self, x, edge_attr, global_attr):
        x_mean = torch.mean(x, dim=0, keepdim=True)
        edge_attr_mean = torch.mean(edge_attr, dim=0, keepdim=True)
        out = torch.cat([x_mean, edge_attr_mean, global_attr], dim=1)
        return self.global_mlp(out)
