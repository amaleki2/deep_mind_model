import itertools
import torch
import torch.nn
import torch.nn.functional as F
from torch.nn import Sequential, ReLU, Linear, LayerNorm, BatchNorm1d
from torch_scatter import scatter_mean, scatter_sum


#+++++++++++++++++++++++++#
#### helper functions #####
#+++++++++++++++++++++++++#
# def pairwise(iterable):
#     "s -> (s0,s1), (s1,s2), (s2, s3), ..."
#     a, b = itertools.tee(iterable)
#     next(b, None)
#     return zip(a, b)
#
#
# def get_edge_counts(data):
#     if data.batch is None:
#         return data.num_edges
#
#     node_indices, counts = torch.unique(data.batch, return_counts=True)
#     node_indices_cum = torch.cumsum(counts, dim=0)
#     node_indices_cum = F.pad(input=node_indices_cum, pad=(1, 0))
#     edge_counts = [
#         torch.sum(torch.all(torch.logical_and(data.edge_index >= i1, data.edge_index < i2), dim=0)).view(1)
#         for i1, i2 in pairwise(node_indices_cum)
#     ]
#     edge_counts = torch.cat(edge_counts)
#     return edge_counts
def get_edge_counts(data):
    counts = torch.bincount(data.batch[data.edge_index[0, :]])
    return counts


def make_mlp_model(n_input, latent_size, n_output, activate_final=False, normalize=True, initializer=False):
    mlp = [Linear(n_input, latent_size),
           # ReLU(),
           # Linear(latent_size, latent_size),
           # ReLU(),
           # Linear(latent_size, latent_size),
           ReLU(),
           Linear(latent_size, n_output)]
    if activate_final:
        mlp.append(ReLU())
    if normalize:
        # mlp.append(BatchNorm1d(n_output))
        mlp.append(LayerNorm(n_output))
    mlp = Sequential(*mlp)

    # this is only for debugging
    if initializer:
        for layer in mlp:
            if hasattr(layer, 'weight'):
                layer.weight.data.fill_(0.35)
                layer.bias.data.fill_(0.)

    return mlp


def cast_globals_to_nodes(data):
    if hasattr(data, "batch") and data.batch is not None:
        node_indices, counts = torch.unique(data.batch, return_counts=True)
        casted_global_attr = [data.u[idx, :] for idx, count in zip(node_indices, counts) for _ in range(count)]
    else:
        assert data.u.size(0) == 1, "batch numbers should be provided."
        casted_global_attr = [data.u] * data.num_nodes
    casted_global_attr = torch.cat(casted_global_attr, dim=0)
    casted_global_attr = casted_global_attr.view(-1, data.u.size(-1))
    return casted_global_attr


def cast_globals_to_edges(data):
    if hasattr(data, "batch") and data.batch is not None:
        if hasattr(data, "edge_counts"):
            edge_counts = data.edge_counts
        else:
            edge_counts = get_edge_counts(data)
        node_indices = torch.unique(data.batch)
        casted_global_attr = [data.u[idx, :] for idx, count in zip(node_indices, edge_counts) for _ in range(count)]
    else:
        assert data.u.size(0) == 1, "batch numbers should be provided."
        casted_global_attr = [data.u] * data.num_edges
    casted_global_attr = torch.cat(casted_global_attr, dim=0)
    casted_global_attr = casted_global_attr.view(-1, data.u.size(-1))
    return casted_global_attr


def cast_edges_to_globals(data, aggr_func=scatter_sum):
    node_indices = torch.unique(data.batch)
    edge_counts = get_edge_counts(data)
    assert sum(edge_counts) == data.num_edges
    indices = [idx.view(1, 1) for idx, count in zip(node_indices, edge_counts) for _ in range(count)]
    indices = torch.cat(indices)
    edge_attr_aggr = aggr_func(data.edge_attr, index=indices, dim=0, dim_size=data.u.size(0))
    return edge_attr_aggr


def cast_nodes_to_globals(data, aggr_func=scatter_sum):
    x_aggr = aggr_func(data.x, index=data.batch, dim=0, dim_size=data.u.size(0))
    return x_aggr


def cast_edges_to_nodes(data, indices, aggr_func=scatter_sum):
    edge_attr_aggr = aggr_func(data.edge_attr, indices, dim=0, dim_size=data.x.size(0))
    return edge_attr_aggr

#+++++++++++++++++++++++++#
## block models: simple ###
#+++++++++++++++++++++++++#
class IndependentEdgeMode(torch.nn.Module):
    def __init__(self,
                 n_edge_feats_in,  # number of input edge features
                 n_edge_feats_out,  # number of output edge features
                 latent_size=128,  # latent size of mlp
                 activate_final=True,  # use activate for the last layer or not?
                 normalize=True  # batch normalize the output
                 ):
        super(IndependentEdgeMode, self).__init__()
        self.params = [n_edge_feats_in, n_edge_feats_out]  # useful for debugging
        self.edge_mlp = make_mlp_model(n_edge_feats_in,
                                       latent_size,
                                       n_edge_feats_out,
                                       activate_final=activate_final,
                                       normalize=normalize)

    def forward(self, data):
        data.edge_attr = self.edge_mlp(data.edge_attr)
        return data


class IndependentNodeModel(torch.nn.Module):
    def __init__(self,
                 n_node_feats_in,    # number of input node features
                 n_node_feats_out,   # number of output node features
                 latent_size=128,    # latent size of mlp
                 activate_final=True,  # use activate for the last layer or not?
                 normalize=True         # batch normalize the output
                 ):
        super(IndependentNodeModel, self).__init__()
        self.params = [n_node_feats_in, n_node_feats_out]
        self.node_mlp = make_mlp_model(n_node_feats_in,
                                       latent_size,
                                       n_node_feats_out,
                                       activate_final=activate_final,
                                       normalize=normalize)

    def forward(self, data):
        data.x = self.node_mlp(data.x)
        return data


class IndependentGlobalModel(torch.nn.Module):
    def __init__(self,
                 n_global_in,
                 n_global_out,
                 latent_size=128,
                 activate_final=True
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

    def forward(self, data):
        data.u = self.global_mlp(data.u)
        return data


#+++++++++++++++++++++++++#
## block models: complex ##
#+++++++++++++++++++++++++#

class EdgeModel(torch.nn.Module):
    def __init__(self,
                 n_edge_feats_in,    # number of input edge features
                 n_edge_feats_out,   # number of output edge features
                 n_node_feats,       # number of input node features
                 n_global_feats,     # number of global (graph) features
                 latent_size=128,    # latent size of mlp
                 activate_final=True,  # use activate for the last layer or not?
                 normalize=True         # batch normalize the output
                 ):
        super(EdgeModel, self).__init__()
        self.params = [n_edge_feats_in, n_edge_feats_out, n_node_feats, n_global_feats]  # useful for debugging
        self.edge_mlp = make_mlp_model(n_edge_feats_in + n_node_feats * 2 + n_global_feats,
                                       latent_size,
                                       n_edge_feats_out,
                                       activate_final=activate_final,
                                       normalize=normalize)

    def forward(self, data):
        row, col = data.edge_index
        sender, receiver = data.x[row, :], data.x[col, :]
        edge_attr = data.edge_attr
        global_attr = cast_globals_to_edges(data)
        out = torch.cat([receiver, sender, edge_attr, global_attr], dim=1)
        data.edge_attr = self.edge_mlp(out)
        return data


class NodeModel(torch.nn.Module):
    def __init__(self,
                 n_node_feats_in,    # number of input node features
                 n_node_feats_out,  # number of output node features
                 n_edge_feats,     # number of input edge features
                 n_global_feats,   # number of global (graph) features
                 latent_size=128,  # latent size of mlp
                 activate_final=True,  # use activate for the last layer or not?
                 agg_func=scatter_sum,  # function to aggregation edges to nodes
                 normalize=True,        # batch normalize the output
                 senders_turned_off=True  # don't aggregate senders
                 ):
        super(NodeModel, self).__init__()
        self.agg_func = agg_func
        self.senders_turned_off = senders_turned_off
        self.params = [n_node_feats_in, n_node_feats_out, n_edge_feats, n_global_feats]
        scalar = 1 if self.senders_turned_off else 2
        self.node_mlp = make_mlp_model(n_node_feats_in + n_edge_feats * scalar + n_global_feats,
                                       latent_size,
                                       n_node_feats_out,
                                       activate_final=activate_final,
                                       normalize=normalize)

    def forward(self, data):
        row, col = data.edge_index
        out = [data.x]
        if not self.senders_turned_off:
            send_edge_attr_agg = cast_edges_to_nodes(data, row, aggr_func=self.agg_func)
            out.append(send_edge_attr_agg)
        recv_edge_attr_agg = cast_edges_to_nodes(data, col, aggr_func=self.agg_func)
        out.append(recv_edge_attr_agg)
        global_attr = cast_globals_to_nodes(data)
        out.append(global_attr)
        out = torch.cat(out, dim=1)
        data.x = self.node_mlp(out)
        return data


class GlobalModel(torch.nn.Module):
    def __init__(self,
                 n_global_in,
                 n_global_out,
                 n_node_feats,
                 n_edge_feats,
                 latent_size=128,
                 activate_final=True
                 ):

        super(GlobalModel, self).__init__()
        self.params = [n_global_in, n_global_out, n_node_feats, n_edge_feats]
        self.global_mlp = make_mlp_model(n_global_in + n_edge_feats + n_node_feats,
                                         latent_size,
                                         n_global_out,
                                         activate_final=activate_final,
                                         normalize=False  # batch normalization does not work when batch size = 1;
                                                          # https://github.com/pytorch/pytorch/issues/7716
                                         )

    def forward(self, data):
        x_aggr = cast_nodes_to_globals(data)
        edge_attr_aggr = cast_edges_to_globals(data)
        global_attr = data.u
        out = torch.cat([x_aggr, edge_attr_aggr, global_attr], dim=1)
        data.u = self.global_mlp(out)
        return data

#  model similar to that introduced in pytorch_geometric
# class NodeModel(torch.nn.Module):
#     def __init__(self,
#                  n_node_feats_in,    # number of input node features
#                  n_nodes_feats_out,  # number of output node features
#                  n_edge_feats,     # number of input edge features
#                  n_global_feats,   # number of global (graph) features
#                  latent_size=128,  # latent size of mlp
#                  activate_final=False,  # use activate for the last layer or not?
#                  normalize=True  # batch normalize the output
#                  ):
#         super(NodeModel, self).__init__()
#         self.params = [n_node_feats_in, n_nodes_feats_out, n_edge_feats, n_global_feats]
#         self.node_mlp_1 = make_mlp_model(n_node_feats_in + n_edge_feats,
#                                          latent_size,
#                                          n_node_feats_in + n_edge_feats,
#                                          activate_final=True,
#                                          normalize=normalize)
#         self.node_mlp_2 = make_mlp_model(n_node_feats_in * 2 + n_edge_feats + n_global_feats,
#                                          latent_size,
#                                          n_nodes_feats_out,
#                                          activate_final=activate_final,
#                                          normalize=normalize)
#
#     def forward(self, x, edge_index, edge_attr, global_attr):
#         row, col = edge_index
#         receiver = x[row]
#         out = torch.cat([receiver, edge_attr], dim=1)
#         out = self.node_mlp_1(out)
#         out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
#         global_attr = pad_to_size(global_attr, out.size(0), mode='repeat')
#         out = torch.cat([x, out, global_attr], dim=1)
#         return self.node_mlp_2(out)


# #  model similar to that introduced in pytorch geometric
# class GlobalModel(torch.nn.Module):
#     def __init__(self,
#                  n_global_in,
#                  n_global_out,
#                  n_node_feats,
#                  latent_size=128,
#                  activate_final=False
#                  ):
#         super(GlobalModel, self).__init__()
#         self.global_mlp = make_mlp_model(n_global_in + n_node_feats,
#                                          latent_size,
#                                          n_global_out,
#                                          activate_final=activate_final,
#                                          normalize=False  # batch normalization does not work when batch size = 1;
#                                                           # https://github.com/pytorch/pytorch/issues/7716
#                                          )
#
#     def forward(self, x, global_attr):
#         xmean = torch.mean(x, dim=0, keepdim=True)
#         out = torch.cat([global_attr, xmean], dim=1)
#         return self.global_mlp(out)



