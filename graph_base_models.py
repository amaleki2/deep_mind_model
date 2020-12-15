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
def get_edge_counts(edge_index, batch):
    return torch.bincount(batch[edge_index[0, :]])


def make_mlp_model(n_input, latent_size, n_output, activate_final=False,
                   normalize=True, initializer=False):
    if latent_size is None:
        mlp = [Linear(n_input, n_output)]
    else:
        mlp = [Linear(n_input, latent_size),
               ReLU(),
               # Linear(latent_size, latent_size),
               # ReLU(),
               # Linear(latent_size, latent_size),
               # ReLU(),
               Linear(latent_size, n_output)]
    if activate_final:
        mlp.append(ReLU())
    if normalize and latent_size is not None:
        # mlp.append(BatchNorm1d(n_output))
        mlp.append(LayerNorm(n_output))
    mlp = Sequential(*mlp)

    # this is only for debugging
    if initializer:
        for layer in mlp:
            if layer._get_name() == "Linear":
                layer.weight.data.fill_(0.01)
                layer.bias.data.fill_(0.01)

    return mlp


def cast_globals_to_nodes(global_attr, batch=None, num_nodes=None):
    if batch is not None:
        node_indices, counts = torch.unique(batch, return_counts=True)
        casted_global_attr = [global_attr[idx, :] for idx, count in zip(node_indices, counts) for _ in range(count)]
    else:
        assert global_attr.size(0) == 1, "batch numbers should be provided."
        assert num_nodes is not None, "number of nodes should be specified."
        casted_global_attr = [global_attr] * num_nodes
    casted_global_attr = torch.cat(casted_global_attr, dim=0)
    casted_global_attr = casted_global_attr.view(-1, global_attr.size(-1))
    return casted_global_attr


def cast_globals_to_edges(global_attr, edge_index=None, batch=None, num_edges=None):
    if batch is not None:
        assert edge_index is not None, "edge index should be specified"
        edge_counts = get_edge_counts(edge_index, batch)
        graph_indices = torch.unique(batch)
        casted_global_attr = [global_attr[idx, :] for idx, count in zip(graph_indices, edge_counts) for _ in range(count)]
    else:
        assert global_attr.size(0) == 1, "batch numbers should be provided."
        assert num_edges is not None, "number of edges should be specified"
        casted_global_attr = [global_attr] * num_edges
    casted_global_attr = torch.cat(casted_global_attr, dim=0)
    casted_global_attr = casted_global_attr.view(-1, global_attr.size(-1))
    return casted_global_attr


def cast_edges_to_globals(edge_attr, edge_index=None, batch=None, num_edges=None, num_globals=None):
    if batch is None:
        edge_attr_aggr = torch.sum(edge_attr, dim=0, keepdim=True)
    else:
        node_indices = torch.unique(batch)
        edge_counts = get_edge_counts(edge_index, batch)
        assert sum(edge_counts) == num_edges
        indices = [idx.view(1, 1) for idx, count in zip(node_indices, edge_counts) for _ in range(count)]
        indices = torch.cat(indices)
        edge_attr_aggr = scatter_sum(edge_attr, index=indices, dim=0, dim_size=num_globals)
    return edge_attr_aggr


def cast_nodes_to_globals(node_attr, batch=None, num_globals=None):
    if batch is None:
        x_aggr = torch.sum(node_attr, dim=0, keepdim=True)
    else:
        x_aggr = scatter_sum(node_attr, index=batch, dim=0, dim_size=num_globals)
    return x_aggr


def cast_edges_to_nodes(edge_attr, indices, num_nodes=None):
    edge_attr_aggr = scatter_sum(edge_attr, indices, dim=0, dim_size=num_nodes)
    return edge_attr_aggr

#+++++++++++++++++++++++++#
## block models: simple ###
#+++++++++++++++++++++++++#
class IndependentEdgeModel(torch.nn.Module):
    def __init__(self,
                 n_edge_feats_in,  # number of input edge features
                 n_edge_feats_out,  # number of output edge features
                 latent_size=128,  # latent size of mlp
                 activate_final=True,  # use activate for the last layer or not?
                 normalize=True  # batch normalize the output
                 ):
        super(IndependentEdgeModel, self).__init__()
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

    def forward(self, node_attr):
        return self.node_mlp(node_attr)


class IndependentGlobalModel(torch.nn.Module):
    def __init__(self,
                 n_global_in,
                 n_global_out,
                 latent_size=128,
                 activate_final=True,
                 normalize=True
                 ):

        super(IndependentGlobalModel, self).__init__()
        self.params = [n_global_in, n_global_out]
        self.global_mlp = make_mlp_model(n_global_in,
                                         latent_size,
                                         n_global_out,
                                         activate_final=activate_final,
                                         normalize=normalize  # careful: batch normalization does not work when
                                                              # batch size = 1;
                                                              # https://github.com/pytorch/pytorch/issues/7716
                                         )

    def forward(self, global_attr):
        return self.global_mlp(global_attr)


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

    def forward(self, receiver, sender, edge_attr, global_attr):
        # row, col = data.edge_index
        # sender, receiver = data.x[row, :], data.x[col, :]
        # edge_attr = data.edge_attr
        # global_attr = cast_globals_to_edges(data)
        out = torch.cat([receiver, sender, edge_attr, global_attr], dim=1)
        return self.edge_mlp(out)


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

    def forward(self, x, global_attr, recv_edge_attr_agg, send_edge_attr_agg):
        # row, col = data.edge_index
        # out = [data.x]
        # if not self.senders_turned_off:
        #     send_edge_attr_agg = cast_edges_to_nodes(data, row, aggr_func=self.agg_func)
        #     out.append(send_edge_attr_agg)
        # recv_edge_attr_agg = cast_edges_to_nodes(data, col, aggr_func=self.agg_func)
        # out.append(recv_edge_attr_agg)
        # global_attr = cast_globals_to_nodes(data)
        # out.append(global_attr)
        # out = torch.cat(out, dim=1)
        # data.x = self.node_mlp(out)
        if self.senders_turned_off:
            out = torch.cat([x, recv_edge_attr_agg, global_attr], dim=1)
        else:
            out = torch.cat([x, send_edge_attr_agg, recv_edge_attr_agg, global_attr], dim=1)
        return self.node_mlp(out)


class GlobalModel(torch.nn.Module):
    def __init__(self,
                 n_global_in,
                 n_global_out,
                 n_node_feats,
                 n_edge_feats,
                 latent_size=128,
                 activate_final=True,
                 normalize=True,
                 ):

        super(GlobalModel, self).__init__()
        self.params = [n_global_in, n_global_out, n_node_feats, n_edge_feats]
        self.global_mlp = make_mlp_model(n_global_in + n_edge_feats + n_node_feats,
                                         latent_size,
                                         n_global_out,
                                         activate_final=activate_final,
                                         normalize=normalize  # careful: batch normalization does not work when
                                                              # batch size = 1;
                                                              # https://github.com/pytorch/pytorch/issues/7716
                                         )

    def forward(self, x_aggr, edge_attr_aggr, global_attr):
        # x_aggr = cast_nodes_to_globals(data)
        # edge_attr_aggr = cast_edges_to_globals(data)
        # global_attr = data.u
        out = torch.cat([x_aggr, edge_attr_aggr, global_attr], dim=1)
        return self.global_mlp(out)



