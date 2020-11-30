import os
import sys
import torch
import numpy as np
from torch_geometric.data import Data, DataLoader

from graph_networks import GraphNetworkMetaLayer, GraphNetworkGNLayer, EncodeProcessDecode
from train_graph import train_sdf


def get_sdf_data_loader(n_objects, data_folder, batch_size, eval_frac=0.2,
                        reversed_edge_already_included=False, edge_weight=False):

    print("preparing sdf data loader")
    random_idx = np.random.permutation(n_objects)
    train_idx  = random_idx[:int((1 - eval_frac) * n_objects)]
    test_idx   = random_idx[int((1 - eval_frac) * n_objects):]

    train_graph_data_list = []
    test_graph_data_list = []

    for idx, graph_data_list in zip([train_idx, test_idx], [train_graph_data_list, test_graph_data_list]):
        for i in idx:
            graph_nodes = np.load(data_folder + "graph_nodes%d.npy" % i).astype(float)
            x = graph_nodes.copy()
            if np.ndim(x) == 2:
                x[:, 2] = (x[:, 2] < 0).astype(float)
                y = graph_nodes.copy()[:, 2]
                y = y.reshape(-1, 1)
            else:  # np.ndim(x) == 3
                x[:, :, 2] = (x[:, :, 2] < 0).astype(float)
                x = x.reshape(x.shape[0], -1)
                y = graph_nodes.copy()[:, :, 2]
                y = np.mean(y, axis=-1, keepdims=True)

            graph_cells = np.load(data_folder + "graph_cells%d.npy" % i).astype(int)
            graph_cells = graph_cells.T
            graph_edges = np.load(data_folder + "graph_edges%d.npy" % i).astype(int)
            if not reversed_edge_already_included:
                graph_edges = add_reversed_edges(graph_edges)
            graph_edges = graph_edges.T
            n_edges = graph_edges.shape[1]
            if edge_weight:
                graph_edge_weights = np.load(data_folder + "graph_weights%d.npy" %i).astype(float)
                if graph_edge_weights.shape[0] == n_edges:
                    pass
                elif graph_edge_weights.shape[0] == n_edges // 2:
                    graph_edge_weights = np.concatenate([graph_edge_weights, graph_edge_weights])
                else:
                    raise("edge weight size is wrong.")
                #graph_edge_weights = graph_edge_weights.reshape(-1, 1)
            else:
                graph_edge_weights = np.ones(n_edges)
            graph_edge_weights = graph_edge_weights.reshape(-1, 1)

            graph_global = np.mean(x[x[:, 2] == 1, :2], axis=0, keepdims=True)
            graph_data = Data(x=torch.from_numpy(x).type(torch.float32),
                              y=torch.from_numpy(y).type(torch.float32),
                              u=torch.from_numpy(graph_global).type(torch.float32),
                              edge_index=torch.from_numpy(graph_edges).type(torch.long),
                              edge_attr=torch.from_numpy(graph_edge_weights).type(torch.float32),
                              face=torch.from_numpy(graph_cells).type(torch.long))
            graph_data_list.append(graph_data)
    train_data = DataLoader(train_graph_data_list, batch_size=batch_size)
    test_data = DataLoader(test_graph_data_list, batch_size=batch_size)
    return train_data, test_data


def add_reversed_edges(edges):
    edges_reversed = np.fliplr(edges)
    edges = np.concatenate([edges, edges_reversed], axis=0)
    return edges


assert len(sys.argv) == 2
data_folder = sys.argv[1]
edge_weight = True

n_objects, batch_size, n_epoch = 25, 1, 1500
lr_0, step_size, gamma, radius = 0.001, 200, 0.6, 0.1

train_data, test_data = get_sdf_data_loader(n_objects, data_folder, batch_size, edge_weight=edge_weight)

n_edge_feat_in, n_edge_feat_out = 1, 1
n_node_feat_in, n_node_feat_out = 3, 1
n_global_feat = 2

model = GraphNetworkMetaLayer(n_edge_feat_in, n_edge_feat_out,
                              n_node_feat_in, n_node_feat_out,
                              n_global_feat, n_global_feat,
                              latent_size=128,
                              activate_final=False)

l1_loss = torch.nn.L1Loss()

train_sdf(model, train_data, loss_func=l1_loss, use_cpu=True, n_epoch=n_epoch)

# plot_results(model, train_data, ndata=5, levels=[-0.2, 0, 0.2, 0.4], border=0.1, save_name="test")
# plot_results_over_line(model, train_data, ndata=5, save_name="test")