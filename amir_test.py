import numpy as np
import torch
from graph_base_models import make_mlp_model
from torch_geometric.data import Data, Batch, DataLoader
from graph_networks import GraphNetworkIndependentBlock, GraphNetworkBlock, EncodeProcessDecode
from graph_base_models import EdgeModel, NodeModel, GlobalModel


# test base mlp
m1 = make_mlp_model(3, 4, 4, activate_final=True, normalize=False, initializer=True)
x = np.array([[1.0000,  0.5000, -1.2000], [0.0000, -1.5000,  2.5000]])
x = torch.from_numpy(x).type(torch.float32)
o = m1(x)

# test independent mlp graphs
nodes = np.array([[2., 3.],
                  [3., 3.5],
                  [-1., -2.],
                  [0, -1.]])
nodes = torch.from_numpy(nodes).type(torch.float32)

edges = np.array([[3, 4., 5.],
                  [5, 6, -1.],
                  [-2., -3, -1.],
                  [2, 3, 1.]])
edges = torch.from_numpy(edges).type(torch.float32)

senders = np.array([0, 3, 1, 2])
receivers = np.array([1, 1, 2, 1])
edge_index = np.array([senders, receivers])
edge_index = torch.from_numpy(edge_index).type(torch.long)

globals = np.array([[1., -1, 0, 4.]])
globals = torch.from_numpy(globals).type(torch.float32)

data = Data(x=nodes, edge_index=edge_index, edge_attr=edges, u=globals)
# data_loader = DataLoader([data], batch_size=1)
# batch = next(iter(data_loader))

nodes = np.array([[-4., -3.],
                  [-3., -1.],
                  [0., 1.]])
nodes = torch.from_numpy(nodes).type(torch.float32)

edges = np.array([[1., 1., -1.],
                  [2., 2., 0.],
                  [0., 0., -1.],
                  [1., 0., -2.]
                  ])
edges = torch.from_numpy(edges).type(torch.float32)

senders = np.array([1, 0, 1, 2])
receivers = np.array([0, 1, 2, 0])
edge_index = np.array([senders, receivers])
edge_index = torch.from_numpy(edge_index).type(torch.long)

globals = np.array([[-2., 0.5, 3., 2.]])
globals = torch.from_numpy(globals).type(torch.float32)

data2 = Data(x=nodes, edge_index=edge_index, edge_attr=edges, u=globals)
data_loader = DataLoader([data, data2], batch_size=2)
batch = next(iter(data_loader))


# model = GraphNetworkIndependentBlock(3, 4,
#                                      2, 4,
#                                      4, 4,
#                                      latent_size=4,
#                                      activate_final=True,
#                                      normalize=False)
#
# output = model(data)
# print(output.edge_attr)
# print(output.x)
# print(output.u)

# complex models
# model = EdgeModel(3,   # number of input edge features
#                   4,   # number of output edge features
#                   2,   # number of input node features
#                   4,   # number of global (graph) features
#                   latent_size=4,    # latent size of mlp
#                   activate_final=True,  # use activate for the last layer or not?
#                   normalize=False)
#
# output = model(batch)
# print(output.edge_attr)
# print(output.x)
# print(output.u)

# model = NodeModel(2,   # number of input node features
#                   4,   # number of output node features
#                   3,   # number of input edge features
#                   4,   # number of global (graph) features
#                   latent_size=4,    # latent size of mlp
#                   activate_final=True,  # use activate for the last layer or not?
#                   normalize=False)
# output = model(batch)
# print(output.edge_attr)
# print(output.x)
# print(output.u)


# model = GlobalModel(4,
#                     4,
#                     2,
#                     3,
#                     latent_size=4,
#                     activate_final=False)
# output = model(batch)
# print(output.edge_attr)
# print(output.x)
# print(output.u)

model = GraphNetworkBlock(3, 4,
                          2, 4,
                          4, 4,
                          latent_size=4,
                          activate_final=True,
                          normalize=False)
output = model(batch)
print(output.edge_attr)
print(output.x)
print(output.u)

# model = EncodeProcessDecode(n_edge_feat_in=3, n_node_feat_in=2, n_global_feat_in=4,
#                             n_edge_feat_out=3, n_node_feat_out=2, n_global_feat_out=1,
#                             mlp_latent_size=4, num_processing_steps=5, full_output=True,
#                             encoder=GraphNetworkIndependentBlock, processor=GraphNetworkBlock,
#                             decoder=GraphNetworkIndependentBlock, output_transformer=GraphNetworkIndependentBlock)
# output = model(batch)
# print(output.edge_attr)
# print(output.x)
# print(output.u)



