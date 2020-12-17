from abc import ABC

import torch.nn
from graph_base_models import *
from torch_geometric.data import Batch


class GraphNetworkBlock(torch.nn.Module, ABC):
    def __init__(self,
                 n_edge_feat_in, n_edge_feat_out,
                 n_node_feat_in, n_node_feat_out,
                 n_global_feat_in, n_global_feat_out,
                 latent_size=128,
                 activate_final=True,
                 normalize=True):
        super(GraphNetworkBlock, self).__init__()
        self.params = [n_edge_feat_in, n_edge_feat_out,
                       n_node_feat_in, n_node_feat_out,
                       n_global_feat_in, n_global_feat_out]

        if n_edge_feat_out is not None:
            self.edge_model = EdgeModel(n_edge_feat_in, n_edge_feat_out, n_node_feat_in, n_global_feat_in,
                                        latent_size=latent_size, activate_final=activate_final, normalize=normalize)
        if n_node_feat_out is not None:
            self.node_model = NodeModel(n_node_feat_in, n_node_feat_out, n_edge_feat_out, n_global_feat_in,
                                        latent_size=latent_size, activate_final=activate_final, normalize=normalize)
        if n_global_feat_out is not None:
            self.global_model = GlobalModel(n_global_feat_in, n_global_feat_out, n_node_feat_out, n_edge_feat_out,
                                            latent_size=latent_size, activate_final=activate_final, normalize=normalize)
        self.reset_parameters()

    def reset_parameters(self):
        for item in ['node_model', 'edge_model', 'global_model']:
            if hasattr(self, item):
                model = getattr(self, item)
                if hasattr(model, 'reset_parameters'):
                    model.reset_parameters()

    def forward(self, edge_attr, node_attr, global_attr, edge_index, batch):
        [_, n_edge_feat_out,
         _, n_node_feat_out,
         _, n_global_feat_out] = self.params
        row, col = edge_index
        num_edges, num_nodes, num_globals = edge_attr.size(0), node_attr.size(0), global_attr.size(0)

        # update edge attr
        if n_edge_feat_out is None:
            edge_attr_new = edge_attr
        else:
            sender_attr, receiver_attr = node_attr[row, :], node_attr[col, :]
            global_attr_to_edge = cast_globals_to_edges(global_attr, edge_index=edge_index, batch=batch, num_edges=num_edges)
            edge_attr_new = self.edge_model(receiver_attr, sender_attr, edge_attr, global_attr_to_edge)

        # update node attr
        if n_node_feat_out is None:
            node_attr_new = node_attr
        else:
            global_attr_to_nodes = cast_globals_to_nodes(global_attr, batch=batch, num_nodes=num_nodes)
            sender_attr_to_node = cast_edges_to_nodes(edge_attr_new, row, num_nodes=num_nodes)
            receiver_attr_to_node = cast_edges_to_nodes(edge_attr_new, col, num_nodes=num_nodes)
            node_attr_new = self.node_model(node_attr, global_attr_to_nodes, receiver_attr_to_node, sender_attr_to_node)

        # update global attr
        if n_global_feat_out is None:
            global_attr_new = global_attr
        else:
            node_attr_to_global = cast_nodes_to_globals(node_attr_new, batch=batch, num_globals=num_globals)
            edge_attr_to_global = cast_edges_to_globals(edge_attr_new, edge_index=edge_index, batch=batch,
                                                        num_edges=num_edges, num_globals=num_globals)
            global_attr_new = self.global_model(node_attr_to_global, edge_attr_to_global, global_attr)
        return edge_attr_new, node_attr_new, global_attr_new


class GraphNetworkIndependentBlock(torch.nn.Module):
    def __init__(self,
                 n_edge_feat_in, n_edge_feat_out,
                 n_node_feat_in, n_node_feat_out,
                 n_global_feat_in, n_global_feat_out,
                 latent_size=128,
                 activate_final=True,
                 normalize=True):
        super(GraphNetworkIndependentBlock, self).__init__()
        if n_edge_feat_out is None:
            self.edge_model = lambda xx: xx
        else:
            self.edge_model = IndependentEdgeModel(n_edge_feat_in, n_edge_feat_out, latent_size=latent_size,
                                                   activate_final=activate_final, normalize=normalize)
        if n_node_feat_out is None:
            self.node_model = lambda xx: xx
        else:
            self.node_model = IndependentNodeModel(n_node_feat_in, n_node_feat_out, latent_size=latent_size,
                                                   activate_final=activate_final, normalize=normalize)
        if n_global_feat_out is None:
            self.global_model = lambda xx: xx
        else:
            self.global_model = IndependentGlobalModel(n_global_feat_in, n_global_feat_out, latent_size=latent_size,
                                                       activate_final=activate_final, normalize=normalize)
        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.node_model, self.edge_model, self.global_model]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(self, edge_attr, node_attr, global_attr, edge_index, batch):
        # signature should be similar to GraphNetworkBlock
        return self.edge_model(edge_attr), self.node_model(node_attr), self.global_model(global_attr)


class EncodeProcessDecode(torch.nn.Module):
    def __init__(self,
                 n_edge_feat_in=None, n_node_feat_in=None, n_global_feat_in=None,
                 n_edge_feat_out=None, n_node_feat_out=None, n_global_feat_out=None,
                 mlp_latent_size=128, num_processing_steps=5, full_output=False,
                 normalize=True,
                 encoder=GraphNetworkBlock, processor=GraphNetworkBlock,
                 decoder=GraphNetworkBlock, output_transformer=GraphNetworkBlock):
        super(EncodeProcessDecode, self).__init__()
        assert not (n_edge_feat_in is None or n_node_feat_in is None or n_global_feat_in is None), \
            "input sizes should be specified"
        self.num_processing_steps = num_processing_steps
        self.full_output = full_output

        self.encoder = encoder(n_edge_feat_in, mlp_latent_size,
                               n_node_feat_in, mlp_latent_size,
                               n_global_feat_in, mlp_latent_size,
                               latent_size=mlp_latent_size,
                               activate_final=True,
                               normalize=normalize)

        self.processor = processor(2 * mlp_latent_size, mlp_latent_size,
                                   2 * mlp_latent_size, mlp_latent_size,
                                   2 * mlp_latent_size, mlp_latent_size,
                                   latent_size=mlp_latent_size,
                                   activate_final=True,
                                   normalize=normalize)

        self.decoder = decoder(mlp_latent_size, mlp_latent_size,
                               mlp_latent_size, mlp_latent_size,
                               mlp_latent_size, mlp_latent_size,
                               latent_size=mlp_latent_size,
                               activate_final=True,
                               normalize=normalize)

        self.output_transformer = output_transformer(mlp_latent_size, n_edge_feat_out,
                                                     mlp_latent_size, n_node_feat_out,
                                                     mlp_latent_size, n_global_feat_out,
                                                     latent_size=None,
                                                     activate_final=False, normalize=False)

    def forward(self, data):
        edge_attr, edge_index, node_attr, global_attr, batch = data.edge_attr, data.edge_index, data.x, data.u, data.batch
        edge_attr, node_attr, global_attr = self.encoder(edge_attr, node_attr, global_attr, edge_index, batch)
        edge_attr0, node_attr0, global_attr0 = edge_attr.clone(), node_attr.clone(), global_attr.clone()
        output_ops = []
        for _ in range(self.num_processing_steps):
            edge_attr = torch.cat((edge_attr0, edge_attr), dim=1)
            node_attr = torch.cat((node_attr0, node_attr), dim=1)
            global_attr = torch.cat((global_attr0, global_attr), dim=1)
            edge_attr, node_attr, global_attr = self.processor(edge_attr, node_attr, global_attr, edge_index, batch)
            edge_attr_de, node_attr_de, global_attr_de = self.decoder(edge_attr, node_attr, global_attr, edge_index, batch)
            edge_attr_op, node_attr_op, global_attr_op = self.output_transformer(edge_attr_de, node_attr_de, global_attr_de, edge_index, batch)
            output_ops.append((edge_attr_op, node_attr_op, global_attr_op))

        if self.full_output:
            return output_ops
        else:
            return output_ops[-1]


class EncodeProcessDecodeNEW(torch.nn.Module):
    def __init__(self,
                 n_edge_feat_in=None, n_node_feat_in=None, n_global_feat_in=None,
                 n_edge_feat_out=None, n_node_feat_out=None, n_global_feat_out=None,
                 mlp_latent_size=128, num_processing_steps=5, full_output=False,
                 normalize=True,
                 encoder=GraphNetworkBlock, processor=GraphNetworkBlock,
                 decoder=GraphNetworkBlock, output_transformer=GraphNetworkBlock):
        super(EncodeProcessDecodeNEW, self).__init__()
        assert not (n_edge_feat_in is None or n_node_feat_in is None or n_global_feat_in is None), \
            "input sizes should be specified"
        self.num_processing_steps = num_processing_steps
        self.full_output = full_output

        self.encoder = encoder(n_edge_feat_in, mlp_latent_size,
                               n_node_feat_in, mlp_latent_size,
                               n_global_feat_in, mlp_latent_size,
                               latent_size=mlp_latent_size,
                               activate_final=True,
                               normalize=normalize)

        self.processors = torch.nn.ModuleList()
        for _ in range(num_processing_steps):
            self.processors.append(processor(2 * mlp_latent_size, mlp_latent_size,
                                             2 * mlp_latent_size, mlp_latent_size,
                                             2 * mlp_latent_size, mlp_latent_size,
                                             latent_size=mlp_latent_size,
                                             activate_final=True,
                                             normalize=normalize))

        self.decoder = decoder(mlp_latent_size, mlp_latent_size,
                               mlp_latent_size, mlp_latent_size,
                               mlp_latent_size, mlp_latent_size,
                               latent_size=mlp_latent_size,
                               activate_final=True,
                               normalize=normalize)

        self.output_transformer = output_transformer(mlp_latent_size, n_edge_feat_out,
                                                     mlp_latent_size, n_node_feat_out,
                                                     mlp_latent_size, n_global_feat_out,
                                                     latent_size=None,
                                                     activate_final=False, normalize=False)

    def forward(self, data):
        edge_attr, edge_index, node_attr, global_attr, batch = data.edge_attr, data.edge_index, data.x, data.u, data.batch
        edge_attr, node_attr, global_attr = self.encoder(edge_attr, node_attr, global_attr, edge_index, batch)
        edge_attr0, node_attr0, global_attr0 = edge_attr.clone(), node_attr.clone(), global_attr.clone()
        output_ops = []
        for i in range(self.num_processing_steps):
            edge_attr = torch.cat((edge_attr0, edge_attr), dim=1)
            node_attr = torch.cat((node_attr0, node_attr), dim=1)
            global_attr = torch.cat((global_attr0, global_attr), dim=1)
            edge_attr, node_attr, global_attr = self.processors[i](edge_attr, node_attr, global_attr, edge_index, batch)
            edge_attr_de, node_attr_de, global_attr_de = self.decoder(edge_attr, node_attr, global_attr, edge_index, batch)
            edge_attr_op, node_attr_op, global_attr_op = self.output_transformer(edge_attr_de, node_attr_de, global_attr_de, edge_index, batch)
            output_ops.append((edge_attr_op, node_attr_op, global_attr_op))

        if self.full_output:
            return output_ops
        else:
            return output_ops[-1]


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(3, 3).type(torch.float32)

    torch.manual_seed(0)
    edge_index = torch.randint(0, 3, (2, 3)).type(torch.long)

    torch.manual_seed(0)
    edge_attr = torch.rand(3, 5).type(torch.float32)

    torch.manual_seed(0)
    global_attr = torch.rand(1, 2).type(torch.float32)

    data = Batch(x=x, edge_index=edge_index, edge_attr=edge_attr, u=global_attr)
    net = EncodeProcessDecodeNEW(n_edge_feat_in=5, n_node_feat_in=3, n_global_feat_in=2,
                                 mlp_latent_size=2, num_processing_steps=5, full_output=True)
    out = net(data)
    print(out[-1])
