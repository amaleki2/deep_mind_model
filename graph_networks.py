import torch.nn
from graph_base_models import *
from torch_geometric.data import Batch


def concat_graph_data(data1, data2):
    # assert torch.all(data1.edge_index == data2.edge_index), "data should have identical edge index for concatenation"
    # assert torch.all(data1.batch == data2.batch), "data should have identical batch indices for concatenation"
    # assert data1.x.dtype == data2.x.dtype, "node features should have same data type for concatenation"
    # assert data1.edge_attr.dtype == data2.edge_attr.dtype, "edge features should have same data type for concatenation"
    # assert data1.u.dtype == data2.u.dtype, "global features should have same data type for concatenation"

    concat_x         = torch.cat([data1.x, data2.x], dim=1)
    concat_edge_attr = torch.cat([data1.edge_attr, data2.edge_attr], dim=1)
    concat_u         = torch.cat([data1.u, data2.u], dim=1)
    concat_data      = Batch(x=concat_x,
                             edge_index=data1.edge_index,
                             edge_attr=concat_edge_attr,
                             u=concat_u,
                             batch=data1.batch)
    return concat_data


class GraphNetworkBlock(torch.nn.Module):
    def __init__(self,
                 n_edge_feat_in, n_edge_feat_out,
                 n_node_feat_in, n_node_feat_out,
                 n_global_feat_in, n_global_feat_out,
                 latent_size=128,
                 activate_final=True,
                 normalize=True):
        super(GraphNetworkBlock, self).__init__()
        self.edge_model = EdgeModel(n_edge_feat_in, n_edge_feat_out, n_node_feat_in, n_global_feat_in,
                                    latent_size=latent_size, activate_final=activate_final, normalize=normalize)
        self.node_model = NodeModel(n_node_feat_in, n_node_feat_out, n_edge_feat_out, n_global_feat_in,
                                    latent_size=latent_size, activate_final=activate_final, normalize=normalize)
        self.global_model = GlobalModel(n_global_feat_in, n_global_feat_out, n_node_feat_out, n_edge_feat_out,
                                        latent_size=latent_size, activate_final=activate_final)
        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.node_model, self.edge_model, self.global_model]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(self, edge_attr, node_attr, global_attr,
                receiver_attr=None, sender_attr=None,
                global_attr_to_edge=None, global_attr_to_nodes=None,
                receiver_attr_to_nodes=None, sender_attr_to_node=None,
                node_attr_to_global=None, edge_attr_to_global=None
                ):
        edge_attr_new = self.edge_model(receiver_attr, sender_attr, edge_attr, global_attr_to_edge)
        node_attr_new = self.node_model(node_attr, global_attr_to_nodes, receiver_attr_to_nodes, sender_attr_to_node)
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
        self.edge_model = IndependentEdgeModel(n_edge_feat_in, n_edge_feat_out, latent_size=latent_size,
                                              activate_final=activate_final, normalize=normalize)
        self.node_model = IndependentNodeModel(n_node_feat_in, n_node_feat_out, latent_size=latent_size,
                                               activate_final=activate_final, normalize=normalize)
        self.global_model = IndependentGlobalModel(n_global_feat_in, n_global_feat_out, latent_size=latent_size,
                                                   activate_final=activate_final)
        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.node_model, self.edge_model, self.global_model]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(self, edge_attr, node_attr, global_attr, **kwargs):
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
        n_edge_feat_out = mlp_latent_size if n_edge_feat_out is None else n_edge_feat_out
        n_node_feat_out = mlp_latent_size if n_node_feat_out is None else n_node_feat_out
        n_global_feat_out = mlp_latent_size if n_global_feat_out is None else n_global_feat_out

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
                                                     latent_size=mlp_latent_size // 2 + 1,
                                                     activate_final=False, normalize=False)

    def get_all_casted(self, edge_attr, edge_index, node_attr, global_attr, batch, num_edges, num_nodes, num_globals):
        row, col = data.edge_index
        sender_attr, receiver_attr = node_attr[row, :], node_attr[col, :]
        global_attr_to_edge = cast_globals_to_edges(global_attr, edge_index=edge_index, batch=batch,
                                                    num_edges=num_edges)
        global_attr_to_nodes = cast_globals_to_nodes(global_attr, batch=batch, num_nodes=num_nodes)
        sender_attr_to_nodes = cast_edges_to_nodes(edge_attr, row, num_nodes=num_nodes)
        receiver_attr_to_node = cast_edges_to_nodes(edge_attr, col, num_nodes=num_nodes)
        node_attr_to_global = cast_nodes_to_globals(node_attr, batch=batch, num_globals=num_globals)
        edge_attr_to_global = cast_edges_to_globals(edge_attr, edge_index=edge_index, batch=batch,
                                                    num_edges=num_edges, num_globals=num_globals)
        all_cast_dict = {"receiver_attr": receiver_attr,
                         "sender_attr": sender_attr,
                         "global_attr_to_edge": global_attr_to_edge,
                         "global_attr_to_nodes": global_attr_to_nodes,
                         "receiver_attr_to_nodes": receiver_attr_to_node,
                         "sender_attr_to_node": sender_attr_to_nodes,
                         "node_attr_to_global": node_attr_to_global,
                         "edge_attr_to_global": edge_attr_to_global}
        return all_cast_dict

    def forward(self, data):
        edge_attr, edge_index, node_attr, global_attr, batch = data.edge_attr, data.edge_index, data.x, data.u, data.batch
        num_edges, num_nodes, num_globals = node_attr.size(0), node_attr.size(0), global_attr.size(0)
        all_cast_dict = self.get_all_casted(edge_attr, edge_index, node_attr, global_attr,
                                            batch, num_edges, num_nodes, num_globals)
        edge_attr, node_attr, global_attr = self.encoder(edge_attr, node_attr, global_attr, all_cast_dict)
        edge_attr0, node_attr0, global_attr0 = edge_attr.clone(), node_attr.clone(), global_attr.clone()
        output_ops = []
        for _ in range(self.num_processing_steps):
            edge_attr = torch.cat(edge_attr0, edge_attr)
            node_attr = torch.cat(node_attr0, node_attr)
            global_attr = torch.cat(global_attr0, global_attr)
            all_cast_dict = self.get_all_casted(edge_attr, edge_index, node_attr, global_attr,
                                                batch, num_edges, num_nodes, num_globals)
            edge_attr, node_attr, global_attr = self.processor(edge_attr, node_attr, global_attr, all_cast_dict)
            all_cast_dict = self.get_all_casted(edge_attr, edge_index, node_attr, global_attr,
                                                batch, num_edges, num_nodes, num_globals)
            edge_attr_de, node_attr_de, global_attr_de = self.decoder(edge_attr, node_attr, global_attr, all_cast_dict)
            all_cast_dict = self.get_all_casted(edge_attr_de, edge_index, node_attr_de, global_attr_de,
                                                batch, num_edges, num_nodes, num_globals)
            output_ops.append(self.output_transformer(edge_attr_de, node_attr_de, global_attr_de, all_cast_dict))

        if self.full_output:
            return output_ops
        else:
            return output_ops[-1]


if __name__ == "__main__":
    x = torch.rand(10, 3).type(torch.float32)
    edge_index = torch.randint(0, 9, (2, 40)).type(torch.long)
    edge_attr = torch.rand(40, 5).type(torch.float32)
    global_attr = torch.rand(1, 2).type(torch.float32)
    data = Batch(x=x, edge_index=edge_index, edge_attr=edge_attr, u=global_attr)
    net = EncodeProcessDecode(n_edge_feat_in=5, n_node_feat_in=3, n_global_feat_in=2,
                              mlp_latent_size=16, num_processing_steps=4, full_output=True)
    out = net(data)
    print(out)
