import torch
import torch.nn
from graph_base_models import (IndependentEdgeMode, IndependentNodeModel, IndependentGlobalModel,
                               NodeModel, EdgeModel, GlobalModel)
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

    def forward(self, data):
        return self.global_model(self.node_model(self.edge_model(data)))


class GraphNetworkIndependentBlock(torch.nn.Module):
    def __init__(self,
                 n_edge_feat_in, n_edge_feat_out,
                 n_node_feat_in, n_node_feat_out,
                 n_global_feat_in, n_global_feat_out,
                 latent_size=128,
                 activate_final=True,
                 normalize=True):
        super(GraphNetworkIndependentBlock, self).__init__()
        self.edge_model = IndependentEdgeMode(n_edge_feat_in, n_edge_feat_out, latent_size=latent_size,
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

    def forward(self, data):
        return self.global_model(self.node_model(self.edge_model(data)))


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

    def forward(self, data):
        data = self.encoder(data)
        data0 = data.clone()
        output_ops = []
        for _ in range(self.num_processing_steps):
            data = concat_graph_data(data0, data)
            data = self.processor(data)
            decoder_op = self.decoder(data.clone())
            output_ops.append(self.output_transformer(decoder_op))

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
