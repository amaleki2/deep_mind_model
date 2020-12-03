import torch
import torch.nn
from graph_base_models import (IndependentEdgeMode, IndependentNodeModel, IndependentGlobalModel,
                               NodeModel, EdgeModel, GlobalModel, NodeModel_GN, GlobalModel_GN)


class GraphNetworkMetaLayer(torch.nn.Module):
    def __init__(self,
                 n_edge_feat_in, n_edge_feat_out,
                 n_node_feat_in, n_node_feat_out,
                 n_global_feat_in, n_global_feat_out,
                 latent_size=128,
                 activate_final=False,
                 normalize=True):
        super(GraphNetworkMetaLayer, self).__init__()
        self.edge_model = EdgeModel(n_edge_feat_in, n_edge_feat_out, n_node_feat_in, n_global_feat_in,
                                    latent_size=latent_size, activate_final=activate_final, normalize=normalize)
        self.node_model = NodeModel(n_node_feat_in, n_node_feat_out, n_edge_feat_out, n_global_feat_in,
                                    latent_size=latent_size, activate_final=activate_final, normalize=normalize)
        self.global_model = GlobalModel(n_global_feat_in, n_global_feat_out, n_node_feat_out,
                                        latent_size=latent_size, activate_final=activate_final)
        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.node_model, self.edge_model, self.global_model]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(self, x, edge_index, edge_attr, global_attr):
        """"""
        row, col = edge_index
        edge_attr = self.edge_model(x[row], x[col], edge_attr, global_attr)
        x = self.node_model(x, edge_index, edge_attr, global_attr)
        global_attr = self.global_model(x, global_attr)
        return x, edge_attr, global_attr


class GraphNetworkIndependentLayer(torch.nn.Module):
    def __init__(self,
                 n_edge_feat_in, n_edge_feat_out,
                 n_node_feat_in, n_node_feat_out,
                 n_global_feat_in, n_global_feat_out,
                 latent_size=128,
                 activate_final=False,
                 normalize=True):
        super(GraphNetworkIndependentLayer, self).__init__()
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

    def forward(self, x, edge_index, edge_attr, global_attr):
        """"""
        edge_attr = self.edge_model(edge_attr)
        x = self.node_model(x)
        global_attr = self.global_model(global_attr)
        return x, edge_attr, global_attr


class GraphNetworkGNLayer(torch.nn.Module):
    def __init__(self,
                 n_edge_feat_in, n_edge_feat_out,
                 n_node_feat_in, n_node_feat_out,
                 n_global_feat_in, n_global_feat_out,
                 latent_size=128,
                 activate_final=False,
                 normalize=True):
        super(GraphNetworkGNLayer, self).__init__()
        self.edge_model = EdgeModel(n_edge_feat_in, n_edge_feat_out, n_node_feat_in, n_global_feat_in,
                                    latent_size=latent_size, activate_final=activate_final, normalize=normalize)
        self.node_model = NodeModel_GN(n_node_feat_in, n_node_feat_out, n_edge_feat_out, n_global_feat_in,
                                       latent_size=latent_size, activate_final=activate_final, normalize=normalize)
        self.global_model = GlobalModel_GN(n_global_feat_in, n_global_feat_out, n_node_feat_out, n_edge_feat_out,
                                           latent_size=latent_size, activate_final=activate_final)
        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.node_model, self.edge_model, self.global_model]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(self, x, edge_index, edge_attr, global_attr):
        """"""
        row, col = edge_index
        edge_attr = self.edge_model(x[row], x[col], edge_attr, global_attr)
        x = self.node_model(x, edge_index, edge_attr, global_attr)
        global_attr = self.global_model(x, edge_attr, global_attr)
        return x, edge_attr, global_attr


class EncodeProcessDecode(torch.nn.Module):
    def __init__(self,
                 n_edge_feat_in=None, n_node_feat_in=None, n_global_feat_in=None,
                 n_edge_feat_mid=None, n_node_feat_mid=None, n_global_feat_mid=None,
                 n_edge_feat_out=None, n_node_feat_out=None, n_global_feat_out=None,
                 mlp_latent_size=128, num_processing_steps=5, full_output=False,
                 graph_layer=GraphNetworkMetaLayer):
        super(EncodeProcessDecode, self).__init__()
        assert not (n_edge_feat_in is None or n_node_feat_in is None or n_global_feat_in is None), \
            "input sizes should be specified"
        self.num_processing_steps = num_processing_steps
        self.full_output = full_output
        self.graph_layer = graph_layer
        n_edge_feat_mid = n_edge_feat_in if n_edge_feat_mid is None else n_edge_feat_mid
        n_edge_feat_out = n_edge_feat_in if n_edge_feat_out is None else n_edge_feat_out
        n_node_feat_mid = n_node_feat_in if n_node_feat_mid is None else n_node_feat_mid
        n_node_feat_out = n_node_feat_in if n_node_feat_out is None else n_node_feat_out
        n_global_feat_mid = n_global_feat_in if n_global_feat_mid is None else n_global_feat_mid
        n_global_feat_out = n_global_feat_in if n_global_feat_out is None else n_global_feat_out

        self.encoder = self.graph_layer(n_edge_feat_in, n_edge_feat_mid,
                                        n_node_feat_in, n_node_feat_mid,
                                        n_global_feat_in, n_global_feat_mid,
                                        latent_size=mlp_latent_size,
                                        activate_final=False)
        self.core = self.graph_layer(2 * n_edge_feat_mid, n_edge_feat_mid,
                                     2 * n_node_feat_mid, n_node_feat_mid,
                                     2 * n_global_feat_mid, n_global_feat_mid,
                                     latent_size=mlp_latent_size,
                                     activate_final=False)
        self.decoder = self.graph_layer(n_edge_feat_mid, n_edge_feat_out,
                                        n_node_feat_mid, n_node_feat_out,
                                        n_global_feat_mid, n_global_feat_out,
                                        latent_size=mlp_latent_size,
                                        activate_final=False)

    def forward(self, x, edge_index, edge_attr, global_attr):
        x, edge_attr, global_attr = self.encoder(x, edge_index, edge_attr, global_attr)
        x0 = x.clone()
        edge_attr0 = edge_attr.clone()
        global_attr0 = global_attr.clone()
        output_ops = []
        for _ in range(self.num_processing_steps):
            core_x = torch.cat([x0, x], dim=1)
            core_edge_attr = torch.cat([edge_attr0, edge_attr], dim=1)
            core_global_attr = torch.cat([global_attr0, global_attr], dim=1)
            x, edge_attr, global_attr = self.core(core_x, edge_index, core_edge_attr, core_global_attr)
            decoded_op = self.decoder(x, edge_index, edge_attr, global_attr)
            output_ops.append(decoded_op)
        if self.full_output:
            return output_ops
        else:
            return output_ops[-1]


class EncodeProcessDecode2(torch.nn.Module):
    def __init__(self,
                 n_edge_feat_in=None, n_node_feat_in=None, n_global_feat_in=None,
                 n_edge_feat_out=None, n_node_feat_out=None, n_global_feat_out=None,
                 mlp_latent_size=128, num_processing_steps=5, full_output=False,
                 graph_layer=GraphNetworkMetaLayer):
        super(EncodeProcessDecode2, self).__init__()
        assert not (n_edge_feat_in is None or n_node_feat_in is None or n_global_feat_in is None), \
            "input sizes should be specified"
        self.num_processing_steps = num_processing_steps
        self.full_output = full_output
        self.graph_layer = graph_layer
        n_edge_feat_out = mlp_latent_size if n_edge_feat_out is None else n_edge_feat_out
        n_node_feat_out = mlp_latent_size if n_node_feat_out is None else n_node_feat_out
        n_global_feat_out = mlp_latent_size if n_global_feat_out is None else n_global_feat_out

        self.encoder = self.graph_layer(n_edge_feat_in, mlp_latent_size,
                                        n_node_feat_in, mlp_latent_size,
                                        n_global_feat_in, mlp_latent_size,
                                        latent_size=mlp_latent_size,
                                        activate_final=True)

        self.core = self.graph_layer(2 * mlp_latent_size, mlp_latent_size,
                                     2 * mlp_latent_size, mlp_latent_size,
                                     2 * mlp_latent_size, mlp_latent_size,
                                     latent_size=mlp_latent_size,
                                     activate_final=True)

        self.decoder = self.graph_layer(mlp_latent_size, mlp_latent_size,
                                        mlp_latent_size, mlp_latent_size,
                                        mlp_latent_size, mlp_latent_size,
                                        latent_size=mlp_latent_size,
                                        activate_final=True)

        self.output_transformer = self.graph_layer(mlp_latent_size, n_edge_feat_out,
                                                   mlp_latent_size, n_node_feat_out,
                                                   mlp_latent_size, n_global_feat_out,
                                                   latent_size=mlp_latent_size // 3 + 1,
                                                   activate_final=False, normalize=False)

    def forward(self, x, edge_index, edge_attr, global_attr):
        x, edge_attr, global_attr = self.encoder(x, edge_index, edge_attr, global_attr)
        x0 = x.clone()
        edge_attr0 = edge_attr.clone()
        global_attr0 = global_attr.clone()
        output_ops = []
        for _ in range(self.num_processing_steps):
            core_x = torch.cat([x0, x], dim=1)
            core_edge_attr = torch.cat([edge_attr0, edge_attr], dim=1)
            core_global_attr = torch.cat([global_attr0, global_attr], dim=1)
            x, edge_attr, global_attr = self.core(core_x, edge_index, core_edge_attr, core_global_attr)
            x, edge_attr, global_attr = self.decoder(x, edge_index, edge_attr, global_attr)
            output_ops.append(self.output_transformer(x, edge_index, edge_attr, global_attr))

        if self.full_output:
            return output_ops
        else:
            return output_ops[-1]


class EncodeProcessDecode3(torch.nn.Module):
    def __init__(self,
                 n_edge_feat_in=None, n_node_feat_in=None, n_global_feat_in=None,
                 n_edge_feat_out=None, n_node_feat_out=None, n_global_feat_out=None,
                 mlp_latent_size=128, num_processing_steps=5, full_output=False,
                 graph_simple_layer=GraphNetworkIndependentLayer, graph_complex_layer=GraphNetworkGNLayer):
        super(EncodeProcessDecode3, self).__init__()
        assert not (n_edge_feat_in is None or n_node_feat_in is None or n_global_feat_in is None), \
            "input sizes should be specified"
        self.num_processing_steps = num_processing_steps
        self.full_output = full_output
        self.graph_simple_layer = graph_simple_layer
        self.graph_complex_layer = graph_complex_layer
        n_edge_feat_out = mlp_latent_size if n_edge_feat_out is None else n_edge_feat_out
        n_node_feat_out = mlp_latent_size if n_node_feat_out is None else n_node_feat_out
        n_global_feat_out = mlp_latent_size if n_global_feat_out is None else n_global_feat_out

        self.encoder = self.graph_simple_layer(n_edge_feat_in, mlp_latent_size,
                                               n_node_feat_in, mlp_latent_size,
                                               n_global_feat_in, mlp_latent_size,
                                               latent_size=mlp_latent_size,
                                               activate_final=True)

        self.core = self.graph_complex_layer(2 * mlp_latent_size, mlp_latent_size,
                                             2 * mlp_latent_size, mlp_latent_size,
                                             2 * mlp_latent_size, mlp_latent_size,
                                             latent_size=mlp_latent_size,
                                             activate_final=True)

        self.decoder = self.graph_simple_layer(mlp_latent_size, mlp_latent_size,
                                               mlp_latent_size, mlp_latent_size,
                                               mlp_latent_size, mlp_latent_size,
                                               latent_size=mlp_latent_size,
                                               activate_final=True)

        self.output_transformer = self.graph_simple_layer(mlp_latent_size, n_edge_feat_out,
                                                          mlp_latent_size, n_node_feat_out,
                                                          mlp_latent_size, n_global_feat_out,
                                                          latent_size=mlp_latent_size // 2 + 1,
                                                          activate_final=False, normalize=False)

    def forward(self, x, edge_index, edge_attr, global_attr):
        x, edge_attr, global_attr = self.encoder(x, edge_index, edge_attr, global_attr)
        x0 = x.clone()
        edge_attr0 = edge_attr.clone()
        global_attr0 = global_attr.clone()
        output_ops = []
        for _ in range(self.num_processing_steps):
            core_x = torch.cat([x0, x], dim=1)
            core_edge_attr = torch.cat([edge_attr0, edge_attr], dim=1)
            core_global_attr = torch.cat([global_attr0, global_attr], dim=1)
            x, edge_attr, global_attr = self.core(core_x, edge_index, core_edge_attr, core_global_attr)
            x, edge_attr, global_attr = self.decoder(x, edge_index, edge_attr, global_attr)
            output_ops.append(self.output_transformer(x, edge_index, edge_attr, global_attr))

        if self.full_output:
            return output_ops
        else:
            return output_ops[-1]


if __name__ == "__main__":
    x = torch.rand(10, 3)
    edge_index = torch.randint(0, 9, (2, 40))
    edge_attr = torch.rand(40, 5)
    global_attr = torch.rand(1, 2)

    net = EncodeProcessDecode(5, 3, 2)
    out = net(x, edge_index, edge_attr, global_attr)
    print(out)
