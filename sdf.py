import os
import sys
import torch
import numpy as np
import meshio
import matplotlib.pyplot as plt
import matplotlib.tri as tri

from torch_geometric.data import Data, DataLoader
from graph_networks import EncodeProcessDecode, GraphNetworkIndependentBlock, GraphNetworkBlock


def find_best_gpu():
    # this function finds the GPU with most free memory.
    if 'linux' in sys.platform and torch.cuda.device_count() > 1:
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
        memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
        gpu_id = np.argmax(memory_available).item()
        print("best gpu is %d with %0.1f Gb available space" % (gpu_id, memory_available[gpu_id] / 1000))
        return gpu_id


def compute_edge_features(x, edge_index, mode="full"):
    edge_attrs = []
    if mode == "full":
        for i, (e1, e2) in enumerate(edge_index.T):
            edge_attr = x[e1, :] - x[e2, :]
            edge_attr_sign = np.abs(edge_attr[:2])  # similar to MeshGraphNet paper.
            edge_attr = np.concatenate((edge_attr, edge_attr_sign))
            edge_attrs.append(edge_attr)
    else:
        raise (NotImplementedError("mode %s is not supported for edge features" % mode))
    edge_attrs = np.array(edge_attrs)
    return edge_attrs


def get_sdf_data_loader(n_objects, data_folder, batch_size, eval_frac=0.2,
                        reversed_edge_already_included=False, edge_weight=False,
                        global_features=True):
    print("preparing sdf data loader")
    random_idx = np.random.permutation(n_objects)
    train_idx = random_idx[:int((1 - eval_frac) * n_objects)]
    test_idx = random_idx[int((1 - eval_frac) * n_objects):]

    train_graph_data_list = []
    test_graph_data_list = []

    for idx, graph_data_list in zip([train_idx, test_idx], [train_graph_data_list, test_graph_data_list]):
        for i in idx:
            graph_nodes = np.load(data_folder + "graph_nodes%d.npy" % i).astype(float)
            x = graph_nodes.copy()
            if np.ndim(x) == 2:
                x[:, 2] = (x[:, 2] < 0).astype(float)
                y = graph_nodes.copy()[:, 2]
                y = y / np.sqrt(8)
                y = y.reshape(-1, 1)
            else:  # np.ndim(x) == 3
                x[:, :, 2] = (x[:, :, 2] < 0).astype(float)
                x = x.reshape(x.shape[0], -1)
                y = graph_nodes.copy()[:, :, 2]
                y = y / np.sqrt(8)
                y = np.mean(y, axis=-1, keepdims=True)

            graph_cells = np.load(data_folder + "graph_cells%d.npy" % i).astype(int)
            graph_cells = graph_cells.T
            graph_edges = np.load(data_folder + "graph_edges%d.npy" % i).astype(int)
            if not reversed_edge_already_included:
                graph_edges = add_reversed_edges(graph_edges)
            graph_edges = graph_edges.T
            n_edges = graph_edges.shape[1]
            if edge_weight:
                graph_edge_weights = compute_edge_features(x, graph_edges)
            else:
                graph_edge_weights = np.ones(n_edges)
                graph_edge_weights = graph_edge_weights.reshape(-1, 1)

            if global_features:
                cent = np.mean(x[x[:, 2] == 1, :2], axis=0, keepdims=True)
                area = np.mean(x[:, :2], keepdims=True)
                graph_global = np.concatenate((cent, area), axis=1)
            else:
                graph_global = np.array([[0.]])
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


def train_sdf(model, train_data, loss_func=None, use_cpu=False, save_name="",
              lr_0=0.001, n_epoch=101, print_every=10, step_size=50, gamma=0.5):
    assert loss_func is not None, "loss function should be specified"

    if use_cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
        gpu_id = find_best_gpu()
        if gpu_id:
            torch.cuda.set_device(gpu_id)

    model = model.to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    epoch_loss_list = []
    for epoch in range(n_epoch):
        epoch_loss = 0
        for data in train_data:
            data = data.to(device)
            model.train()
            optimizer.zero_grad()

            output = model(data)
            if not hasattr(model, 'full_output') or model.full_output is False:
                x_out, edge_attr_out, global_attr_out = output
                loss = loss_func(x_out[1], data.y)
            else:
                loss = [loss_func(out[1], data.y) for out in output]
                loss = sum(loss) / len(loss)

            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        epoch_loss = epoch_loss / len(train_data)
        epoch_loss_list.append(epoch_loss)
        scheduler.step()

        if epoch % print_every == 0:
            lr = optimizer.param_groups[0]['lr']
            print("epoch %d: learning rate=%0.3e, training loss:%0.4f" % (epoch, lr, epoch_loss))
            torch.save(model.state_dict(), "models/model" + save_name + ".pth")
            np.save("models/loss" + save_name + ".npy", epoch_loss_list)


def plot_sdf_results(model, data_loader, ndata=5, levels=None, border=None, save_name=""):
    try:
        loss_history = np.load("models/loss" + save_name + ".npy")
        plt.plot(loss_history)
        plt.yscale('log')
    except:
        pass

    device = 'cpu'
    model = model.to(device)
    model.load_state_dict(torch.load("models/model" + save_name + ".pth", map_location=device))
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            if i > ndata:
                break
            data = data.to(device=device)
            output = model(data)
            if hasattr(model, 'full_output') or model.full_output is True:
                output = output[-1]
            x_pred = output[1]  # node features

            cells = data.face.numpy()
            points = data.x.numpy()
            points[:, 2] = 0.
            mesh = meshio.Mesh(points=points, cells=[("triangle", cells.T)])

            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plot_mesh(mesh, vals=x_pred.numpy()[:, 0], with_colorbar=False, levels=levels, border=border)
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            plt.subplot(1, 2, 2)
            p = plot_mesh(mesh, vals=data.y.numpy()[:, 0], with_colorbar=False, levels=levels, border=border)
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            plt.gcf().subplots_adjust(right=0.8)
            cbar_ax = plt.gcf().add_axes([0.85, 0.15, 0.05, 0.7])
            plt.gcf().colorbar(p, cax=cbar_ax)
            plt.show()


def plot_mesh(mesh, dims=2, node_labels=False, vals=None, with_colorbar=False, levels=None, border=None):
    if not isinstance(mesh.points, np.ndarray):
        mesh.points = np.array(mesh.points)
    nodes_x = mesh.points[:, 0]
    nodes_y = mesh.points[:, 1]
    if dims == 2:
        elements_tris = [c for c in mesh.cells if c.type == "triangle"][0].data
        # plt.figure(figsize=(8, 8))
        if vals is None:
            plt.triplot(nodes_x, nodes_y, elements_tris, alpha=0.9, color='r')
        else:
            triangulation = tri.Triangulation(nodes_x, nodes_y, elements_tris)
            p = plt.tricontourf(triangulation, vals, 30)
            if with_colorbar: plt.colorbar()
            if levels:
                cn = plt.tricontour(triangulation, vals, levels, colors='w')
                plt.clabel(cn, fmt='%0.2f', colors='k', fontsize=10)
        if border:
            plt.hlines(1 - border, -1 + border, 1 - border, 'r')
            plt.hlines(-1 + border, -1 + border, 1 - border, 'r')
            plt.vlines(1 - border, -1 + border, 1 - border, 'r')
            plt.vlines(-1 + border, -1 + border, 1 - border, 'r')

    if node_labels:
        for i, (x, y) in enumerate(zip(nodes_x, nodes_y)):
            plt.text(x, y, i)

    if vals is not None:
        return p


if __name__ == "__main__":
    assert len(sys.argv) == 2
    data_folder = sys.argv[1]
    edge_weight = True

    n_objects, batch_size, n_epoch = 30, 12, 150
    lr_0, step_size, gamma, radius = 0.001, 200, 0.6, 0.1

    train_data, test_data = get_sdf_data_loader(n_objects, data_folder, batch_size, edge_weight=edge_weight)

    n_edge_feat_in, n_edge_feat_out = 5, 1
    n_node_feat_in, n_node_feat_out = 3, 1
    n_global_feat = 3

    model = EncodeProcessDecode(n_edge_feat_in=n_edge_feat_in, n_edge_feat_out=n_edge_feat_out,
                                n_node_feat_in=n_node_feat_in, n_node_feat_out=n_node_feat_out,
                                n_global_feat_in=n_global_feat, n_global_feat_out=n_global_feat,
                                mlp_latent_size=16, num_processing_steps=10, full_output=True,
                                encoder=GraphNetworkIndependentBlock, decoder=GraphNetworkIndependentBlock,
                                processor=GraphNetworkBlock, output_transformer=GraphNetworkIndependentBlock
                                )

    l1_loss = torch.nn.L1Loss()

    train_sdf(model, train_data, loss_func=l1_loss, use_cpu=False, n_epoch=n_epoch)

    plot_sdf_results(model, train_data)
    # plot_results(model, train_data, ndata=5, levels=[-0.2, 0, 0.2, 0.4], border=0.1, save_name="test")
    # plot_results_over_line(model, train_data, ndata=5, save_name="test")
