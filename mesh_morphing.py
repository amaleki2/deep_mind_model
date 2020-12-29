import torch
import meshio
import numpy as np
import matplotlib.pyplot as plt
from sdf import find_best_gpu, plot_mesh
from pytorch3d.loss import chamfer_distance
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import knn
from graph_networks import EncodeProcessDecode, GraphNetworkBlock, GraphNetworkIndependentBlock, EncodeProcessDecodeNEW


def add_reversed_edges(edges):
    edges_reversed = np.flipud(edges)
    edges = np.concatenate([edges, edges_reversed], axis=1)
    return edges


def cells_to_edges(cells):
    edge_pairs = []
    for cell in cells:
        for p1, p2 in [(0, 1), (1, 2), (0, 2)]:
            edge = sorted([cell[p1], cell[p2]])
            if edge not in edge_pairs:
                edge_pairs.append(edge)
    edge_pairs = np.array(edge_pairs).T
    edge_pairs = add_reversed_edges(edge_pairs)
    return edge_pairs


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


def compute_area_triangle(vertices):
    v1x, v1y = vertices[0]
    v2x, v2y = vertices[1]
    v3x, v3y = vertices[2]
    area = 0.5 * abs(v1x * (v2y - v3y) + v2x * (v3y - v1y) + v3x * (v1y - v2y))
    return area


def compute_global_features(nodes, cells):
    area = 0
    for cell in cells:
        vertices = nodes[cell, :]
        area += compute_area_triangle(vertices)
    return np.array([[area]])


def add_control_pnts(nodes, vertices, vertices_perturbed):
    new_feat = np.zeros((len(nodes), 2))
    vertices[:, 1] = -vertices[:, 1]
    vertices_perturbed[:, 1] = -vertices_perturbed[:, 1]
    for v, vp in zip(vertices, vertices_perturbed):
        dist = np.linalg.norm(nodes - v, axis=1)
        idx = np.argmin(dist)
        new_feat[idx] = vp - v
    new_nodes = np.concatenate((nodes, new_feat), axis=1)
    return new_nodes


def get_meshmorph_data_loader(n_objects, data_folder, batch_size, eval_frac=0.2):
    print("preparing mesh morphing data loader")
    random_idx = np.random.permutation(n_objects)
    train_idx = random_idx[:int((1 - eval_frac) * n_objects)]
    test_idx = random_idx[int((1 - eval_frac) * n_objects):]

    train_graph_data_list = []
    test_graph_data_list = []

    for idx, graph_data_list in zip([train_idx, test_idx], [train_graph_data_list, test_graph_data_list]):
        for i in idx:
            mesh_file = data_folder + "mesh_%d.vtk" % i
            mesh_data = meshio.read(mesh_file)
            nodes = mesh_data.points[:, :2]
            cells = np.array([c.data for c in mesh_data.cells if c.type == 'triangle'][0])
            edges = cells_to_edges(cells)
            edges_attr = compute_edge_features(nodes, edges)
            global_attr = compute_global_features(nodes, cells)
            control_vertices_file = data_folder + "vertices_%d.npy" % i
            control_vertices_file_perturbed = data_folder + "vertices_perturbed_%d.npy" % i
            cnt_vertices = np.load(control_vertices_file)
            cnt_vertices_perturbed = np.load(control_vertices_file_perturbed)
            nodes = add_control_pnts(nodes, cnt_vertices, cnt_vertices_perturbed)

            mesh_file_perturbed = data_folder + "mesh_perturbed_%d.vtk" % i
            mesh_data_perturbed = meshio.read(mesh_file_perturbed)
            nodes_perturbed = mesh_data_perturbed.points[:, :2]
            cells_perturbed = np.array([c.data for c in mesh_data_perturbed.cells if c.type == 'triangle'][0])
            edges_perturbed = cells_to_edges(cells_perturbed)
            global_attr_perturbed = compute_global_features(nodes_perturbed, cells_perturbed)

            graph_data = Data(x=torch.from_numpy(nodes).type(torch.float32),
                              edge_index=torch.from_numpy(edges).type(torch.long),
                              edge_attr=torch.from_numpy(edges_attr).type(torch.float32),
                              u=torch.from_numpy(global_attr).type(torch.float32),
                              face=torch.from_numpy(cells).type(torch.long),
                              x_p=torch.from_numpy(nodes_perturbed).type(torch.float32),
                              edge_index_p=torch.from_numpy(edges_perturbed).type(torch.long),
                              u_p=torch.from_numpy(global_attr_perturbed).type(torch.float32),
                              face_p=torch.from_numpy(cells_perturbed).type(torch.long))

            graph_data_list.append(graph_data)
    train_data = DataLoader(train_graph_data_list, batch_size=batch_size)
    test_data = DataLoader(test_graph_data_list, batch_size=batch_size)
    return train_data, test_data


def train_mesh_morphing(model, train_data, use_cpu=False, save_name="",
                        lr_0=0.001, n_epoch=101, print_every=10, step_size=250, gamma=0.5):
    print("training begins")

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
                loss1 = chamfer_loss(data, output)
                loss2 = edge_loss(data, output)
            else:
                loss1 = [chamfer_loss(data, out) for out in output]
                loss1 = sum(loss1) / len(loss1)
                loss2 = [edge_loss(data, out) for out in output]
                loss2 = sum(loss2) / len(loss2)
            loss = loss1 #+ loss2
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        epoch_loss = epoch_loss / len(train_data)
        epoch_loss_list.append(epoch_loss)
        scheduler.step()

        if epoch % print_every == 0:
            print("loss %0.4f %0.4f" %(loss1, loss2), end=" ")
            lr = optimizer.param_groups[0]['lr']
            print("epoch %d: learning rate=%0.3e, training loss:%0.4f" % (epoch, lr, epoch_loss))
            torch.save(model.state_dict(), "models/model" + save_name + ".pth")
            np.save("models/loss" + save_name + ".npy", epoch_loss_list)


def chamfer_loss(data, preds):
    edge_attr, node_attr, global_attr = preds
    s1, s2 = data.x_p.shape
    s3, s4 = node_attr.shape
    loss, _ = chamfer_distance(data.x_p.view(1, s1, s2), node_attr.view(1, s3, s4))
    if loss is None:
        print("Nan loss")
    return loss


def edge_loss(data, preds, delta=0.1):
    edge_attr, node_attr, global_attr = preds
    edge_index = data.edge_index
    senders = node_attr[edge_index[0, :], :]
    receivers = node_attr[edge_index[1, :], :]
    edge_lengths = torch.sum(abs(senders - receivers), dim=1)
    large_lengths = abs(edge_lengths) > delta
    if large_lengths.sum() == 0:
        loss = 0.
    else:
        loss = torch.mean(edge_lengths[large_lengths] ** 2)
    if loss is None:
        print("Nan loss")
    return loss


def plot_mesh_morphing_results(model, data_loader, ndata=5, save_name=""):
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
            vertices_input = data.x[:, :2]
            vertices_pred = output[1]  # node features
            vertices_gt = data.x_p
            faces_input = data.face
            faces_pred = data.face
            faces_gt = data.face_p

            mesh_1 = meshio.Mesh(points=vertices_input, cells=[("triangle", faces_input)])
            mesh_2 = meshio.Mesh(points=vertices_gt, cells=[("triangle", faces_gt)])
            mesh_3 = meshio.Mesh(points=vertices_pred, cells=[("triangle", faces_pred)])

            plt.figure(figsize=(10, 4))
            plt.subplot(1, 3, 1)
            plot_mesh(mesh_1)
            plt.xlim((vertices_input[:, 0].min() - 0.1, vertices_gt[:, 0].max() + 0.1))
            plt.ylim((vertices_input[:, 1].min() - 0.1, vertices_gt[:, 1].max() + 0.1))
            # plt.gca().set_xticks([])
            # plt.gca().set_yticks([])
            plt.subplot(1, 3, 2)
            plot_mesh(mesh_2)
            plt.xlim((vertices_input[:, 0].min() - 0.1, vertices_gt[:, 0].max() + 0.1))
            plt.ylim((vertices_input[:, 1].min() - 0.1, vertices_gt[:, 1].max() + 0.1))
            # plt.gca().set_xticks([])
            # plt.gca().set_yticks([])
            plt.subplot(1, 3, 3)
            plot_mesh(mesh_3)
            # plt.scatter(vertices_pred[:, 0], vertices_pred[:, 1], c='k', marker='.')
            plt.xlim((vertices_input[:, 0].min() - 0.1, vertices_gt[:, 0].max() + 0.1))
            plt.ylim((vertices_input[:, 1].min() - 0.1, vertices_gt[:, 1].max() + 0.1))
            # plt.gca().set_xticks([])
            # plt.gca().set_yticks([])
            plt.show()


n_objects = 10
data_folder = "../mesh_gen/mesh_morphing/"
batch_size = 1
train_data, test_data = get_meshmorph_data_loader(n_objects, data_folder, batch_size, eval_frac=0.2)

n_edge_feat_in, n_edge_feat_out = 4, 1
n_node_feat_in, n_node_feat_out = 4, 2
n_global_feat_in, n_global_feat_out = 1, 1

model = EncodeProcessDecodeNEW(n_edge_feat_in=n_edge_feat_in, n_edge_feat_out=n_edge_feat_out,
                               n_node_feat_in=n_node_feat_in, n_node_feat_out=n_node_feat_out,
                               n_global_feat_in=n_global_feat_in, n_global_feat_out=n_global_feat_out,
                               mlp_latent_size=16, num_processing_steps=5, full_output=True,
                               encoder=GraphNetworkIndependentBlock, decoder=GraphNetworkIndependentBlock,
                               processor=GraphNetworkBlock, output_transformer=GraphNetworkIndependentBlock
                               )


# train_mesh_morphing(model, train_data, use_cpu=True)
plot_mesh_morphing_results(model, train_data)
