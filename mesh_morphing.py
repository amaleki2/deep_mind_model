import torch
import meshio
import numpy as np
import matplotlib.pyplot as plt
from sdf import find_best_gpu, plot_mesh
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import knn
from torch_scatter import scatter_sum
from graph_networks import EncodeProcessDecode, GraphNetworkBlock, GraphNetworkIndependentBlock, EncodeProcessDecodeNEW


def add_reversed_edges(edges):
    edges_reversed = np.flipud(edges)
    edges = np.concatenate([edges, edges_reversed], axis=1)
    return edges


def add_self_edges(edges):
    n_nodes = edges.max()
    self_edges = [list(range(n_nodes))] * 2
    self_edges = np.array(self_edges)
    edges = np.concatenate([edges, self_edges], axis=1)
    return edges


def cells_to_edges(cells, with_reversed_edges=True, with_self_edges=True):
    # edge_pairs = []
    # for cell in cells:
    #     for p1, p2 in [(0, 1), (1, 2), (0, 2)]:
    #         edge = sorted([cell[p1], cell[p2]])
    #         if edge not in edge_pairs:
    #             edge_pairs.append(edge)

    v0v1 = cells[:, :2]
    v1v2 = cells[:, 1:]
    v0v2 = cells[:, :3:2]
    edge_pairs = np.concatenate((v0v1, v1v2, v0v2), axis=0)
    edge_pairs = np.sort(edge_pairs, axis=1)
    edge_pairs = np.unique(edge_pairs, axis=0)

    edge_pairs = np.array(edge_pairs).T
    if with_reversed_edges:
        edge_pairs = add_reversed_edges(edge_pairs)
    if with_self_edges:
        edge_pairs = add_self_edges(edge_pairs)
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
                              cntr_pnt=torch.from_numpy(cnt_vertices).type(torch.float32),
                              x_p=torch.from_numpy(nodes_perturbed).type(torch.float32),
                              edge_index_p=torch.from_numpy(edges_perturbed).type(torch.long),
                              u_p=torch.from_numpy(global_attr_perturbed).type(torch.float32),
                              face_p=torch.from_numpy(cells_perturbed).type(torch.long),
                              cntr_pnt_p=torch.from_numpy(cnt_vertices_perturbed).type(torch.float32),
                              )

            graph_data_list.append(graph_data)
    train_data = DataLoader(train_graph_data_list, batch_size=batch_size)
    test_data = DataLoader(test_graph_data_list, batch_size=batch_size)
    return train_data, test_data


# def chamfer_loss(data, preds):
#     edge_attr, node_attr, global_attr = preds
#     s1, s2 = data.x_p.shape
#     s3, s4 = node_attr.shape
#     loss, _ = chamfer_distance(data.x_p.view(1, s1, s2), node_attr.view(1, s3, s4))
#     return loss

def compute_chamfer_region_weight(x, cntr_points, eps=1e-5):
    for i, pnt in enumerate(cntr_points):
        dist = torch.norm(x - pnt, dim=1)
        if i == 0:
            min_dist_from_cntr_points = dist
        else:
            # equivalent of torch.minimum which is not available in 1.6 version
            tmp = min_dist_from_cntr_points - dist > 0
            min_dist_from_cntr_points[tmp] = dist[tmp]
    weight = 1.0 / (min_dist_from_cntr_points + eps)
    weight = torch.clamp(weight, 1.0, 10.)
    return weight


def chamfer_distance(x, y, weight=None):
    idx = knn(y, x, 1)
    xs, ys = x[idx[0, :], :], y[idx[1, :], :]
    dist = torch.norm(xs - ys, dim=1)
    if weight is not None:
        dist = dist * weight
    dist_reduced = torch.mean(dist)
    return dist_reduced


def chamfer_loss(ground_mesh_nodes, predicted_mesh_nodes, ground_mesh_weight=None, predicted_mesh_weight=None):
    dist_ground = chamfer_distance(ground_mesh_nodes, predicted_mesh_nodes, weight=ground_mesh_weight)
    dist_predicted = chamfer_distance(predicted_mesh_nodes, ground_mesh_nodes, weight=predicted_mesh_weight)
    loss = 0.5 * (dist_ground + dist_predicted)
    return loss


def edge_loss(data, preds, delta=0.02):
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
    return loss

l2loss = torch.nn.MSELoss()
def laplace_loss(node_attr_in, node_attr_out, edge_index, enforced_region=None):
    n_nodes = len(node_attr_in)
    e1, e2 = edge_index[0, :], edge_index[1, :]
    laplace_in = scatter_sum(node_attr_in[e2, :], e1, dim=0, dim_size=n_nodes)
    laplace_in = 2 * node_attr_in - laplace_in
    laplace_out = scatter_sum(node_attr_out[e2, :], e1, dim=0, dim_size=n_nodes)
    laplace_out = 2 * node_attr_out - laplace_out
    if enforced_region is not None:
        loss = l2loss(laplace_in[enforced_region, :], laplace_out[enforced_region, :])
    else:
        loss = l2loss(laplace_in[:, :2], laplace_out)
    return loss


def get_angles(nodes, face):
    tri = nodes[face]
    v1 = tri[:, 0, :] - tri[:, 1, :]
    v2 = tri[:, 0, :] - tri[:, 2, :]
    return torch.sign(torch.sum(v1 * v2, dim=1))


def loss_angles(in_nodes, out_nodes, faces):
    angle_in_mesh = get_angles(in_nodes, faces)
    angle_out_mesh = get_angles(out_nodes, faces)
    loss = torch.sum((angle_in_mesh != angle_out_mesh).type(torch.float32))
    return loss


def train_mesh_morphing(model, train_data, use_cpu=False, save_name="",
                        chamfer_loss_lambda=1., edge_loss_lambda=0., laplace_loss_lambda=0., angle_loss_lambda=0.,
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
                w1 = compute_chamfer_region_weight(data.x_p, data.cntr_pnt_p)
                w2 = compute_chamfer_region_weight(output[1], data.cntr_pnt)
                loss1 = chamfer_loss(data.x_p, output[1], ground_mesh_weight=w1, predicted_mesh_weight=w2)
                loss1 *= chamfer_loss_lambda
                loss2 = edge_loss(data, output)
                loss2 *= edge_loss_lambda
                moved_node_idx = torch.any(abs(data.x[:, 2:]) > 0., dim=1)
                dist_from_moved_node = torch.norm(data.x[:, :2] - data.x[moved_node_idx, :2], dim=1)
                enforced_region = dist_from_moved_node > 0.2
                loss3 = laplace_loss(data.x, output[1], data.edge_index, enforced_region=enforced_region)
                loss3 *= laplace_loss_lambda
                loss4 = loss_angles(data.x, output[1], data.face)
            else:
                loss1 = 0.
                w1 = compute_chamfer_region_weight(data.x_p, data.cntr_pnt_p)
                for out in output:
                    w2 = None #compute_chamfer_region_weight(out[1], data.cntr_pnt)
                    loss1 += chamfer_loss(data.x_p, out[1], ground_mesh_weight=w1, predicted_mesh_weight=w2)
                loss1 /= len(output)
                loss1 *= chamfer_loss_lambda
                loss2 = [edge_loss(data, out) for out in output]
                loss2 = sum(loss2) / len(output)
                loss2 *= edge_loss_lambda
                loss3 = 0.
                moved_node_idx = torch.any(abs(data.x[:, 2:]) > 0., dim=1)
                dist_from_moved_node = torch.norm(data.x[:, :2] - data.x[moved_node_idx, :2], dim=1)
                enforced_region = dist_from_moved_node > 0.2
                # for i in range(len(output)):
                #     x_in = data.x[:, :2] if i == 0 else output[i-1][1]
                #     x_out = output[i][1]
                #     loss3 += laplace_loss(x_in, x_out, data.edge_index, enforced_region=enforced_region)
                # loss3 = loss3 / len(output)
                loss3 = [laplace_loss(data.x[:, :2], out[1], data.edge_index, enforced_region=enforced_region)
                         for out in output]
                loss3 = sum(loss3) / len(output)
                loss3 *= laplace_loss_lambda
                loss4 = [loss_angles(data.x, out[1], data.face) for out in output]
                loss4 = sum(loss4) / len(output)
                loss4 *= angle_loss_lambda

            loss = loss1 + loss2 + loss3 + loss4
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        epoch_loss = epoch_loss / len(train_data)
        epoch_loss_list.append(epoch_loss)
        scheduler.step()

        if epoch % print_every == 0:
            print("loss %0.5f %0.5f %0.5f %0.5f" %(loss1, loss2, loss3, loss4), end=" ")
            lr = optimizer.param_groups[0]['lr']
            print("epoch %d: learning rate=%0.3e, training loss:%0.5e" % (epoch, lr, epoch_loss))
            torch.save(model.state_dict(), "models/model" + save_name + ".pth")
            np.save("models/loss" + save_name + ".npy", epoch_loss_list)


def plot_mesh_morphing_results(model, data_loader, ndata=5, save_name="", output_idx=(-1,)):
    device = 'cpu'
    model = model.to(device)
    model.load_state_dict(torch.load("models/model" + save_name + ".pth", map_location=device))
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            if i > ndata:
                break
            data = data.to(device=device)
            vertices_input = data.x[:, :2]
            vertices_gt = data.x_p
            faces_input = data.face
            faces_pred = data.face
            faces_gt = data.face_p

            mesh_1 = meshio.Mesh(points=vertices_input, cells=[("triangle", faces_input)])
            mesh_2 = meshio.Mesh(points=vertices_gt, cells=[("triangle", faces_gt)])

            output = model(data)
            output_meshes = []
            for idx in output_idx:
                output_mesh = meshio.Mesh(points=output[idx][1], cells=[("triangle", faces_pred)])
                output_meshes.append(output_mesh)
            mx1 = min(vertices_input[:, 0].min(), vertices_gt[:, 0].min())
            mx2 = max(vertices_input[:, 0].max(), vertices_gt[:, 0].max())
            my1 = min(vertices_input[:, 1].min(), vertices_gt[:, 1].min())
            my2 = max(vertices_input[:, 1].max(), vertices_gt[:, 1].max())
            plt.figure(figsize=(15, 4))
            plt.subplot(1, len(output_idx) + 2, 1)

            plot_mesh(mesh_1)
            plt.xlim((mx1, mx2))
            plt.ylim((my1, my2))

            plt.subplot(1, len(output_idx) + 2, 2)
            plot_mesh(mesh_2)
            plt.xlim((mx1, mx2))
            plt.ylim((my1, my2))

            for i, mesh in enumerate(output_meshes):
                plt.subplot(1, len(output_idx) + 2, i + 3)
                plot_mesh(mesh)
                plt.xlim((mx1, mx2))
                plt.ylim((my1, my2))

            plt.show()


if __name__ == "__main__":
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
                                   mlp_latent_size=128, num_processing_steps=5, full_output=True,
                                   encoder=GraphNetworkIndependentBlock, decoder=GraphNetworkIndependentBlock,
                                   processor=GraphNetworkBlock, output_transformer=GraphNetworkIndependentBlock
                                   )

    # train_mesh_morphing(model, train_data, use_cpu=True,
    #                     laplace_loss_lambda=1., edge_loss_lambda=1., angle_loss_lambda=1.)
    plot_mesh_morphing_results(model, train_data, output_idx=(0, -1))
