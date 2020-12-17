import itertools
import collections
import numpy as np
import networkx as nx
from scipy import spatial
import pickle
import torch
import time
from torch.nn import CrossEntropyLoss
from torch_geometric.data import Data, DataLoader
from graph_networks import EncodeProcessDecode, GraphNetworkIndependentBlock, EncodeProcessDecodeNEW
from graph_base_models import get_edge_counts

DISTANCE_WEIGHT_NAME = "distance"  # The name for the distance edge attribute.


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def set_diff(seq0, seq1):
    """Return the set difference between 2 sequences as a list."""
    return list(set(seq0) - set(seq1))


def to_one_hot(indices, max_value, axis=-1):
    one_hot = np.eye(max_value)[indices]
    if axis not in (-1, one_hot.ndim):
        one_hot = np.moveaxis(one_hot, -1, axis)
    return one_hot


def get_node_dict(graph, attr):
    """Return a `dict` of node:attribute pairs from a graph."""
    return {k: v[attr] for k, v in graph.nodes.items()}


def generate_graph(rand, num_nodes_min_max, dimensions=2, theta=20.0, rate=1.0):
    """Creates a connected graph.

    The graphs are geographic threshold graphs, but with added edges via a
    minimum spanning tree algorithm, to ensure all nodes are connected.

    Args:
        rand: A random seed for the graph generator. Default= None.
        num_nodes_min_max: A sequence [lower, upper) number of nodes per graph.
        dimensions: (optional) An `int` number of dimensions for the positions.
            Default= 2.
        theta: (optional) A `float` threshold parameters for the geographic
            threshold graph's threshold. Large values (1000+) make mostly trees. Try
            20-60 for good non-trees. Default=1000.0.
        rate: (optional) A rate parameter for the node weight exponential sampling
          distribution. Default= 1.0.

    Returns:
        The graph.
    """
    # Sample num_nodes.
    num_nodes = rand.randint(*num_nodes_min_max)

    # Create geographic threshold graph.
    pos_array = rand.uniform(size=(num_nodes, dimensions))
    pos = dict(enumerate(pos_array))
    weight = dict(enumerate(rand.exponential(rate, size=num_nodes)))
    geo_graph = nx.geographical_threshold_graph(num_nodes, theta, pos=pos, weight=weight)

    # Create minimum spanning tree across geo_graph's nodes.
    distances = spatial.distance.squareform(spatial.distance.pdist(pos_array))
    i_, j_ = np.meshgrid(range(num_nodes), range(num_nodes), indexing="ij")
    weighted_edges = list(zip(i_.ravel(), j_.ravel(), distances.ravel()))
    mst_graph = nx.Graph()
    mst_graph.add_weighted_edges_from(weighted_edges, weight=DISTANCE_WEIGHT_NAME)
    mst_graph = nx.minimum_spanning_tree(mst_graph, weight=DISTANCE_WEIGHT_NAME)
    # Put geo_graph's node attributes into the mst_graph.
    for i in mst_graph.nodes():
        mst_graph.nodes[i].update(geo_graph.nodes[i])

    # Compose the graphs.
    combined_graph = nx.compose_all([mst_graph, geo_graph.copy()])
    # Put all distance weights into edge attributes.
    for i, j in combined_graph.edges():
        combined_graph.get_edge_data(i, j).setdefault(DISTANCE_WEIGHT_NAME, distances[i, j])
    return combined_graph, mst_graph, geo_graph


def add_shortest_path(rand, graph, min_length=1):
    """Samples a shortest path from A to B and adds attributes to indicate it.

    Args:
        rand: A random seed for the graph generator. Default= None.
        graph: A `nx.Graph`.
        min_length: (optional) An `int` minimum number of edges in the shortest
          path. Default= 1.

    Returns:
        The `nx.DiGraph` with the shortest path added.

    Raises:
        ValueError: All shortest paths are below the minimum length
    """
    # Map from node pairs to the length of their shortest path.
    pair_to_length_dict = {}
    try:
        # This is for compatibility with older networkx.
        lengths = nx.all_pairs_shortest_path_length(graph).items()
    except AttributeError:
        # This is for compatibility with newer networkx.
        lengths = list(nx.all_pairs_shortest_path_length(graph))
    for x, yy in lengths:
        for y, l in yy.items():
            if l >= min_length:
                pair_to_length_dict[x, y] = l
    if max(pair_to_length_dict.values()) < min_length:
        raise ValueError("All shortest paths are below the minimum length")
    # The node pairs which exceed the minimum length.
    node_pairs = list(pair_to_length_dict)

    # Computes probabilities per pair, to enforce uniform sampling of each
    # shortest path lengths.
    # The counts of pairs per length.
    counts = collections.Counter(pair_to_length_dict.values())
    prob_per_length = 1.0 / len(counts)
    probabilities = [prob_per_length / counts[pair_to_length_dict[x]] for x in node_pairs]

    # Choose the start and end points.
    i = rand.choice(len(node_pairs), p=probabilities)
    start, end = node_pairs[i]
    path = nx.shortest_path(graph, source=start, target=end, weight=DISTANCE_WEIGHT_NAME)

    # Creates a directed graph, to store the directed path from start to end.
    digraph = graph.to_directed()

    # Add the "start", "end", and "solution" attributes to the nodes and edges.
    digraph.add_node(start, start=True)
    digraph.add_node(end, end=True)
    digraph.add_nodes_from(set_diff(digraph.nodes(), [start]), start=False)
    digraph.add_nodes_from(set_diff(digraph.nodes(), [end]), end=False)
    digraph.add_nodes_from(set_diff(digraph.nodes(), path), solution=False)
    digraph.add_nodes_from(path, solution=True)
    path_edges = list(pairwise(path))
    digraph.add_edges_from(set_diff(digraph.edges(), path_edges), solution=False)
    digraph.add_edges_from(path_edges, solution=True)

    return digraph

def argmax_keepdims(x, axis=None):
    """
    Returns the indices of the maximum values along an axis.

    The axis which is reduced is left in the result as dimension with size one.
    The result will broadcast correctly against the input array.

    Original numpy.argmax() implementation does not currently support the keepdims parameter.
    See https://github.com/numpy/numpy/issues/8710 for further information.
    """
    if axis is None:
        return np.argmax(x)

    output_shape = list(x.shape)
    output_shape[axis] = 1
    return np.argmax(x, axis=axis).reshape(output_shape)


def graph_to_input_target(graph):
    """Returns 2 graphs with input and target feature vectors for training.

    Args:
        graph: An `nx.DiGraph` instance.

    Returns:
        The input `nx.DiGraph` instance.
        The target `nx.DiGraph` instance.

    Raises:
        ValueError: unknown node type
    """

    def create_feature(attr, fields):
        return np.hstack([np.array(attr[field], dtype=float) for field in fields])

    input_node_fields = ("pos", "weight", "start", "end")
    input_edge_fields = ("distance",)
    target_node_fields = ("solution",)
    target_edge_fields = ("solution",)

    input_graph = graph.copy()
    target_graph = graph.copy()

    solution_length = 0
    for node_index, node_feature in graph.nodes(data=True):
        input_graph.add_node(node_index, features=create_feature(node_feature, input_node_fields))
        target_node = to_one_hot(create_feature(node_feature, target_node_fields).astype(int), 2)[0]
        target_graph.add_node(node_index, features=target_node)
        solution_length += int(node_feature["solution"])
    solution_length /= graph.number_of_nodes()

    for receiver, sender, features in graph.edges(data=True):
        input_graph.add_edge(sender, receiver, features=create_feature(features, input_edge_fields))
        target_edge = to_one_hot(create_feature(features, target_edge_fields).astype(int), 2)[0]
        target_graph.add_edge(sender, receiver, features=target_edge)

    input_graph.graph["features"] = np.array([0.0])
    target_graph.graph["features"] = np.array([solution_length], dtype=float)

    return input_graph, target_graph


def generate_networkx_graphs(rand, num_examples, num_nodes_min_max, theta):
    """Generate graphs for training.

    Args:
        rand: A random seed (np.RandomState instance).
        num_examples: Total number of graphs to generate.
        num_nodes_min_max: A 2-tuple with the [lower, upper) number of nodes per
            graph. The number of nodes for a graph is uniformly sampled within this
            range.
        theta: (optional) A `float` threshold parameters for the geographic
            threshold graph's threshold. Default= the number of nodes.

    Returns:
        input_graphs: The list of input graphs.
        target_graphs: The list of output graphs.
        graphs: The list of generated graphs.
    """
    input_graphs = []
    target_graphs = []
    graphs = []
    for _ in range(num_examples):
        graph = generate_graph(rand, num_nodes_min_max, theta=theta)[0]
        graph = add_shortest_path(rand, graph)
        input_graph, target_graph = graph_to_input_target(graph)
        input_graphs.append(input_graph)
        target_graphs.append(target_graph)
        graphs.append(graph)
    return input_graphs, target_graphs, graphs


def get_data_from_networkx(graph_nx):
    """Generate torch_geometric data from graphs for training and testing.

    Args:
        graph_nx: an input graph.

    Returns:
        data: an object of type torch_geometric.data.Data with following keys populated
            x: node data
            edge_index: edge data
            edge_attr: edge features
            u: global features
    """
    nodes = graph_nx.nodes(data=True)
    node_data = []
    for node_i, (key, data) in enumerate(nodes):
        assert node_i == key, "Nodes of the networkx must have sequential"
        node_data.append(data['features'])
    node_data = np.array(node_data)
    senders, receivers, edge_attr_dicts = zip(*graph_nx.edges(data=True))
    edge_index = np.array([senders, receivers])
    edge_data = [x["features"] for x in edge_attr_dicts if x["features"] is not None]
    edge_data = np.array(edge_data)
    global_data = graph_nx.graph["features"]
    global_data = np.atleast_2d(global_data)
    # if is_target:
    #     node_data = argmax_keepdims(node_data, axis=1)
    #     node_data = np.concatenate((node_data, 1-node_data), axis=1)
    #     edge_data = argmax_keepdims(edge_data, axis=1)
    #     edge_data = np.concatenate((edge_data, 1-edge_data), axis=1)
    #     data = Data(x=torch.from_numpy(node_data).type(torch.long),
    #                 edge_index=torch.from_numpy(edge_index).type(torch.long),
    #                 edge_attr=torch.from_numpy(edge_data).type(torch.long),
    #                 u=torch.from_numpy(global_data).type(torch.float32))
    # else:
    data = Data(x=torch.from_numpy(node_data).type(torch.float32),
                edge_index=torch.from_numpy(edge_index).type(torch.long),
                edge_attr=torch.from_numpy(edge_data).type(torch.float32),
                u=torch.from_numpy(global_data).type(torch.float32))
    return data


def setup_data_loader(rand, num_examples, num_nodes_min_max, theta, batch_size=None):
    """Generate torch_geometric data from graphs for training and testing.

        Args:
            rand: random seed to generate random graphs
            num_examples: number of examples for train and test
            num_nodes_min_max: a tuple of min and max number of nodes
            theta: parameter to control graph generation.
            batch_size: batch size
        Returns:
            x_data_loader, y_data_loader: objects of type torch_geometric.data.DataLoader
        """
    # batch_size = num_examples if batch_size is None else batch_size
    batch_size = num_examples if batch_size is None else batch_size
    input_graphs, target_graphs, _ = generate_networkx_graphs(rand, num_examples, num_nodes_min_max, theta)
    X = [get_data_from_networkx(input_graph) for input_graph in input_graphs]
    Y = [get_data_from_networkx(target_graph) for target_graph in target_graphs]
    x_data_loader = DataLoader(X, batch_size=batch_size)
    y_data_loader = DataLoader(Y, batch_size=batch_size)
    return x_data_loader, y_data_loader


def setup_data_loader_from_saved_networks(num_graphs, batch_size=None):
    """Generate torch_geometric data from saved pickle files of network.
       implemented for comparison between tensorflow and pytorch.

        Args:
            rand: random seed to generate random graphs
            num_examples: number of examples for train and test
            num_nodes_min_max: a tuple of min and max number of nodes
            theta: parameter to control graph generation.
            batch_size: batch size
        Returns:
            x_data_loader, y_data_loader: objects of type torch_geometric.data.DataLoader
        """
    batch_size = num_graphs if batch_size is None else batch_size

    input_graphs = []
    target_graphs = []
    for i in range(num_graphs):
        input_graphs.append(nx.read_gpickle("test_graphs/input_graph_%d" % i))
        target_graphs.append(nx.read_gpickle("test_graphs/target_graph_%d" % i))

    X = [get_data_from_networkx(input_graph) for input_graph in input_graphs]
    Y = [get_data_from_networkx(target_graph) for target_graph in target_graphs]
    x_data_loader = DataLoader(X, batch_size=batch_size)
    y_data_loader = DataLoader(Y, batch_size=batch_size)
    return x_data_loader, y_data_loader


ce_loss = CrossEntropyLoss()
def create_loss_batched(x_edge_attr, x_node_attr, y_edge_attr, y_node_attr, edge_index, batch, use_edges=True):
    """
    loss function for shortest path example
    :param y: target graph
    :param x: predicted graph
    :return: a list of loss values
    """
    y_node_attr_reduce_1d = torch.argmax(y_node_attr, dim=1)
    loss = ce_loss(x_node_attr, y_node_attr_reduce_1d)
    if use_edges:
        y_edge_attr_reduce_1d = torch.argmax(y_edge_attr, dim=1)
        loss += ce_loss(x_edge_attr, y_edge_attr_reduce_1d)
    return loss


def compute_accuracy_batched(x_edge_attr, x_node_attr, y_edge_attr, y_node_attr, edge_index, batch, use_edges=False):
    """
    compute accuracy for the shortest path problem when graphs are batched.
    :y: target graph
    :x: predicted graph
    :return: batch accur
    """
    node_counts = torch.unique(batch, return_counts=True)[1]
    edge_counts = get_edge_counts(edge_index, batch)
    node_indices = torch.cumsum(node_counts, dim=0)
    edge_indices = torch.cumsum(edge_counts, dim=0)
    node_istart = 0
    edge_istart = 0
    acc_batch = 0
    for node_iend, edge_iend in zip(node_indices, edge_indices):
        x_edge_attr_b = x_edge_attr[edge_istart:edge_iend]
        x_node_attr_b = x_node_attr[node_istart:node_iend]
        y_edge_attr_b = y_edge_attr[edge_istart:edge_iend]
        y_node_attr_b = y_node_attr[node_istart:node_iend]
        acc = compute_accuracy(x_edge_attr_b, x_node_attr_b, y_edge_attr_b, y_node_attr_b, use_edges=use_edges)
        acc_batch += acc
        node_istart = node_iend
        edge_istart = edge_iend
    acc_batch /= len(node_indices)
    return acc_batch


def compute_accuracy(x_edge_attr, x_node_attr, y_edge_attr, y_node_attr, use_edges=False):
    c1 = torch.argmax(x_node_attr, dim=1) == torch.argmax(y_node_attr, dim=1)
    if use_edges:
        c2 = torch.argmax(x_edge_attr, dim=1) == torch.argmax(y_edge_attr, dim=1)
        c = torch.cat([c1, c2])
    else:
        c = c1
    c_mean = torch.mean(c.float())
    c_all = torch.all(c)
    return torch.cat([c_mean.view(1), c_all.float().view(1)])


def train_batched(model, data_generator, train_data_params, test_data_params, loss_func,
                  accuracy_func=None, use_cpu=False, lr_0=0.001, n_epoch=101,
                  print_every=10, step_size=50, gamma=0.5):
    if use_cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
        # gpu_id = find_best_gpu()
        # if gpu_id:
        #     torch.cuda.set_device(gpu_id)

    model = model.to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    data_saver_dict = {"train": {"loss": [], "acc1": [], "acc2": []},
                       "test":  {"loss": [], "acc1": [], "acc2": []}}

    for epoch in range(n_epoch):
        epoch_metrics = None
        x_data_loader, y_data_loader = data_generator(*train_data_params)
        for x_in, y in zip(x_data_loader, y_data_loader):
            x_in.to(device=device)
            y.to(device=device)
            model.train()
            optimizer.zero_grad()
            output = model(x_in)
            if not hasattr(model, 'full_output') or model.full_output is False:
                x_out = output
                loss = loss_func(x_out[0], x_out[1], y.edge_attr, y.x, x_in.edge_index, x_in.batch)
            else:
                loss = [loss_func(x_out[0], x_out[1], y.edge_attr, y.x, x_in.edge_index, x_in.batch) for x_out in output]
                loss = sum(loss) / len(loss)
                x_out = output[-1]

            if accuracy_func is not None:
                acc = accuracy_func(x_out[0], x_out[1], y.edge_attr, y.x, x_in.edge_index, x_in.batch)
                epoch_metric = torch.cat((loss.view(1), acc))
            else:
                epoch_metric = loss
            if epoch_metrics is None:
                epoch_metrics = epoch_metric
            else:
                epoch_metrics += epoch_metric

            loss.backward()
            optimizer.step()
        epoch_metrics = epoch_metrics / len(x_data_loader)
        scheduler.step()
        if epoch % print_every == 0:
            epoch_metrics_test = torch.zeros_like(epoch_metrics)
            x_test_data_loader, y_test_data_loader = data_generator(*test_data_params)
            for x_in_test, y_test in zip(x_test_data_loader, y_test_data_loader):
                x_in_test.to(device=device)
                y_test.to(device=device)
                model.eval()

                output_test = model(x_in_test)
                if hasattr(model, 'full_output') and model.full_output:
                    x_out_test = output_test[-1]
                else:
                    x_out_test = output_test
                loss_test = loss_func(x_out_test[0], x_out_test[1], y_test.edge_attr, y_test.x, x_in_test.edge_index, x_in_test.batch)
                if accuracy_func is not None:
                    acc_test = accuracy_func(x_out_test[0], x_out_test[1], y_test.edge_attr, y_test.x, x_in_test.edge_index, x_in_test.batch)
                    epoch_metric_test = torch.cat((loss_test.view(1), acc_test))
                else:
                    epoch_metric_test = loss_test

                epoch_metrics_test += epoch_metric_test
            epoch_metrics_test = epoch_metrics_test / len(x_test_data_loader)
            if accuracy_func is None:
                epoch_metric_print_ready = round(epoch_metrics.tolist(), 3)
                epoch_metrics_test_print_ready = round(epoch_metrics_test.tolist(), 3)
            else:
                epoch_metric_print_ready = [round(x, 3) for x in epoch_metrics.tolist()]
                epoch_metrics_test_print_ready = [round(x, 3) for x in epoch_metrics_test.tolist()]
            print("epoch %d: lr: %0.5e, training metrics:" % (epoch, optimizer.param_groups[0]['lr']),
                  epoch_metric_print_ready, ", test metrics:", epoch_metrics_test_print_ready)
            data_saver_dict["train"]["loss"].append(epoch_metric_print_ready[0])
            data_saver_dict["train"]["acc1"].append(epoch_metric_print_ready[1])
            data_saver_dict["train"]["acc2"].append(epoch_metric_print_ready[2])
            data_saver_dict["test"]["loss"].append(epoch_metrics_test_print_ready[0])
            data_saver_dict["test"]["acc1"].append(epoch_metrics_test_print_ready[1])
            data_saver_dict["test"]["acc2"].append(epoch_metrics_test_print_ready[2])

            with open("data_saver.pkl", "wb") as fid:
                pickle.dump(data_saver_dict, fid, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    seed = 1
    rand = np.random.RandomState(seed=seed)
    theta = 20   # Large values (1000+) make trees. Try 20-60 for good non-trees.

    num_training_examples = 32
    num_training_nodes_min_max = (8, 17)

    num_test_examples = 100
    num_test_nodes_min_max = (16, 33)

    n_edge_feat_in, n_edge_feat_out = 1, 2
    n_node_feat_in, n_node_feat_out = 5, 2
    n_global_feat = 1

    model = EncodeProcessDecodeNEW(n_edge_feat_in=n_edge_feat_in, n_edge_feat_out=n_edge_feat_out,
                                n_node_feat_in=n_node_feat_in, n_node_feat_out=n_node_feat_out,
                                n_global_feat_in=n_global_feat, n_global_feat_out=n_global_feat,
                                mlp_latent_size=64, num_processing_steps=10, full_output=True,
                                encoder=GraphNetworkIndependentBlock, decoder=GraphNetworkIndependentBlock,
                                output_transformer=GraphNetworkIndependentBlock
                                )

    train_data_generator_params = (rand, num_training_examples, num_training_nodes_min_max, theta)
    test_data_generator_params = (rand, num_test_examples, num_test_nodes_min_max, theta)
    train_batched(model, setup_data_loader, train_data_generator_params, test_data_generator_params,
                  loss_func=create_loss_batched,
                  accuracy_func=compute_accuracy_batched,
                  lr_0=1e-3, n_epoch=5000, print_every=25, step_size=5000, gamma=0.25)

import numpy as np
import matplotlib.pyplot as plt
tf_file_name = "tf_data_saver_1.pkl"
torch_file_name = "data_saver_1.pkl"
with open(tf_file_name, "rb") as fid1:
    tf_results = pickle.load(fid1)

with open(torch_file_name, "rb") as fid2:
    torch_results = pickle.load(fid2)

tf_train_results = tf_results["train"]
tf_test_results = tf_results["test"]
torch_train_results = torch_results["train"]
torch_test_results = torch_results["test"]

num_epoch = np.linspace(0, 4225, 170)
plt.figure(figsize=(15, 10))
plt.subplot(2, 3, 1)
plt.plot(num_epoch, tf_train_results['loss'][:170], linewidth=1, color='k')
plt.plot(num_epoch, torch_train_results['loss'], linewidth=1, color='r')
plt.yscale('log')
plt.ylabel('train loss')
plt.xlabel('epochs')

plt.subplot(2, 3, 4)
plt.plot(num_epoch, tf_test_results['loss'][:170], linewidth=1, color='k', linestyle='--')
plt.plot(num_epoch, torch_test_results['loss'], linewidth=1, color='r', linestyle='--')
plt.yscale('log')
plt.ylabel('test loss')
plt.xlabel('epochs')

plt.subplot(2, 3, 2)
plt.plot(num_epoch, tf_train_results['acc1'][:170], linewidth=1, color='k')
plt.plot(num_epoch, torch_train_results['acc1'], linewidth=1, color='r')
plt.ylabel('train acc1')
plt.xlabel('epochs')

plt.subplot(2, 3, 5)
plt.plot(num_epoch, tf_test_results['acc1'][:170], linewidth=1, color='k', linestyle='--')
plt.plot(num_epoch, torch_test_results['acc1'], linewidth=1, color='r', linestyle='--')
plt.ylabel('test acc1')
plt.xlabel('epochs')

plt.subplot(2, 3, 3)
plt.plot(num_epoch, tf_train_results['acc2'][:170], linewidth=1, color='k')
plt.plot(num_epoch, torch_train_results['acc2'], linewidth=1, color='r')
plt.ylabel('train acc2')
plt.xlabel('epochs')

plt.subplot(2, 3, 6)
plt.plot(num_epoch, torch_test_results['acc2'], linewidth=1, color='r', linestyle='--')
plt.plot(num_epoch, tf_test_results['acc2'][:170], linewidth=1, color='k', linestyle='--')
plt.ylabel('test acc2')
plt.xlabel('epochs')
