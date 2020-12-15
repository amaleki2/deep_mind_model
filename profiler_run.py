import torch
import numpy as np
import torch.autograd.profiler as profiler
from shortest_path import setup_data_loader, create_loss_ops_GN, compute_accuracy_batched
from graph_networks import EncodeProcessDecode, GraphNetworkIndependentBlock


def one_train_step(device, optimizer, x_data, y_data, x_test_data, y_test_data, loss_func, accuracy_func):
    for x_in, y in zip(x_data, y_data):
        x_in.to(device=device)
        y.to(device=device)
        model.train()
        optimizer.zero_grad()
        output = model(x_in)

        if not hasattr(model, 'full_output') or model.full_output is False:
            x_out = output
            loss = loss_func(y, x_out)
        else:
            loss = [loss_func(y, x_out) for x_out in output]
            loss = sum(loss) / len(loss)
            x_out = output[-1]

        if accuracy_func is not None:
            acc = accuracy_func(y, x_out)
            epoch_metric = torch.cat((loss.view(1), acc))
        else:
            epoch_metric = loss

        loss.backward()
        # optimizer.step()

    # for x_in_test, y_test in zip(x_test_data, y_test_data):
    #     x_in_test.to(device=device)
    #     y_test.to(device=device)
    #     model.eval()
    #
    #     output_test = model(x_in_test)
    #     if hasattr(model, 'full_output') and model.full_output:
    #         x_out_test = output_test[-1]
    #     else:
    #         x_out_test = output_test
    #     loss_test = loss_func(y_test, x_out_test)
    #     if accuracy_func is not None:
    #         acc_test = accuracy_func(y_test, x_out_test)
    #         epoch_metric_test = torch.cat((loss_test.view(1), acc_test))
    #     else:
    #         epoch_metric_test = loss_test


n_edge_feat_in, n_edge_feat_out = 1, 2
n_node_feat_in, n_node_feat_out = 5, 2
n_global_feat = 1
rand = np.random.RandomState(seed=1)
x_data_loader, y_data_loader = setup_data_loader(rand, 32, (8, 17), 20)
x_test_data_loader, y_test_data_loader = setup_data_loader(rand, 100, (16, 25), 20)
x_in = next(iter(x_data_loader))
model = EncodeProcessDecode(n_edge_feat_in=n_edge_feat_in, n_edge_feat_out=n_edge_feat_out,
                            n_node_feat_in=n_node_feat_in, n_node_feat_out=n_node_feat_out,
                            n_global_feat_in=n_global_feat, n_global_feat_out=n_global_feat,
                            mlp_latent_size=16, num_processing_steps=10, full_output=True,
                            encoder=GraphNetworkIndependentBlock, decoder=GraphNetworkIndependentBlock,
                            output_transformer=GraphNetworkIndependentBlock
                            )

with profiler.profile(record_shapes=True, use_cuda=False) as prof:
    with profiler.record_function("model_inference"):
        one_train_step(torch.device('cpu'),
                       torch.optim.Adam(model.parameters(), lr=0.001),
                       x_data_loader, y_data_loader,
                       x_test_data_loader, y_test_data_loader,
                       create_loss_ops_GN, compute_accuracy_batched)

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))