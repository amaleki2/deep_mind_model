import os
import sys
import torch
import numpy as np


def find_best_gpu():
    # this function finds the GPU with most free memory.
    if 'linux' in sys.platform and torch.cuda.device_count() > 1:
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
        memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
        gpu_id = np.argmax(memory_available).item()
        print("best gpu is %d with %0.1f Gb available space" %(gpu_id, memory_available[gpu_id]/1000))
        return gpu_id


def train_shortest_path(model, x_data_loader, y_data_loader, loss_func,
                        test_data=None, accuracy_func=None,
                        use_cpu=False, lr_0=0.001, n_epoch=101,
                        print_every=10, step_size=50, gamma=0.5):
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

    for epoch in range(n_epoch):
        epoch_metrics = None
        for x_in, y_in in zip(x_data_loader, y_data_loader):
            x_in.to(device=device)
            y_in.to(device=device)
            model.train()
            optimizer.zero_grad()

            x_out, edge_attr_out, global_attr_out = model(x_in.x, x_in.edge_index, x_in.edge_attr, x_in.u)
            loss = loss_func(y_in, x_out, edge_attr_out, global_attr_out)
            if accuracy_func is not None:
                acc = accuracy_func(y_in, x_out, edge_attr_out, global_attr_out)
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

        if epoch % print_every == 0:
            epoch_metrics_test = torch.zeros_like(epoch_metrics)
            if test_data is not None:
                x_test_data_loader, y_test_data_loader = test_data
                for x_in_test, y_in_test in zip(x_test_data_loader, y_test_data_loader):
                    x_in_test.to(device=device)
                    y_in_test.to(device=device)
                    model.eval()

                    x_out_test, edge_attr_out_test, global_attr_out_test = model(x_in_test.x,
                                                                                 x_in_test.edge_index,
                                                                                 x_in_test.edge_attr,
                                                                                 x_in_test.u)
                    loss_test = loss_func(y_in_test, x_out_test, edge_attr_out_test, global_attr_out_test)
                    if accuracy_func is not None:
                        acc_test = accuracy_func(y_in_test, x_out_test, edge_attr_out_test, global_attr_out_test)
                        epoch_metric_test = torch.cat((loss_test.view(1), acc_test))
                    else:
                        epoch_metric_test = loss_test

                    epoch_metrics_test += epoch_metric_test
            epoch_metrics_test = epoch_metrics_test / len(x_test_data_loader)
            epoch_metric_print_ready = [round(x, 3) for x in epoch_metrics.tolist()]
            epoch_metrics_test_print_ready = [round(x, 3) for x in epoch_metrics_test.tolist()]
            print("epoch %d: training metrics:" % epoch, epoch_metric_print_ready,
                  ", test metrics:", epoch_metrics_test_print_ready)


def train_sdf(model, train_data, loss_func=None, use_cpu=False,
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

    for epoch in range(n_epoch):
        epoch_loss = torch.tensor(0.)
        for data in train_data:
            data = data.to(device)
            model.train()
            optimizer.zero_grad()

            x_out, edge_attr_out, global_attr_out = model(data.x, data.edge_index, data.edge_attr.view(-1, 1), data.u)
            loss = loss_func(x_out, data.y)
            epoch_loss += loss
            loss.backward()
            optimizer.step()
        epoch_loss = epoch_loss / len(train_data)
        scheduler.step()

        if epoch % print_every == 0:
            print("epoch %d: training loss:%0.4f" % (epoch, epoch_loss.item()))
