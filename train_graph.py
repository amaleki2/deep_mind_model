import os
import sys
import torch
import meshio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri


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
            output = model(x_in.x, x_in.edge_index, x_in.edge_attr, x_in.u)

            if not hasattr(model, 'full_output') or model.full_output is False:
                x_out, edge_attr_out, global_attr_out = output
                loss = loss_func(y_in, x_out, edge_attr_out, global_attr_out)
            else:
                loss = [loss_func(y_in, x_out, edge_attr_out, global_attr_out)
                        for x_out, edge_attr_out, global_attr_out in output]
                loss = sum(loss) / len(loss)
                x_out, edge_attr_out, global_attr_out = output[-1]

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
        scheduler.step()

        if epoch % print_every == 0:
            epoch_metrics_test = torch.zeros_like(epoch_metrics)
            if test_data is not None:
                x_test_data_loader, y_test_data_loader = test_data
                for x_in_test, y_in_test in zip(x_test_data_loader, y_test_data_loader):
                    x_in_test.to(device=device)
                    y_in_test.to(device=device)
                    model.eval()

                    output_test = model(x_in_test.x, x_in_test.edge_index, x_in_test.edge_attr, x_in_test.u)
                    if hasattr(model, 'full_output') and model.full_output:
                        x_out_test, edge_attr_out_test, global_attr_out_test = output_test[-1]
                    else:
                        x_out_test, edge_attr_out_test, global_attr_out_test = output_test
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
            print("epoch %d: lr: %0.5e, training metrics:" % (epoch, optimizer.param_groups[0]['lr']),
                  epoch_metric_print_ready, ", test metrics:", epoch_metrics_test_print_ready)


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

            x_out, edge_attr_out, global_attr_out = model(data.x, data.edge_index, data.edge_attr, data.u)
            loss = loss_func(x_out, data.y)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        epoch_loss = epoch_loss / len(train_data)
        epoch_loss_list.append(epoch_loss)
        scheduler.step()

        if epoch % print_every == 0:
            print("epoch %d: training loss:%0.4f" % (epoch, epoch_loss))
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
            x_pred, _, _ = model(data.x, data.edge_index, data.edge_attr, data.u)
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
        #plt.figure(figsize=(8, 8))
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
            plt.hlines(1-border, -1+border, 1-border, 'r')
            plt.hlines(-1+border, -1+border, 1-border, 'r')
            plt.vlines(1-border, -1+border, 1-border, 'r')
            plt.vlines(-1+border, -1+border, 1-border, 'r')

    if node_labels:
            for i, (x, y) in enumerate(zip(nodes_x, nodes_y)):
                plt.text(x, y, i)

    if vals is not None:
        return p

