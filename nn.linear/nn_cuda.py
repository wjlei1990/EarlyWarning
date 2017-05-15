from __future__ import print_function, division
import os
import sys
import time
import h5py
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np
import json
from sklearn.metrics import mean_squared_error
from obspy.signal.filter import envelope


NPTS = 60
input_size = 3 * NPTS + 1
hidden_size = 90
num_layers = 10
LR = 0.001
weight_decay=0.005
nepochs = 20
DROPOUT=0.01


torch.manual_seed(1)  # reproducible
CUDA_FLAG = torch.cuda.is_available()


def dump_json(data, fn):
    with open(fn, 'w') as fh:
        json.dump(data, fh, indent=2, sort_keys=True)


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(Net, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cuda_flag = torch.cuda.is_available()

        self.hidden0 = nn.Linear(input_size, hidden_size)
        self.hidden1 = nn.Linear(hidden_size, hidden_size)
        self.hidden2 = nn.Linear(hidden_size, hidden_size)
        self.predict = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # input layer
        x = F.relu(self.hidden0(x))
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        # predict layer
        x = self.predict(x)

        return x


def construct_nn(input_size, hidden_size, num_layers):
    model = nn.Sequential()
    # input layer
    model.add_module("input", nn.Linear(input_size, hidden_size))
    model.add_module("ReLU_input", nn.ReLU())
    #model.add_module("Tanh", nn.Tanh())

    # add hidden layer
    for idx in range(num_layers):
        model.add_module("hidden-%d" % idx, nn.Linear(hidden_size, hidden_size))
        model.add_module("ReLU-%d" % idx, nn.ReLU())
        #model.add_module("Tanh-%d" % idx, nn.Tanh())

    model.add_module("output", nn.Linear(hidden_size, 1))
    return model


def load_data(npts=60):
    print("Loading waveform npts: %d" % npts)
    t1 = time.time()
    f = h5py.File("./data/input.h5")
    #waveforms = np.array(f["waveform"])[:2000, :, :]
    #magnitudes = np.array(f["magnitude"])[:2000]
    train_x = np.array(f["train_x"])[:, :, 0:npts]
    train_y = np.array(f["train_y"])
    test_x = np.array(f["test_x"])[:, :, 0:npts]
    test_y = np.array(f["test_y"])
    train_d = np.array(f["train_distance"])
    test_d = np.array(f["test_distance"])
    t2 = time.time()
    print("Time used in reading data: %.2f sec" % (t2 - t1))
    print("train x and y shape: ", train_x.shape, train_y.shape)
    print("test x and y shape: ", test_x.shape, test_y.shape)
    print("train d and test d shape: ", train_d.shape, test_d.shape)
    return {"train_x": train_x, "train_y": train_y,
            "test_x": test_x, "test_y": test_y,
            "train_d": train_d, "test_d": test_d}


def make_dataloader(xs, ys):
    xs = torch.Tensor(xs).cuda()
    ys = torch.Tensor(ys).cuda()
    torch_dataset = Data.TensorDataset(data_tensor=xs, target_tensor=ys)
    loader = Data.DataLoader(dataset=torch_dataset, batch_size=1,
                             shuffle=True)
    return loader


def predict_on_test(net, test_x):
    print("Predict...")
    pred_y = []
    for idx in range(len(test_x)):
        x = test_x[idx, :]
        x = Variable(torch.unsqueeze(torch.Tensor(x), dim=0)).cuda()
        y_p = net(x)
        _y = float(y_p.cpu().data.numpy()[0])
        # print("pred %d: %f | true y: %f" % (idx, _y, test_y[idx]))
        pred_y.append(_y)

    return pred_y


def transfer_data_into_envelope(data):
    t1 = time.time()
    data_env = np.zeros(data.shape) 
    for idx1 in range(data.shape[0]):
        for idx2 in range(data.shape[1]):
            data_env[idx1, idx2, :] = envelope(data[idx1, idx2, :])
    t2 = time.time()
    print("Time used to convert envelope: %.2f sec" % (t2 - t1))

    return data_env


def transform_features(input_data, dtype="disp", dt=0.05,
                       envelope_flag=False):
    """
    Transform from displacement to a certrain data type,
    such as accelaration, velocity, or displacement itself.
    """
    print("[Transform]Input data shape before transform: ", input_data.shape)
    t1 = time.time()
    if dtype == "disp":
        data = input_data
    elif dtype == "vel":
        vel = np.gradient(input_data, dt, axis=2)
        data = vel
    elif dtype == "acc":
        vel = np.gradient(input_data, dt, axis=2)
        acc = np.gradient(vel, dt, axis=2)
        data = acc
    elif dtype == "acc_cumul_log_sum":
        vel = np.gradient(input_data, dt, axis=2)
        acc = np.gradient(vel, dt, axis=2)
        data = np.log(np.cumsum(np.abs(acc) * dt, axis=2) + 1)
    else:
        raise ValueError("unkonw dtype: %s" % dtype)
    
    if envelope_flag:
        data = transfer_data_into_envelope(data)

    t2 = time.time()
    print("time used in transform: %.2f sec" % (t2 - t1))
    return data


def add_distance_to_features(x, d):
    nlen = x.shape[1]
    x_new = np.zeros([x.shape[0], x.shape[1]+1])
    x_new[:, 0:nlen] = x[:, :]
    x_new[:, nlen] = np.log(d)

    print("[Add distance]shape change after adding distance as feature: ",
          x.shape, "-->", x_new.shape)
    return x_new


def combine_components_waveform(x):
    time_step = x.shape[2]
    print("time step in waveform: %d" % time_step)
    x_new = np.zeros([x.shape[0], time_step*3])
    for idx in range(len(x)):
        x_new[idx, 0:time_step] = x[idx, 0, :]
        x_new[idx, time_step:(2*time_step)] = x[idx, 1, :]
        x_new[idx, (2*time_step):(3*time_step)] = x[idx, 2, :]

    print("[Combine]shape change after combining components: ", x.shape, "-->",
          x_new.shape)
    return x_new


def standarize_features(train_x, test_x):
    vmax = np.max(np.abs(train_x))
    print("[Norm]max value of input waveform: %f" % vmax)
    return train_x / vmax, test_x / vmax


def load_and_process_features(data_split, dtype, envelope_flag):
    train_x = transform_features(data_split["train_x"], dtype=dtype,
                                 envelope_flag=envelope_flag)
    test_x = transform_features(data_split["test_x"], dtype=dtype,
                                envelope_flag=envelope_flag)

    train_x = combine_components_waveform(train_x)
    test_x = combine_components_waveform(test_x)

    train_x, test_x = standarize_features(train_x, test_x)

    train_x = add_distance_to_features(train_x, data_split["train_d"])
    test_x = add_distance_to_features(test_x, data_split["test_d"])
    return train_x, test_x


def main(outputdir, dtype, npts=60, envelope_flag=False):
    print("Working on dtype(%s) --- outputdir(%s)" % (dtype, outputdir))
    print("Envelope flag: %s" % envelope_flag)

    data_split = load_data(npts=npts)
    train_x, test_x = load_and_process_features(
        data_split, dtype, envelope_flag)
    train_loader = make_dataloader(train_x, data_split["train_y"])

    #net = Net(input_size, hidden_size, num_layers)
    net = construct_nn(input_size, hidden_size, num_layers)
    net.cuda()
    print(net)

    optimizer = torch.optim.Adam(net.parameters(), lr=LR,
                                 weight_decay=0.0005)
    loss_func = nn.MSELoss()

    # train
    ntest = data_split["train_x"].shape[0]
    all_loss = {}
    for epoch in range(nepochs):
        loss_epoch = []
        for step, (batch_x, batch_y) in enumerate(train_loader):
            if step % int((ntest/10) + 1) == 1:
                print('Epoch: ', epoch, '| Step: %d/%d' % (step, ntest),
                      "| Loss: %f" % np.mean(loss_epoch))
            if CUDA_FLAG:
                x = Variable(batch_x).cuda()
                y = Variable(torch.Tensor([batch_y.numpy(), ])).cuda()
            else:
                x = Variable(x)
                y = Variable(torch.Tensor([batch_y.numpy(), ]))

            prediction = net(x)
            loss = loss_func(prediction, y)
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()
            loss_epoch.append(loss.data[0])

        all_loss["epoch_%d" % epoch] = loss_epoch
        outputfn = os.path.join(outputdir, "loss.epoch_%d.json" % epoch)
        print("=== Mean loss in epoch(%d): %f(log: %s) ==="
              % (epoch, np.mean(loss_epoch), outputfn))
        dump_json(loss_epoch, outputfn)

    # test
    pred_y = predict_on_test(net, test_x)
    test_y = data_split["test_y"]
    _mse = mean_squared_error(test_y, pred_y)
    _std = np.std(test_y - pred_y)
    print("MSE and error std: %f, %f" % (_mse, _std))

    outputfn = os.path.join(outputdir, "prediction.json")
    print("output file: %s" % outputfn)
    data = {"test_y": list(test_y), "test_y_pred": list(pred_y),
            "epoch_loss": all_loss, "mse": _mse, "err_std": _std}
    dump_json(data, outputfn)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("input dtype please...exit")
        sys.exit()

    dtype = sys.argv[1]

    outputdir = "output.%s" % dtype
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    envelope_flag=True
    if dtype == "acc_cumul_log_sum":
        envelope_flag=False

    main(outputdir, dtype, npts=NPTS, envelope_flag=envelope_flag)
