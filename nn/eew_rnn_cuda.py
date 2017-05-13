from __future__ import print_function, division
import os
import time
import h5py
import torch
from torch import nn
from torch.autograd import Variable
import torch.utils.data as Data
import numpy as np
import json
from sklearn.metrics import mean_squared_error


input_size = 3
hidden_size = 500
num_layers = 4
LR = 0.001

torch.manual_seed(1)  # reproducible
CUDA_FLAG = torch.cuda.is_available()


def dump_json(data, fn):
    with open(fn, 'w') as fh:
        json.dump(data, fh, indent=2, sort_keys=True)


class RNN(nn.Module):
    # RNN Model (Many-to-One)
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, 1)
        self.cuda_flag = torch.cuda.is_available()

    def forward(self, x):
        # Set initial states
        h0 = Variable(torch.zeros(
            self.num_layers, x.size()[1], self.hidden_size))
        c0 = Variable(torch.zeros(
            self.num_layers, x.size()[1], self.hidden_size))
        if self.cuda_flag:
            h0 = h0.cuda()
            c0 = c0.cuda()

        # Forward propagate RNN
        out, _ = self.lstm(x, (h0, c0))

        # Decode hidden state of last time step
        out = self.fc(out[-1, :, :])
        return out


def load_data():
    t1 = time.time()
    f = h5py.File("./test.h5")
    print(f.keys())
    #waveforms = np.array(f["waveform"])[:10000, :, :]
    #magnitudes = np.array(f["magnitude"])[:10000]
    waveforms = np.abs(np.array(f["waveform"]))
    magnitudes = np.array(f["magnitude"])
    t2 = time.time()
    print("Time used in reading data: %.2f sec" % (t2 - t1))
    print("waveforms and magnitude shape: {0} and {1}".format(
        waveforms.shape, magnitudes.shape))
    return waveforms, magnitudes


def split_data(x, y, train_percentage=0.8):
    msk = np.random.rand(len(y)) < train_percentage
    train_x = x[msk]
    train_y = y[msk]
    test_x = x[~msk]
    test_y = y[~msk]
    return {"train_x": train_x, "train_y": train_y,
            "test_x": test_x, "test_y": test_y}


def make_dataloader(xs, ys):
    xs = torch.Tensor(xs).cuda()
    ys = torch.Tensor(ys).cuda()
    torch_dataset = Data.TensorDataset(data_tensor=xs, target_tensor=ys)
    loader = Data.DataLoader(dataset=torch_dataset, batch_size=1,
                             shuffle=True)
    return loader


def predict_on_test(rnn, test_x):
    print("Predict...")
    pred_y = []
    for idx in range(len(test_x)):
        x = test_x[idx, :, :]
        x = Variable(torch.unsqueeze(torch.Tensor(x).t(), dim=1)).cuda()
        y_p = rnn(x)
        _y = float(y_p.cpu().data.numpy()[0])
        # print("pred %d: %f | true y: %f" % (idx, _y, test_y[idx]))
        pred_y.append(_y)

    return pred_y


def main():
    outputdir = "output.disp.abs"
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    waveforms, magnitudes = load_data()
    data_split = split_data(waveforms, magnitudes, train_percentage=0.9)
    print("dimension of train x and y: ", data_split["train_x"].shape,
          data_split["train_y"].shape)
    print("dimension of test x and y: ", data_split["test_x"].shape,
          data_split["test_y"].shape)
    train_loader = make_dataloader(data_split["train_x"],
                                   data_split["train_y"])

    rnn = RNN(input_size, hidden_size, num_layers)
    rnn.cuda()
    print(rnn)

    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
    loss_func = nn.MSELoss()

    # train
    ntest = data_split["train_x"].shape[0]
    all_loss = {}
    for epoch in range(3):
        loss_epoch = []
        for step, (batch_x, batch_y) in enumerate(train_loader):
            x = torch.unsqueeze(batch_x[0, :, :].t(), dim=1)
            if step % int((ntest/100) + 1) == 1:
                print('Epoch: ', epoch, '| Step: %d/%d' % (step, ntest),
                      "| Loss: %f" % np.mean(loss_epoch))
            if CUDA_FLAG:
                x = Variable(x).cuda()
                y = Variable(torch.Tensor([batch_y.numpy(), ])).cuda()
            else:
                x = Variable(x)
                y = Variable(torch.Tensor([batch_y.numpy(), ]))
            prediction = rnn(x)
            loss = loss_func(prediction, y)
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()
            loss_epoch.append(loss.data[0])
        all_loss["epoch_%d" % epoch] = loss_epoch

        outputfn = os.path.join(outputdir, "loss.epoch_%d.json" % epoch)
        print("epoch loss file: %s" % outputfn)
        dump_json(loss_epoch, outputfn)

    # test
    pred_y = predict_on_test(rnn, data_split["test_x"])
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
    main()
