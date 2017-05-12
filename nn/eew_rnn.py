from __future__ import print_function, division
import time
import h5py
import torch
from torch import nn
from torch.autograd import Variable
import torch.utils.data as Data
import numpy as np

input_size = 3
hidden_size = 200
num_layers = 3
LR = 0.001

torch.manual_seed(1)  # reproducible


class RNN(nn.Module):
    # RNN Model (Many-to-One)
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Set initial states
        h0 = Variable(torch.zeros(
            self.num_layers, x.size()[1], self.hidden_size))
        c0 = Variable(torch.zeros(
            self.num_layers, x.size()[1], self.hidden_size))

        # Forward propagate RNN
        out, _ = self.lstm(x, (h0, c0))

        # Decode hidden state of last time step
        out = self.fc(out[-1, :, :])
        return out


def load_data():
    t1 = time.time()
    f = h5py.File("./test.h5")
    print(f.keys())
    waveforms = f["waveform"]
    magnitudes = f["magnitude"]
    t2 = time.time()
    print("Time used in reading data: %.2f sec" % (t2 - t1))
    print("waveforms and magnitude shape: {0} and {1}".format(
        waveforms.shape, magnitudes.shape))
    return waveforms, magnitudes


def make_dataloader(xs, ys):
    xs = torch.Tensor(xs)
    ys = torch.Tensor(ys)
    torch_dataset = Data.TensorDataset(data_tensor=xs, target_tensor=ys)
    loader = Data.DataLoader(dataset=torch_dataset, batch_size=1,
                             shuffle=True)
    return loader


def main():
    waveforms, magnitudes = load_data()
    loader = make_dataloader(waveforms, magnitudes)

    rnn = RNN(input_size, hidden_size, num_layers)
    print(rnn)

    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
    loss_func = nn.MSELoss()

    for epoch in range(3):
        loss_epoch = []
        for step, (batch_x, batch_y) in enumerate(loader):
            x = torch.unsqueeze(batch_x[0, :, :].t(), dim=1)
            print('Epoch: ', epoch, '| Step: ', step, '| x: ',
                  x.size(), '| y: ', batch_y.numpy())
            x = Variable(x)
            y = Variable(torch.Tensor([batch_y.numpy(), ]))
            prediction = rnn(x)
            loss = loss_func(prediction, y)
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()
            loss_epoch.append(loss.data[0])
            print("Current loss: %e --- loss mean: %f"
                  % (loss.data[0], np.mean(loss_epoch)))


if __name__ == "__main__":
    main()
