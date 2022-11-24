import torch
import torch.nn as nn

from config import ModelParams


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1):
        super(CNNBlock, self).__init__()
        self.conv = nn.utils.weight_norm(nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride, padding, dilation,
        ))
        self.conv.weight.data.normal_(0, 0.01)
        self.conv.bias.data.normal_(0, 0.01)

        self.res = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else None
        if self.res is not None:
            self.res.weight.data.normal_(0, 0.01)
            self.res.bias.data.normal_(0, 0.01)

        # self.leaky_relu = nn.LeakyReLU(0.2)
        self.leaky_relu = nn.ELU()

    def forward(self, x):
        out = self.leaky_relu(self.conv(x))
        res = x if self.res is None else self.res(x)
        return out + res


class MyNet(nn.Module):
    def __init__(self, args: ModelParams):
        super().__init__()
        self.args = args
        self.mode = args.mode
        self.lstmSeq = nn.LSTM(input_size=5, hidden_size=30, num_layers=5, batch_first=True)  # 效果巨差
        self.cnn = nn.Sequential(
            CNNBlock(5, 16, 3, 1, 1, 1),
            CNNBlock(16, 32, 3, 1, 2, 2),
            CNNBlock(32, 32, 3, 1, 4, 4),
            nn.AdaptiveAvgPool2d((1, 8)),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Linear(32*8, 64),
            nn.Dropout(0.3),nn.ELU(),
            nn.Linear(64, 1),
            # nn.Dropout(0.3),nn.ELU(),
            # nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = None
        if self.mode == 'CNN':
            x = self.cnn(x)
            # x = torch.mean(x, dim=2)
            out = self.fc(x)
        if self.mode == 'LSTM':
            # input(batch, input_size, seq_len) turn to (batch, seq_len, input_size)
            x = x.permute(0, 2, 1).contiguous()
            h0 = torch.zeros(1, x.shape[0], 30)  # shape: (n_layers, batch, hidden_size)
            c0 = torch.zeros(5, x.shape[0], 30)
            out, (h_n, h_c) = self.lstmSeq(x, (h0, c0))
            out = self.fc(out[:, -1, :])
        return out

    def init_weight(self):
        pass
