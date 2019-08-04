from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

import torchvision.models as tvm


class STN(nn.Module):

    def __init__(self):
        super().__init__()
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        self.mid_dim = 10 * 60 * 60
        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(self.mid_dim, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, self.mid_dim)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x


class SmallNet(nn.Module):

    def __init__(self, out_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5, stride=2),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(10, 20, kernel_size=5, stride=2),
            nn.Dropout2d(),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
        )

        self.mid_dim = 20 * 15 * 15
        self.fc = nn.Sequential(
            nn.Linear(self.mid_dim, 100),
            nn.Linear(100, out_dim)
        )

    def forward(self, x):
        # Perform the usual forward pass
        x = self.conv(x)
        x = x.view(-1, self.mid_dim)
        x = self.fc(x)
        return x


class WindowOptimizer(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = SmallNet(out_dim=2)
        self.tanh = nn.Hardtanh()

    def forward(self, x):
        k, c = self.net(x).unsqueeze(2).unsqueeze(3).split(1, dim=1)
        x = torch.mul(x, k).add(c)
        x = self.tanh(x)
        return x


class Network(nn.Module):

    def __init__(self, out_dim=14, mode="per_image"):
        super().__init__()
        #self.stn = STN()
        #self.winopt = WindowOptimizer()

        #self.main = tvm.resnext101_32x8d(pretrained=True)
        #self.main.conv1 = nn.Conv2d(20, 64, kernel_size=7, stride=2, padding=3, bias=False)
        #self.main.fc = nn.Linear(self.main.fc.in_features, out_dim)
        self.main = tvm.densenet121(pretrained=False, drop_rate=0.5)
        self.main.features.conv0 = nn.Conv2d(20, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.custom = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ('fc0', nn.Linear(1000, 256)),
                ('do0', nn.Dropout(0.5)),
                ('fc1', nn.Linear(256, 256)),
                ('do1', nn.Dropout(0.5)),
                ('fc2', nn.Linear(256, 1)),
            ])) for _ in range(out_dim)
        ])
        self.mode = mode

    def to_distributed(self, device):
        #modules = self.main.features.__dict__.get('_modules')
        #
        #def closure(name):
        #    modules[name] = DistributedDataParallel(modules[name], device_ids=[device], output_device=device,
        #                                            find_unused_parameters=True)
        #
        #for name in modules.keys():
        #    if 'denseblock' in name: # and name != 'denseblock1':
        #        closure(name)
        #    if 'transition' in name: # and name != 'transition1':
        #        closure(name)

        self.main = DistributedDataParallel(self.main, device_ids=[device], output_device=device,
                                            find_unused_parameters=True)

    def forward(self, x):
        if self.mode == "per_image":
            x = self.stn(x)
            x = self.winopt(x)
            x = x.repeat(1, 3, 1, 1)
            x = self.main(x)
        elif self.mode == "per_study":
            x = self.main(x)
            xs = [m(x) for m in self.custom]
            x = torch.cat(xs, dim=1)
        else:
            raise RuntimeError

        return x


if __name__ == "__main__":
    m = Network()
    m.to_distributed("cuda:0")
