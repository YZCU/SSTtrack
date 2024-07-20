import torch
from torch import nn


class Bdater(nn.Module):
    def __init__(self):
        super().__init__()
        cl = 512
        mc = cl // 64
        self.ater_d = nn.Linear(cl, mc)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.ater_m = nn.Linear(mc, mc)
        self.ater_u = nn.Linear(mc, cl)
        self.ater_s = nn.Parameter(torch.ones(1))
        nn.init.zeros_(self.ater_d.weight)
        nn.init.zeros_(self.ater_d.bias)
        nn.init.zeros_(self.ater_m.bias)
        nn.init.zeros_(self.ater_m.weight)
        nn.init.zeros_(self.ater_u.weight)
        nn.init.zeros_(self.ater_u.bias)

    def forward(self, x):
        x_a = self.act(self.ater_d(x))
        x_up = self.ater_u(self.act(x_a + self.ater_m(self.dropout(x_a))))
        return x_up * self.ater_s
