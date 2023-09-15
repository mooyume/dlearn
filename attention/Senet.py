import torch
from torch import nn


# 通道注意力机制
class Senet(nn.Module):
    def __init__(self, channel, ratio=16):
        super(Senet, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, False),
            nn.ReLU(),
            nn.Linear(channel // ratio, channel, False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        avg = self.avg_pool(x).view([b, c])
        fc = self.fc(avg).view([b, c, 1, 1])
        return x * fc


model = Senet(512)
print(model)
inputs = torch.ones([2, 512, 26, 26])
outputs = model(inputs)
print(outputs)
