import torch
from torch import nn


# 通道注意力
class channel_attention(nn.Module):
    def __init__(self, channel, ratio=16):
        super(channel_attention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, False),
            nn.ReLU(),
            nn.Linear(channel // ratio, channel, False),
            nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        max_pool = self.max_pool(x).view([b, c])
        avg_pool = self.avg_pool(x).view([b, c])
        max_fc = self.fc(max_pool)
        avg_fc = self.fc(avg_pool)
        print(max_fc.shape)
        out = max_fc + avg_fc
        out = self.sigmoid(out).view([b, c, 1, 1])
        return out * x


# 空间注意力机制
class spatial_attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(spatial_attention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, 1, padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # torch.max 返回一个元组，需要用 _ 来接受参数，或者直接使用如下代码：max_pool_out = torch.max(x, dim=1, keepdim=True).values
        max_pool_out, _ = torch.max(x, dim=1, keepdim=True)
        mean_pool_out = torch.mean(x, dim=1, keepdim=True)
        pool_out = torch.cat([max_pool_out, mean_pool_out], dim=1)
        out = self.sigmoid(self.conv(pool_out))
        return out * x


class Cbam(nn.Module):
    def __init__(self, channel, ratio=16, kernel_size=7):
        super(Cbam, self).__init__()
        self.channel_attention = channel_attention(channel, ratio)
        self.spatial_attention = spatial_attention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


model = Cbam(512)
print(model)
inputs = torch.ones([2, 512, 26, 26])
outputs = model(inputs)
