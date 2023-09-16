import numpy as np

list = [1, 2, 3, 4, 5]
print(*list)

print(np.prod((3, 3, 3)))

import torch

# 创建一个形状为(3, 32, 32)的张量
img = torch.randn(3, 32, 32)

# 使用size()函数获取张量的形状
print(img.size(1))  # 输出：torch.Size([3, 32, 32])

