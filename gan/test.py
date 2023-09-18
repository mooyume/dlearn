import numpy as np

list = [1, 2, 3, 4, 5]
print(*list)

print(np.prod((3, 3, 3)))

import torch

# 创建一个形状为(3, 32, 32)的张量
img = torch.randn(1, 2, 28, 28)

img_flat = img.view(img.size(0), img.size(1), -1)
print(img_flat.size())
