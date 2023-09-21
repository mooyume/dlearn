import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.utils import save_image

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # 内部函数,一个块包含 全连接层->BN层（根据传入参数判断是否需要）-> LeakyReLU层
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                # 0.8: 动量值，用于计算运行时的平均值和方差。动量值通常设置在0到1之间。较高的值会给予过去的统计信息更多的权重，使得平均值和方差的计算更加稳定。
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            #     当输入值小于0时，LeakyReLU函数会返回0.2 * input。
            # 这个参数决定了是否在原地进行操作。如果设置为True，那么在进行前向计算时，它会直接修改输入数据，而不需要额外的空间来存储输出。这可以节省一些内存，但可能会覆盖原始数据。
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            # 输入1024   输出: 图片的总像素数（图像形状的乘积）
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        # torch.Size([64, 784])
        # 64为batch_size(也就是z的batch_size,这里我们前面设置的是64) 784 为图像转换为一维向量大小  28*28=784 model的最后一个线性层输出的是np.prod(img_shape)
        print(img.size())
        # view()函数用于改变张量的形状   torch.Size([64, 1, 28, 28])
        img = img.view(img.size(0), *img_shape)
        print(f'after view() shape is {img.size()}')
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        # 第一个维度为 img.size(0)，将其他维度自动展开为一维 此处img.size()为torch.Size([64, 1, 28, 28]),img_flat.size()=torch.Size([64, 784])
        print(f'D --> img.size()=:{img.size()}')
        img_flat = img.view(img.size(0), -1)
        print(f'D --> img_flat.size()=:{img_flat.size()}')
        validity = self.model(img_flat)

        return validity


# Loss function    二元交叉熵损失函数
# 用于衡量生成器生成的图像和真实图像之间的差异。
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

# 使用GPU计算
if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Configure data loader
os.makedirs("../../../data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../../data/mnist",
        train=True,
        download=True,
        #
        transform=transforms.Compose(
            [
                # 调整图像大小
                transforms.Resize(opt.img_size),
                # 这个操作将PIL Image或NumPy ndarray转换为torch.FloatTensor，并将图像的形状从(H,W,C)变为(C,H,W)，其中H是高度，W是宽度，C是通道数。
                # 这是因为PyTorch中的卷积操作要求输入的形状为(C,H,W)。
                transforms.ToTensor(),
                # 对图像进行归一化。它需要两个参数，分别是均值和标准差。在这里，均值和标准差都设置为0.5，所以这个操作会将图像的像素值从[0,1]变为[-1,1]。
                transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    # 在每个训练周期开始时打乱数据
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(
    # 生成器的所有可学习参数
    generator.parameters(),
    # 上面指定的学习率
    lr=opt.lr,
    # 设置了Adam优化器的两个超参数，这两个参数控制了梯度的指数加权平均的衰减率。
    betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
# 根据是否存在cuda来选择合适的张量类型
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        # Configure input  将图像(真实图像)转换为适当的（GPU or CPU）张量
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        # 清除生成器优化器中的梯度
        optimizer_G.zero_grad()

        # Sample noise as generator input  生成随机噪音z
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images  根据z生成一批图像
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator   计算判别器的判别结构与真实标签之间的二元交叉熵损失
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        # 反向传播误差并更新生成器的参数。
        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
