import torch.nn as nn
import torch.nn.functional as F


# 残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        # 卷积块
        conv_block = [nn.ReflectionPad2d(1),
                      # 二维卷积层
                      nn.Conv2d(in_features, in_features, 3),
                      # 归一化层
                      nn.InstanceNorm2d(in_features),
                      nn.ReLU(inplace=True),
                      # 对输入进行填充
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        # 返回输入和卷积块的输出之和  ，x为输入
        return x + self.conv_block(x)


# 生成器
class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block       
        model = [nn.ReflectionPad2d(3),  # 对输入填充
                 nn.Conv2d(input_nc, 64, 7),
                 nn.InstanceNorm2d(64),  # 归一化层
                 nn.ReLU(inplace=True)]

        # Downsampling   下采样  使得特征图的尺寸减半，特征的数量增加一倍
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling   上采样
        # 每次上采样操作后，输入特征的数量变为输出特征的数量，输出特征的数量变为其自身的一半。这样，每次上采样都会使特征图的尺寸增大一倍，特征的数量减少一半。
        out_features = in_features // 2
        for _ in range(2):
            model += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),  # 转置卷积层
                      nn.InstanceNorm2d(out_features),  # 归一化层
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2

        # Output layer  将特征图转换为输出图像
        model += [nn.ReflectionPad2d(3),  # 对输入进行填充，反射填充可以在一定程度上保持填充像素与原始像素的一致性，从而减少填充带来的影响
                  nn.Conv2d(64, output_nc, 7),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


# 判别器
class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        # (输入通道数，输出通道数，卷积核大小，步长，填充)
        model = [nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(64, 128, 4, stride=2, padding=1),
                  # 归一化层     128 为输入通道数（也就是上一层的输出通道数）
                  nn.InstanceNorm2d(128),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(128, 256, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(256),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(256, 512, 4, padding=1),
                  nn.InstanceNorm2d(512),
                  nn.LeakyReLU(0.2, inplace=True)]

        # FCN classification layer   全卷积层（FCN）
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        # Average pooling and flatten      x.size()  [1,1,30,30]   x.size()[2:]  [30,30]
        print(x.size()[2:])
        # avg_pool2d(输入张量，池化窗口大小) 平均池化操作     view(新张量的第一个维度，第二个维度) -1表示这个维度带大小会被自动计算
        # 当x形状为[2,3,4]时，执行x.view(x.size()[0], -1)后的新张量的形状为[2,12]第一位和x一致，剩下的维度自动计算
        y = F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
        # y.size() = torch.Size([1, 1])
        print(y.size())
        return y
