import torch
import torch.nn as nn
import torch.nn.functional as F


# DeepHiC Function
def swish(x):
    return x * torch.sigmoid(x)
# 
# 
# DeepHiC Residual Block
# class residualBlock(nn.Module):
#     def __init__(self, channels, k=3, s=1):
#         super(residualBlock, self).__init__()
# 
#         self.conv1 = nn.Conv2d(channels, channels, k, stride=s, padding=1)
#         self.bn1 = nn.BatchNorm2d(channels)
#         a swish layer here
#         self.conv2 = nn.Conv2d(channels, channels, k, stride=s, padding=1)
#         self.bn2 = nn.BatchNorm2d(channels)
# 
#     def forward(self, x):
#         residual = swish(self.bn1(self.conv1(x)))
#         residual =       self.bn2(self.conv2(residual))
#         return x + residual
# 
# 
# DeepHiC Generator
# class Generator(nn.Module):
#     def __init__(self, scale_factor, in_channel=3, resblock_num=5):
#         super(Generator, self).__init__()
#         self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=9, stride=1, padding=4)
#         have a swish here in forward
#         
#         resblocks = [residualBlock(64) for _ in range(resblock_num)]
#         self.resblocks = nn.Sequential(*resblocks)
# 
#         self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm2d(64)
#         have a swish here in forward
# 
#         self.conv3 = nn.Conv2d(64, in_channel, kernel_size=9, stride=1, padding=4)
# 
#     def forward(self, x):
#         emb = swish(self.conv1(x))
#         x   =       self.resblocks(emb)
#         x   = swish(self.bn2(self.conv2(x)))
#         x   =       self.conv3(x + emb)
#         return (torch.tanh(x) + 1) / 2

class Basic_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

    def forward(self, x):
        """
        Applies a 1x1, 2D convolutional layer to the input image.
        Args: Matrix of dim=4
        """
        out = self.conv(x)

        return out


class Residual_Block(nn.Module):
    def __init__(self, num_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x):
        """
        Applies the following sequence of functions to the input image:
            3x3 Conv --> Relu --> 3x3 Conv + input --> Relu.
        Args: Matrix of dim=4
        """
        out = F.relu(self.conv1(x), inplace=True)
        out = F.relu(self.conv2(out) + x, inplace=True)

        return out


class Cascading_Block(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.r1 = Residual_Block(channels)
        self.r2 = Residual_Block(channels)
        self.r3 = Residual_Block(channels)
        self.c1 = Basic_Block(channels * 2, channels)
        self.c2 = Basic_Block(channels * 3, channels)
        self.c3 = Basic_Block(channels * 4, channels)

    def forward(self, x):
        """
        Applies the following sequence of functions to the input 3 times:
            Residual Block --> Add previous block input --> 1x1 Conv.
        """
        c0 = o0 = x

        b1 = self.r1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)

        b2 = self.r2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)

        b3 = self.r3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        return o3

class Generator(nn.Module):
    def __init__(self, scale_factor, num_channels):
        super().__init__()

        #self.sub_mean = Ops.MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        #self.add_mean = Ops.MeanShift(, sub=False)

        # Entry 3x3 convolution layer
        self.entry = nn.Conv2d(1, num_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        # Cascading blocks
        self.cb1 = Cascading_Block(num_channels)
        self.cb2 = Cascading_Block(num_channels)
        self.cb3 = Cascading_Block(num_channels)

        # Body 1x1 convolution layers
        self.cv1 = nn.Conv2d(num_channels * 2, num_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.cv2 = nn.Conv2d(num_channels * 3, num_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.cv3 = nn.Conv2d(num_channels * 4, num_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

        # 3x3 exit convolution layer
        self.exit = nn.Conv2d(num_channels, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x):
        """
        Applies the following sequence of functions to the input:
           3x3 entry Conv --> 3 * (Cascading Block --> Add previous block input --> 1x1 Conv) --> 3x3 exit Conv.
        """
        #x = self.sub_mean(x)
        x = self.entry(x)
        c0 = o0 = x
        b1 = self.cb1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.cv1(c1)

        b2 = self.cb2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.cv2(c2)

        b3 = self.cb3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.cv3(c3)

        out = self.exit(o3)
        #out = self.add_mean(out)
        return out

# DeepHiC Discriminator
class Discriminator(nn.Module):
    def __init__(self, in_channel=3):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 64, 3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        # Replaced original paper FC layers with FCN
        self.conv7 = nn.Conv2d(256, 1, 1, stride=1, padding=0)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size = x.size(0)

        x = swish(self.conv1(x))
        x = swish(self.bn2(self.conv2(x)))
        x = swish(self.bn3(self.conv3(x)))
        x = swish(self.bn4(self.conv4(x)))
        x = swish(self.bn5(self.conv5(x)))
        x = swish(self.bn6(self.conv6(x)))

        x = self.conv7(x)
        x = self.avgpool(x)
        return torch.sigmoid(x.view(batch_size))


# CARN Network
# class Basic_Block(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
#
#     def forward(self, x):
#         """
#         Applies a 1x1, 2D convolutional layer to the input image.
#         Args: Matrix of dim=4
#         """
#         out = self.conv(x)
#
#         return out
#
#
# class Residual_Block(nn.Module):
#     def __init__(self, num_channels):
#         super().__init__()
#
#         self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#
#     def forward(self, x):
#         """
#         Applies the following sequence of functions to the input image:
#             3x3 Conv --> Relu --> 3x3 Conv + input --> Relu.
#         Args: Matrix of dim=4
#         """
#         out = F.relu(self.conv1(x), inplace=True)
#         out = F.relu(self.conv2(out) + x, inplace=True)
#
#         return out
#
#
# class Cascading_Block(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#
#         self.r1 = Residual_Block(channels)
#         self.r2 = Residual_Block(channels)
#         self.r3 = Residual_Block(channels)
#         self.c1 = Basic_Block(channels * 2, channels)
#         self.c2 = Basic_Block(channels * 3, channels)
#         self.c3 = Basic_Block(channels * 4, channels)
#
#     def forward(self, x):
#         """
#         Applies the following sequence of functions to the input 3 times:
#             Residual Block --> Add previous block input --> 1x1 Conv.
#         """
#         c0 = o0 = x
#
#         b1 = self.r1(o0)
#         c1 = torch.cat([c0, b1], dim=1)
#         o1 = self.c1(c1)
#
#         b2 = self.r2(o1)
#         c2 = torch.cat([c1, b2], dim=1)
#         o2 = self.c2(c2)
#
#         b3 = self.r3(o2)
#         c3 = torch.cat([c2, b3], dim=1)
#         o3 = self.c3(c3)
#
#         return o3

# class Net(nn.Module):
#     def __init__(self, num_channels, scale=4):
#         super().__init__()
#
#         #self.sub_mean = Ops.MeanShift((0.4488, 0.4371, 0.4040), sub=True)
#         #self.add_mean = Ops.MeanShift(, sub=False)
#
#         # Entry 3x3 convolution layer
#         self.entry = nn.Conv2d(1, num_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#
#         # Cascading blocks
#         self.cb1 = Blocks.Cascading_Block(num_channels)
#         self.cb2 = Blocks.Cascading_Block(num_channels)
#         self.cb3 = Blocks.Cascading_Block(num_channels)
#
#         # Body 1x1 convolution layers
#         self.cv1 = nn.Conv2d(num_channels * 2, num_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
#         self.cv2 = nn.Conv2d(num_channels * 3, num_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
#         self.cv3 = nn.Conv2d(num_channels * 4, num_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
#
#         # Upsampling
#         self.upsample = Ops.UpsampleBlock(num_channels, scale=scale)
#
#         # 3x3 exit convolution layer
#         self.exit = nn.Conv2d(num_channels, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#
#     def forward(self, x, scale):
#         """
#         Applies the following sequence of functions to the input:
#            3x3 entry Conv --> 3 * (Cascading Block --> Add previous block input --> 1x1 Conv) --> 3x3 exit Conv.
#         """
#         #x = self.sub_mean(x)
#         x = self.entry(x)
#         c0 = o0 = x
#         b1 = self.cb1(o0)
#         c1 = torch.cat([c0, b1], dim=1)
#         o1 = self.cv1(c1)
#
#         b2 = self.cb2(o1)
#         c2 = torch.cat([c1, b2], dim=1)
#         o2 = self.cv2(c2)
#
#         b3 = self.cb3(o2)
#         c3 = torch.cat([c2, b3], dim=1)
#         o3 = self.cv3(c3)
#
#         out = self.exit(o3)
#         #out = self.add_mean(out)
#         return out
#
# import math
# import torch
# import torch.nn as nn
# from torch.functional import F
#
#
# class MeanShift(nn.Module):
#     def __init__(self, mean_if, sub):
#         super(MeanShift, self).__init__()
#
#         sign = -1 if sub else 1
#         n = mean_if * sign
#
#         self.shifter = nn.Conv2d(3, 3, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
#         self.shifter.weight.data = torch.eye(3).view(3, 3, 1, 1)
#         self.shifter.bias.data = torch.Tensor([n])
#
#         # Freeze the mean shift layer
#         for params in self.shifter.parameters():
#             params.requires_grad = False
#
#     def forward(self, x):
#         x = self.shifter(x)
#         return x
#
#
# class UpsampleBlock(nn.Module):
#     def __init__(self, n_channels, scale, groups=1):
#         super().__init__()
#
#         self.up = _UpsampleBlock(n_channels, scale=scale, groups=groups)
#
#     #     self.up4 = _UpsampleBlock(n_channels, scale=4, groups=groups)
#     #
#     def forward(self, x, scale):
#         return self.up(x)
#         # if scale == 2:
#         #     return self.up2(x)
#         # if scale == 3:
#         #     return self.up3(x)
#         # if scale == 4:
#         #     return self.up4(x)
#         # raise NotImplementedError
#
#
# class _UpsampleBlock(nn.Module):
#     def __init__(self, n_channels, scale, groups=1):
#         super().__init__()
#
#         self.body = nn.ModuleList()
#         if scale in [2, 4, 8]:
#             for _ in range(int(math.log(scale, 2))):
#                 self.body.append(
#                     nn.Conv2d(n_channels, 4*n_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
#                     groups=groups)
#                 )
#                 self.body.append(nn.ReLU(inplace=True))
#                 self.body.append(nn.PixelShuffle(2))
#         else:
#             self.body.append(
#                 nn.Conv2d(n_channels, 16*n_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=groups)
#             )
#             self.body.append(nn.ReLU(inplace=True))
#             self.body.append(nn.PixelShuffle(5))
#
#     def forward(self, x):
#         out = x
#         for layer in self.body:
#             out = layer(out)
#         return out
#
# import math
# import functools
# import torch.nn as nn
#
#
# def default(m, scale=1.0):
#     classname = m.__class__.__name__
#     if classname.find("Conv") != -1:
#         if m.weight.requires_grad:
#             nn.init.kaiming_uniform_(m.weight.data, a=math.sqrt(5))
#             m.weight.data *= scale
#         if m.bias is not None and m.bias.requires_grad:
#             m.bias.data.zero_()
#
#
# def msra_uniform(m, scale=1.0):
#     classname = m.__class__.__name__
#     if classname.find("Conv") != -1:
#         if m.weight.requires_grad:
#             nn.init.kaiming_uniform_(m.weight.data, a=0)
#             m.weight.data *= scale
#         if m.bias is not None and m.bias.requires_grad:
#             m.bias.data.zero_()
#
#
# def msra_normal(m, scale=1.0):
#     classname = m.__class__.__name__
#     if classname.find("Conv") != -1:
#         if m.weight.requires_grad:
#             nn.init.kaiming_normal_(m.weight.data, a=0)
#             m.weight.data *= scale
#         if m.bias is not None and m.bias.requires_grad:
#             m.bias.data.zero_()
#
#
# def init_weights(net, init_type, scale=1.0):
#     if init_type == "default":
#         _init = functools.partial(default, scale=scale)
#     elif init_type == "msra_uniform":
#         _init = functools.partial(msra_uniform, scale=scale)
#     elif init_type == "msra_normal":
#         _init = functools.partial(msra_normal, scale=scale)
#     else:
#         raise NotImplementedError
#
#     net.apply(_init)
#
# PCARN Discriminator
# class Discriminator(nn.Module):
#     def __init__(self, downsample=1):
#         super().__init__()
#
#         def conv_bn_lrelu(in_channels, out_channels, ksize, stride, pad):
#             return nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
#                 nn.BatchNorm2d(out_channels),
#                 nn.LeakyReLU()
#             )
#
#         self.body = nn.Sequential(
#             nn.Conv2d(3, 64, 3, 1, 1), nn.LeakyReLU(),
#             conv_bn_lrelu(64, 64, 4, 2, 1),
#             conv_bn_lrelu(64, 128, 3, 1, 1),
#             conv_bn_lrelu(128, 128, 4, 2, 1),
#             conv_bn_lrelu(128, 256, 3, 1, 1),
#             conv_bn_lrelu(256, 256, 4, 2, 1),
#             conv_bn_lrelu(256, 512, 3, 1, 1),
#             conv_bn_lrelu(512, 512, 3, 1, 1),
#             nn.Conv2d(512, 1, 3, 1, 1),
#             nn.Sigmoid()
#         )
#
#         if downsample > 1:
#             self.avg_pool = nn.AvgPool2d(downsample)
#
#         self.downsample = downsample
#
#     def forward(self, x):
#         if self.downsample > 1:
#             x = self.avg_pool(x)
#
#         out = self.body(x)
#         return out
