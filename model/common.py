import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


def matrix_init():
    a = torch.linspace(1, 15, steps=15) - 8
    a = a.float()

    mat_1 = torch.mul(a, a)
    mat_1, mat_3 = torch.meshgrid([mat_1, mat_1])
    a = a.view(15, 1)
    mat_2 = torch.mul(a, a.t())
    mat_1 = mat_1.contiguous().view(225, 1, 1, 1)
    mat_2 = mat_2.contiguous().view(225, 1, 1, 1)
    mat_3 = mat_3.contiguous().view(225, 1, 1, 1)

    return torch.cat((mat_1, mat_2, mat_3), 1).cuda()


def dwt_init(x):
    in_batch, in_channel, in_height, in_width = x.size()
    # h_list_1 = [i for i in range(0, in_height, 2)]
    # h_list_2 = [i for i in range(1, in_height, 2)]
    # w_list_1 = [i for i in range(0, in_width, 2)]
    # w_list_2 = [i for i in range(1, in_width, 2)]
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)

def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    # h_list_1 = [i for i in range(0, out_height, 2)]
    # h_list_2 = [i for i in range(1, out_height, 2)]
    # w_list_1 = [i for i in range(0, out_width, 2)]
    # w_list_2 = [i for i in range(1, out_width, 2)]

    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    # h = x.reshape(r, r, out_batch, in_channel, in_height, in_width)
    # h = h.transpose(2, 3, 4, 0, 5, 1)
    return h  # h.reshape(out_batch, out_channel, out_height, out_width)


class COV2PCA(nn.Module):
    def __init__(self, matrix, V_pca):
        super(COV2PCA, self).__init__()
        self.requires_grad = False
        # self.volatile = True
        self.matrix = matrix
        self.V_pca = V_pca

    def forward(self, x):
        return cov2pca(self.matrix, self.V_pca, x)
    # def backward(self, x, y):
    #     return torch.zeros(1).float().cuda()

def cov2pca(matrix, V_pca, x):

    sum_weight = torch.ones([1, 225, 1, 1]).float().cuda()
    # print(x[:, 2, :, :])
    # print(x.shape)
    # sum_weight15 = torch.ones([1, 15, 1, 1]).float().cuda()

    cov_a = F.relu(x[:, :1, :, :], inplace=True) #* 255.0 #+ 4e-4
    cov_b = x[:, 1:2, :, :] #* 255.0
    cov_c = F.relu(x[:, 2:, :, :], inplace=True) #* 255.0 #+ 4e-4
    cov_inv_denominator_1 = cov_a * cov_c
    cov_b_2 = cov_b * cov_b
    cov_inv_denominator = cov_inv_denominator_1 - cov_b_2
    gt_idx = torch.gt(cov_inv_denominator, 0).float()
    cov_b_2 = cov_b_2 * gt_idx

    # cov_inv_denominator = cov_inv_denominator * gt_idx + 1e-5
    cov_inv_denominator = cov_inv_denominator_1 - cov_b_2 + 1e-8

    inv_covmat = torch.cat((cov_c, -2*cov_b*gt_idx, cov_a), 1) / cov_inv_denominator

    kernel = torch.exp(-0.5 * F.conv2d(inv_covmat, matrix))

    kernel_sum = F.conv2d(kernel, sum_weight)

    kernel = kernel / kernel_sum

    kernel = F.conv2d(kernel, V_pca)

    return kernel



class UpSample(nn.Module):
    def __init__(self):
        super(UpSample, self).__init__()

    def forward(self, x, y):
        _, _, hei, wid = y.size()
        return nn.functional.interpolate(x, [hei, wid], mode='nearest')

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)

class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False
        if sign==-1:
            self.create_graph = False
            self.volatile = True
class MeanShift2(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift2, self).__init__(8, 8, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(8).view(8, 8, 1, 1)
        self.weight.data.div_(std.view(8, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False
        if sign==-1:
            self.volatile = True

class BasicBlock(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=False, act=nn.ReLU(True)):

        m = [nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size//2), stride=stride, bias=bias)
        ]
        if bn: m.append(nn.BatchNorm2d(out_channels))
        if act is not None: m.append(act)
        super(BasicBlock, self).__init__(*m)

class BBlock(nn.Module):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(BBlock, self).__init__()
        m = []
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias))
        if bn: m.append(nn.BatchNorm2d(out_channels))
        m.append(act)


        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x).mul(self.res_scale)
        return x

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Block(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(Block, self).__init__()
        m = []
        for i in range(4):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        # res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)




