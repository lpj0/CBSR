from model import common
import torch
import torch.nn as nn
import scipy.io as sio

def make_model(args, parent=False):
    return CBSR(args)


def matrix_init():
    a = torch.linspace(1, 15, steps=15) - 8
    a = a.float()

    mat_1 = torch.mul(a, a)
    mat_1, mat_3 = torch.meshgrid([mat_1, mat_1])
    a = a.view(15, 1)
    mat_2 = torch.mul(a, a.t())
    mat_1 = mat_1.contiguous().view(1, 1, 225, 1, 1)
    mat_2 = mat_2.contiguous().view(1, 1, 225, 1, 1)
    mat_3 = mat_3.contiguous().view(1, 1, 225, 1, 1)

    return torch.cat((mat_1, mat_2, mat_3), 1).cuda()

class CBSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(CBSR, self).__init__()
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        self.scale_idx = 0

        act = nn.ReLU(True)

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)


        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        self.DWT = common.DWT()
        self.IWT = common.IWT()
        n = n_resblocks

        m_head = [conv(args.n_colors*4+20, n_feats*2, kernel_size+2)]
        # d_l1 = [common.BBlock(conv, n_feats*4, n_feats*2, 3, act=act)]
        d_l1 = []
        for _ in range(n):
            #d_l1.append(common.BBlock(conv, n_feats*2, n_feats*2, 3, act=act))
            d_l1.append(common.ResBlock(conv, n_feats*2, 3, act=act))
            d_l1.append(act)

        # dwt_l2 = [common.DWT]
        d_l2 = [common.BBlock(conv, n_feats * 8, n_feats * 4, 3, act=act)]
        for _ in range(n):
            d_l2.append(common.ResBlock(conv, n_feats*4, 3, act=act))
            d_l2.append(act)
            #d_l2.append(common.BBlock(conv, n_feats * 4, n_feats * 4, 3, act=act))

        # pro_l3 = [common.DWT]
        pro_l3 = [common.BBlock(conv, n_feats * 16, n_feats * 8, 3, act=act)]
        for _ in range(n*2):
            pro_l3.append(common.ResBlock(conv, n_feats*8, 3, act=act))
            pro_l3.append(act)
        # pro_l3.append(common.BBlock(conv, n_feats * 8, n_feats * 16, 3, act=act))
        pro_l3.append(conv(n_feats * 8, n_feats * 16, 3))
        # pro_l3.append(common.IWT)

        i_l2 = []
        for _ in range(n):
            #i_l2.append(common.BBlock(conv, n_feats * 4, n_feats * 4, 3, act=act))
            i_l2.append(common.ResBlock(conv, n_feats*4, 3, act=act))
            i_l2.append(act)
        # i_l2.append(common.BBlock(conv, n_feats * 4, n_feats * 8, 3, act=act))
        i_l2.append(conv(n_feats * 4, n_feats * 8, 3))
        # IWT = common.IWT
        # DWT = common.DWT

        i_l1 = []
        for _ in range(n):
            #i_l1.append(common.BBlock(conv, n_feats * 2, n_feats * 2, 3, act=act))
            i_l1.append(common.ResBlock(conv, n_feats*2, 3, act=act))
            i_l1.append(act)

        m_tail = [conv(n_feats*2, args.n_colors*4, kernel_size+4)]

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*m_head)
        # self.d_l0 = nn.Sequential(*d_l0)
        self.d_l1 = nn.Sequential(*d_l1)
        self.d_l2 = nn.Sequential(*d_l2)
        self.pro_l3 = nn.Sequential(*pro_l3)
        self.i_l2 = nn.Sequential(*i_l2)
        self.i_l1 = nn.Sequential(*i_l1)
        # self.i_l0 = nn.Sequential(*i_l0)
        self.tail = nn.Sequential(*m_tail)
        self.act = nn.Sequential(act)

    def forward(self, x_o, covmat):
        x = self.sub_mean(x_o)
        # covmat = self.UpSample(self.COV2PCA(covmat), x)
        x = torch.cat((self.DWT(x), covmat), 1)
        x1 = self.head(x)
        # x1 = self.d_l0(x0)
        x2 = self.d_l1(x1)
        x3 = self.d_l2(self.DWT(x2))
        x = self.act(self.IWT(self.pro_l3(self.DWT(x3))) + x3)
        x = self.act(self.IWT(self.i_l2(x)) + x2)
        x = self.act(self.i_l1(x)+ x1)
        # x = self.i_l0(x) + x0
        x = self.IWT(self.tail(x))
        x = self.add_mean(x) + x_o

        return x

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx

