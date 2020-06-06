from model import common
import torch
import torch.nn as nn

def make_model(args, parent=False):
    return NL_EST(args)

class NL_EST(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(NL_EST, self).__init__()
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        n_feats = 48
        kernel_size = 3
        self.scale_idx = 0

        act = nn.ReLU(True)

        # rgb_mean = (0.4488, 0.4371, 0.4040)
        # rgb_std = (1.0, 1.0, 1.0)

        self.DWT = common.DWT()
        self.IWT = common.IWT()


        m_head = [conv(args.n_colors*4+4, n_feats*2, kernel_size)]



        m_body = []
        for _ in range(10):
            m_body.append(common.BBlock(conv, n_feats*2, n_feats*2, kernel_size, act=act))
        # m_body.append(conv(n_feats, n_feats, kernel_size))
        #
        # self.upsample = nn.ModuleList([
        #     common.Upsampler(
        #         conv, s, n_feats, act=False
        #     ) for s in args.scale
        # ])

        m_tail = [conv(n_feats*2, args.n_colors*4, kernel_size)]

        # self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)

        self.tail = nn.Sequential(*m_tail)

    def forward(self, x, quality):
        # x = self.sub_mean(x)
        # x = torch.cat((x, quality), 1)
        
        x = torch.cat((x, quality), 1)
        x = self.DWT(x)
        
        x = self.head(x)
        #
        x = self.body(x)
        # res += x
        #
        # x = self.upsample[self.scale_idx](res)
        x = self.IWT(self.tail(x))
        # x = self.add_mean(x)

        return x

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx

