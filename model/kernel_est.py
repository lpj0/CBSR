from model import common
import torch
import torch.nn as nn

def make_model(args, parent=False):
    return KERNEL_EST(args)

class KERNEL_EST(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(KERNEL_EST, self).__init__()
        n_resblocks = args.n_resblocks
        n_feats = 48#args.n_feats
        kernel_size = 3
        self.scale_idx = 0

        act = nn.ReLU(True)


        self.DWT = common.DWT()
        self.IWT = common.IWT()

        m_head = [conv(args.n_colors*4+5, n_feats*2, kernel_size)]

        m_body = []
        for _ in range(12):
            m_body.append(common.BBlock(conv, n_feats*2, n_feats*2, kernel_size, act=act))

        m_tail = [conv(n_feats*2, args.n_colors*4, kernel_size)]


        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x, quality):

        x = self.DWT(x)
        x = torch.cat((x, quality), 1)
        x = self.head(x)
        x = self.body(x)
        x = self.IWT(self.tail(x))

        return x

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx

