import os

from data import common
from data import srdata

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data

class Benchmark(srdata.SRData):
    def __init__(self, args, train=True):
        super(Benchmark, self).__init__(args, train, benchmark=True)

    def _scan(self):
        # print(self.train)
        if self.train:
            list_hr = [i for i in range(self.num)]
            list_lr = list_hr
        else:
            list_hr = []
            list_lr = []
            # list_lr = [[] for _ in self.scale]
            for entry in os.scandir(self.dir_hr):
                filename = os.path.splitext(entry.name)[0]
                list_hr.append(os.path.join(self.dir_hr, filename + '.bmp'))

            for entry in os.scandir(self.dir_lr):
                filename = os.path.splitext(entry.name)[0]
                list_lr.append(os.path.join(self.dir_lr, filename + self.ext))
            list_hr.sort()
            list_lr.sort()
                # for si, s in enumerate(self.scale):
                #     list_lr[si].append(os.path.join(
                #         self.dir_lr,
                #         'X{}/{}x{}{}'.format(s, filename, s, self.ext)
                #     ))



        # for l in list_lr:
        #     l.sort()

        return list_lr, list_hr

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, 'benchmark', self.args.data_test)
        self.dir_lr = self.args.testpath  # 'Set5_x248_bic'
        self.dir_hr = '/home/genk/disks/D/Test/Set5/'
        #self.dir_hr = '/share/Dataset/Test/Set5/'
        # self.dir_lr = os.path.join(self.args.testpath, 'Test_Bic', 'x{}'.format(self.args.scale[0])) #'Set5_x248_bic'
        # self.dir_hr = os.path.join(self.args.testpath, 'GT', 'x{}'.format(self.args.scale[0])) #'Set5_x248_gt'
        # self.dir_hr = os.path.join(self.apath, 'HR')
        # self.dir_lr = os.path.join(self.apath, 'LR_bicubic')
        self.ext = self.args.ext_tt #'.bmp'
