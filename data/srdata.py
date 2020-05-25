import os

from data import common

import numpy as np
import scipy.misc as misc
import scipy.io as sio
from imageio import imread
from data.common import imresize_np

import h5py
import torch
import torch.utils.data as data

class SRData(data.Dataset):
    def __init__(self, args, train=True, benchmark=False):

        self.args = args
        self.train = train

        if train:
            # print('load training Set')
            self.args.ext = 'mat'
        self.split = 'train' if train else 'test'
        self.benchmark = benchmark
        self.scale = args.scale
        self.idx_scale = 0
        
        

       
        if train:
            kernel_train = sio.loadmat('data/kernels_matrix_ms.mat')
     
            mat = h5py.File('/home/zhongzhaoyu/Desktop/disks/D/data_gen/CBSR_FT/CBSR_FT_0.mat', 'r')

            self.hr_data = mat['images']['labels'][:, :, :, :]
          
            self.num = self.hr_data.shape[0]





            print(self.num)
            self.images_hr = self._scan()
            # print(self.hr_data.shape[0])


        if self.split == 'test':
            self._set_filesystem(args.dir_data)
            self.images_lr, self.images_hr = self._scan()





    def _scan(self):
        raise NotImplementedError

    def _set_filesystem(self, dir_data):
        raise NotImplementedError

    def _name_hrbin(self):
        raise NotImplementedError

    # def _name_lrbin(self, scale):
    #     raise NotImplementedError

    def __getitem__(self, idx):
        #idx = self.hr_list[idx]
        if self.train:
            idx = idx % self.num


        # print(idx)
        if self.train:
            ## this code is not released
            hr, filename = self._load_file(idx)
            ## lr_tensor
            
            ## quality_factor_tensor
            
            ## noise_sigma_tensor
            
            ## scale_factor_tensor
            
            ## covmat_tensor
            
            

            
            return lr_tensor, quality_factor_tensor, noise_sigma_tensor, scale_factor_tensor, covmat_tensor, hr_tensor, filename
        else:
            lr, hr, filename = self._load_file(idx)
            if hr == 0:
                hr = torch.zeros(1)
                [lr] = common.np2Tensor([lr], self.args.rgb_range)
            else:
                lr, hr = common.np2Tensor([lr, hr], self.args.rgb_range)

            return lr, hr, filename


    def __len__(self):
        return len(self.images_lr)

    def _get_index(self, idx):
        return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        
        # lr = self.images_lr[self.idx_scale][idx]


        if self.args.ext == 'img' or self.benchmark:
            # hr = self.images_hr[idx]

            lr = self.images_lr[idx]
            filename = lr
            lr = imread(lr)
            hr = 0 #imread(hr)
            lr = self._get_patch_test(lr)
            # w, h = lr.shape[:2]
            lr = imresize_np(lr, self.args.scale[0])
            lr = np.array(np.round(np.clip(lr, 0, 255)), np.uint8)
            # hr = hr[:w*2, :h*2, :]


        elif self.args.ext.find('sep') >= 0:
            hr = self.images_hr[idx]
            filename = hr
            # lr = np.load(lr)
            hr = np.load(hr)
        elif self.args.ext == 'mat' or self.train:
            hr = self.hr_data[idx, :, :, :]
            
            filename = str(100000+idx) + '.png'
            return hr
        else:
            filename = str(idx + 1)

        filename = os.path.splitext(os.path.split(filename)[-1])[0]

        return lr, hr, filename

    def _get_patch(self, hr, filter, filename, scale_factor, quality_factor, sigma0, sigma1, blur_flag):
        patch_size = self.args.patch_size
        # scale = self.scale[self.idx_scale]
        # multi_scale = len(self.scale) > 1
        if self.train:
            sigma, lr, hr = common.get_patch(
                hr, patch_size, filename, filter, scale_factor, quality_factor, sigma0, sigma1, blur_flag
            )
            # sigma, lr, hr = common.augment([sigma, lr, hr])
            return sigma, lr, hr
            # lr = common.add_noise(lr, self.args.noise)


    def _get_patch_test(self, hr):
        ih, iw = hr.shape[0:2]
        ih = ih // 8 * 8
        iw = iw // 8 * 8
        hr = hr[0:ih, 0:iw, :]
        return hr




    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale

