import os

from data import common

import numpy as np
import scipy.misc as misc
import scipy.io as sio

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
        kernel_train = sio.loadmat('data/kernels_matrix_ms.mat')
        # kernel_train = h5py.File('../Kernel_PCA/kernels_matrix_ms+pca.mat')
        # self.kernel_train = kernel_train['ks']['kernels'][0][0][0]
        # self.covmat_train = kernel_train['ks']['sigma_xy_ori'][0][0][0]

        # print(self.pcas_train[0])
        # mat = h5py.File('../../imdb_RGB2.mat')
        if train:
            kernel_train = h5py.File('../Kernel_PCA/kernels_matrix_ms_pca.mat')
            # print(kernel_train.keys())

            self.pcas_train = [kernel_train[element[0]][:, :] for element in kernel_train['pcas_c']]

            mat = h5py.File('/home/genk/disks/E/CBSR_Training/CBSR_Training_0.mat', 'r')
            num = 96000

            self.hr_data = mat['images']['labels'][:num, :, :, :]
            self.lr_data = mat['images']['data'][:num, :, :, :]
            self.PCs = mat['images']['PCs'][:num]
            self.Scales = mat['images']['Scales'][:num]
            self.k_pcas = mat['images']['k_pcas'][:num, :]
            self.noise_sigma = mat['images']['noise_simga'][:num, :, :, :]
            self.KSigma = mat['images']['KSigma'][:num, :3]



            # print(mat['images']['labels'].shape)
            # self.hr_data = mat['images']['labels'][:,:,:,:]
            self.num = self.hr_data.shape[0] * 4





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
            lr, hr, noise_sigma, filename = self._load_file(idx)
            quality_factor =  self.PCs[idx]
            scale_factor = self.Scales[idx]
            # pcas_coff = self.k_pcas[idx, :]
            KSigma = self.KSigma[idx, :]
            covmat_tensor = KSigma * scale_factor * scale_factor / 255.0
            lr, hr, noise_sigma = common.get_patch_three(lr, hr, noise_sigma, self.args.patch_size)


            # pcas_coff = np.squeeze(pcas_coff.astype(np.float32))
            # pcas_coff_tensor = torch.from_numpy(pcas_coff)

            # sigma, lr, hr = self._get_patch(hr, filter, filename, scale_factor, quality_factor, sigma0, sigma1, blur_falg)
            # noise_sigma, lr, hr = common.set_channel([noise_sigma, lr, hr], self.args.n_colors)
            lh, lw = lr.shape[:2]
            # hh, hw = hr.shape[:2]


            # covmat_tensor = torch.mul(covmat_tensor.view(3, 1, 1).contiguous(), torch.ones([1, lh, lw]).float())
            # pcas_coff_tensor = torch.mul(pcas_coff_tensor.view(15, 1, 1).contiguous(), torch.ones([1, lh//2, lw//2]).float())

            covmat_tensor = np.squeeze(covmat_tensor.astype(np.float32))
            covmat_tensor = torch.from_numpy(covmat_tensor)
            scale_factor_tensor = torch.ones([1, lh//4, lw//4]).float()
            scale_factor_tensor.mul_(int(scale_factor) / 255.0)
            covmat_tensor = torch.mul(covmat_tensor.view(3, 1, 1).contiguous(),
                                      torch.ones([1, lh//2, lw//2]).float())

            quality_factor_tensor = torch.ones([1, lh//4, lw//4]).float()
            quality_factor_tensor.mul_(int(110 - quality_factor) / 255.0)

            noise_sigma_tensor, lr_tensor, hr_tensor = common.np2Tensor([noise_sigma, lr, hr], self.args.rgb_range)

            # lr_tensor = torch.cat((lr_tensor, quality_factor_tensor), 0)0
            return lr_tensor, quality_factor_tensor, noise_sigma_tensor, scale_factor_tensor, covmat_tensor, hr_tensor, filename
        else:
            lr, hr, filename = self._load_file(idx)
            lr, hr = common.np2Tensor([lr, hr], self.args.rgb_range)
            return lr, hr, filename


    def __len__(self):
        return len(self.images_lr)

    def _get_index(self, idx):
        return idx % 96000

    def _load_file(self, idx):
        idx = self._get_index(idx)
        
        # lr = self.images_lr[self.idx_scale][idx]


        if self.args.ext == 'img' or self.benchmark:
            hr = self.images_hr[idx//200]

            lr = self.images_lr[idx]
            filename = lr
            lr = misc.imread(lr)
            hr = misc.imread(hr)
            lr = self._get_patch_test(lr)
            w, h = lr.shape[:2]
            lr = misc.imresize(lr, [w*2, h*2], 'bicubic')
            hr = hr[:w*2, :h*2, :]
        elif self.args.ext.find('sep') >= 0:
            hr = self.images_hr[idx]
            filename = hr
            # lr = np.load(lr)
            hr = np.load(hr)
        elif self.args.ext == 'mat' or self.train:
            hr = self.hr_data[idx, :, :, :]
            lr = self.lr_data[idx, :, :, :]
            noise_sigma = self.noise_sigma[idx, :, :, :]

            hr = np.squeeze(hr.transpose((1, 2, 0)))
            lr = np.squeeze(lr.transpose((1, 2, 0)))
            noise_sigma = np.squeeze(noise_sigma.transpose((1, 2, 0)))
            lr, hr, noise_sigma = common.augment([lr, hr, noise_sigma])
            filename = str(100000+idx) + '.png'
            return lr, hr, noise_sigma, filename
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

