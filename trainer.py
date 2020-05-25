import os
import math
from decimal import Decimal

import utility
import random
import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
import scipy.io as sio
from data import common
import numpy as np
from model.common import cov2pca, matrix_init

# from scipy.misc import imresize
# import model

class Trainer():
    def __init__(self, args, loader, my_model, model_NLEst, model_KMEst, my_loss, ckp):
        # freeze_support()
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.model_NLEst = model_NLEst
        self.model_KMEst= model_KMEst

        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)
        self.loss_NLEst = my_loss
        # args.lr = args.lr
        self.optimizer_NLEst = utility.make_optimizer(args, self.model_NLEst)
        self.scheduler_NLEst = utility.make_scheduler(args, self.optimizer_NLEst)
        self.loss_KMEst = my_loss
        self.optimizer_KMEst = utility.make_optimizer(args, self.model_KMEst)
        self.scheduler_KMEst = utility.make_scheduler(args, self.optimizer_KMEst)

        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step()

        self.error_last = 1e2



    def train(self):

        epoch = self.scheduler.last_epoch + 1
        self.optimizer.step()
        self.optimizer_NLEst.step()
        self.optimizer_KMEst.step()
        self.scheduler.step()
        self.scheduler_NLEst.step()
        self.scheduler_KMEst.step()
        self.loss_NLEst.step()
        self.loss_KMEst.step()
        self.loss.step()


        matrix = matrix_init()
        V_pca_ = sio.loadmat('data/V.mat')
        V_pca_ = V_pca_['V_pca']
        V_pca = torch.from_numpy(V_pca_).float().cuda()
        V_pca = V_pca.contiguous().view(15, 225, 1, 1)

        lr = self.scheduler.get_last_lr()[0]

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()


        self.loader_train.dataset.num = self.loader_train.dataset.hr_data.shape[0]

        print("Data pairs: {}".format(self.loader_train.dataset.num))

        #
        # self.model_NLEst.train()
        # self.model_KMEst.train()


        timer_data, timer_model = utility.timer(), utility.timer()
        for batch, (lr, quality_factor, sigma, scale_factor, covmat, hr, _)\
                in enumerate(self.loader_train):
            lr, quality_factor, sigma, scale_factor, covmat, hr = \
                self.prepare([lr, quality_factor, sigma, scale_factor, covmat, hr])
            # print(scale_factor[0,0,0,0])
            timer_data.hold()
            timer_model.tic()
            _, _, hei, wid = hr.data.size()
            _, _, hei_l, wid_l = lr.data.size()

            

            sigma_est = self.model_NLEst(lr, torch.cat((scale_factor, quality_factor), 1), 0)
            #
            loss_nlest = self.loss_NLEst(sigma_est, sigma)
            #
            self.optimizer_NLEst.zero_grad()
            loss_nlest.backward()
            self.optimizer_NLEst.step()
            #
            sigma_est = F.interpolate(sigma_est.detach(), [hei_l, wid_l], mode='bicubic')
            ker_est = self.model_KMEst(lr, torch.cat((scale_factor, quality_factor, sigma_est.detach()), 1), 0)
            #
            #
            loss_mest = self.loss_KMEst(ker_est, covmat)
            self.optimizer_KMEst.zero_grad()
            loss_mest.backward()
            self.optimizer_KMEst.step()

            ker_est = ker_est * (F.interpolate(scale_factor, [hei_l, wid_l], mode='bicubic') ** 2)  * 255.0
      
            ker_est = cov2pca(matrix.cuda(), V_pca, ker_est)



            idx_scale = 0
            sigma_est = F.interpolate(sigma_est, [hei, wid], mode='bicubic')
            quality_factor = F.interpolate(quality_factor, [hei, wid], mode='bicubic')
            scale_factor = F.interpolate(scale_factor, [hei, wid], mode='bicubic')
            ker_est = F.interpolate(ker_est, [hei, wid], mode='bicubic')

            deg_map = torch.cat((quality_factor.detach(), sigma_est.detach(), scale_factor.detach(), ker_est.detach()), 1)

            self.optimizer.zero_grad()
            sr = self.model(lr, deg_map, idx_scale)
            loss = self.loss(sr, hr)
            if loss.item() < self.args.skip_threshold * self.error_last:
                loss.backward()
                self.optimizer.step()


            else:
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, loss.item()
                ))
                break

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

    def test(self):
        epoch = self.scheduler.last_epoch + 1
        self.ckp.write_log('\nEvaluation:')

        scale_list = self.args.scale
        self.ckp.add_log(torch.zeros(1, len(scale_list)))
        self.model.eval()
        no_eval = 0
        self.model_NLEst.eval()
        self.model_KMEst.eval()
        matrix = matrix_init()
        V_pca_ = sio.loadmat('data/V.mat')
        V_pca_ = V_pca_['V_pca']
        V_pca = torch.from_numpy(V_pca_).float().cuda()
        # V_pca = V_pca.t()
        V_pca = V_pca.contiguous().view(15, 225, 1, 1)




        timer_test = utility.timer()
        with torch.no_grad():
            best_psnr = 0
            for idx_scale, scale in enumerate(scale_list):

                eval_acc = 0
                self.loader_test.dataset.set_scale(idx_scale)


                tqdm_test = tqdm(self.loader_test, ncols=120)
                for idx_img, (lr_, hr_, filename) in enumerate(tqdm_test): 
                    filename = filename[0]

                    quality_factor = 90
                    no_eval = (hr_.nelement() == 1)
                    if no_eval:
                        [lr_] = self.prepare([lr_])
                    else:
                        lr_, hr_ = self.prepare([lr_, hr_])

                    _, _, hei, wid = lr_.data.size()

                    hei, wid = lr_.shape[2:]
                    
                    hei, wid = hei // 4, wid //4

                    quality_factor = (105.0 - quality_factor) / 255.0*torch.ones([1, 1, hei, wid]).float().cuda()
                    sf = scale / 255.0



                    scale_factor = torch.ones(1, 1, hei, wid).float().cuda() * sf

                    lr_ = F.interpolate(lr_, [hei*2, wid*2], mode='bicubic')
                    sigma_est = self.model_NLEst(lr_, torch.cat((scale_factor, quality_factor), 1), 0)



                    ker_est = self.model_KMEst(lr_,
                                               torch.cat((scale_factor, quality_factor,
                                                          F.interpolate(sigma_est.detach(), [hei, wid],
                                                                        mode='bicubic')), 1), 0)

                    ker_est = ker_est * ( scale ** 2)  * 255.0


                    ker_est = cov2pca(matrix.cuda(), V_pca, ker_est)

                    sigma = F.interpolate(sigma_est, [hei, wid], mode='bicubic')
                    quality_factor = F.interpolate(quality_factor, [hei, wid], mode='bicubic')
                    scale_factor = F.interpolate(scale_factor, [hei, wid], mode='bicubic')
                    ker_est = F.interpolate(ker_est, [hei, wid], mode='bicubic')


                    deg_map = torch.cat(
                        (quality_factor.detach(), sigma.detach(), scale_factor.detach(), ker_est.detach()), 1)





                    sr = self.model(lr_, deg_map, idx_scale)

                    sr = utility.quantize(sr, self.args.rgb_range)

                    save_list = [sr]
                    if no_eval:
                        eval_acc += 0
                    else:
                        eval_acc += utility.calc_psnr(
                            sr, hr_, scale, self.args.rgb_range,
                            benchmark=self.loader_test.dataset.benchmark
                        )



                    if self.args.save_results:
                        self.ckp.save_results(filename, save_list, idx_img, scale)


                self.ckp.log[-1, idx_scale] = eval_acc / len(self.loader_test)
                if best_psnr < self.ckp.log[-1, idx_scale]:
                    is_best = True
                    best_psnr = self.ckp.log[-1, idx_scale]
                else:
                    is_best = False

                best = self.ckp.log.max(0)

                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        self.args.data_test,
                        scale,
                        self.ckp.log[-1, idx_scale],
                        best[0][idx_scale],
                        best[1][idx_scale] + 1
                    )
                )

            mean_st = self.ckp.log.mean(1)
            best_mean = mean_st.max(0)
            # print(best_mean)
            # print(best_mean[1][0] + 1 == epoch)

        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best)
            # self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))



    def prepare(self, l, volatile=False):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)
           
        return [_prepare(_l) for _l in l]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs

