import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer

print(torch.__version__)

#torch.set_num_threads(6)
# torch.set_num_threads(args.n_threads*args.n_GPUs)
# torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

if checkpoint.ok:
    #aise ValueError('Why why why'
    model_t = args.model
    args.model = 'NL_EST'
    model_ne = model.Model(args, checkpoint)
    # aise ValueError('Why why why')
    args.model = 'KERNEL_EST'
    model_ke = model.Model(args, checkpoint)
    args.model  = model_t
    # args.resume = -1
    # args.pre_train = 'experiment/CBSR_0.5_N2_0/model/'
    model_sr = model.Model(args, checkpoint)



    #print('Why') 
    loader = data.Data(args)
    loss = loss.Loss(args, checkpoint) if not args.test_only else None
    t = Trainer(args, loader, model_sr, model_ne, model_ke, loss, checkpoint)
    while not t.terminate():
        t.train()
        t.test()
        # t.test()

    checkpoint.done()

