#!usr/bin/env python3
# -*- coding:utf-8 -*-

import argparse
import torch
import sys
import torch.nn as nn
from torch.utils.data import DataLoader
from torchnet import meter
from tensorboardX import SummaryWriter
import time
import random
import pickle

sys.path.append('..')
from utils.funcs import *
from base_model.schmodel import SchNetModel
from bayes_al.mc_sch import MC_SchNetModel
from bayes_al.mm_sch import MM_SchNetModel
from pre_training.wsch import WSchnet_R
from config import *

def train(args,train_dataset,test_dataset, model,optimizer, writer,device):
    print("start")

    train_loader = DataLoader(dataset=train_dataset,batch_size=args.batchsize,collate_fn=batcher,shuffle=args.shuffle,num_workers=args.workers)
    test_loader = DataLoader(dataset=test_dataset,batch_size=args.batchsize*2,collate_fn=batcher,shuffle=args.shuffle,num_workers=args.workers)
    print(model)
    print(train_dataset.mean.item(), train_dataset.std.item())
    # if model.name in ["MGCN", "SchNet"]:
    if args.multi_gpu:
        model.module.set_mean_std(train_dataset.mean, train_dataset.std)
    else:
        model.set_mean_std(train_dataset.mean,train_dataset.std)
    model.to(device)
    loss_fn = nn.MSELoss()
    MAE_fn = nn.L1Loss()
    mse_meter = meter.AverageValueMeter()
    mae_meter = meter.AverageValueMeter()
    init_lr = args.lr
    info = {'train_loss':[],
            'train_mae':[],
            'test_loss':[],
            'test_mae':[]}
    for epoch in range(args.epochs):
        mse_meter.reset()
        mae_meter.reset()
        model.train()
        for idx, (mols, label)  in enumerate(train_loader):
            g = dgl.batch([mol.ful_g for mol in mols])
            g.to(device)
            label = label.to(device)

            res = model.inference(g).squeeze()

            loss = loss_fn(res, label)
            mae = MAE_fn(res, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mae_meter.add(mae.detach().item())
            mse_meter.add(loss.detach().item())
            if idx%50 == 0 and args.use_tb:
                writer.add_scalar('training_loss',mse_meter.value()[0],int((idx+1+epoch*len(train_loader))/50))
                writer.add_scalar('training_mae',mae_meter.value()[0],int((idx+1+epoch*len(train_loader))/50))
                print('training loss {} mae {}'.format(mse_meter.value()[0], mae_meter.value()[0]))
        loss_test, mae_test = test(args,test_loader,model,device)
        print("Epoch {:2d}, training: loss: {:.7f}, mae: {:.7f} test: loss{:.7f}, mae:{:.7f}".format(epoch, mse_meter.value()[0], mae_meter.value()[0],loss_test,mae_test))
        if (epoch+1) % 100 == 0:
            init_lr = init_lr / 1.5
            for param_group in optimizer.param_groups:
                param_group['lr'] = init_lr
            print('current learning rate: {}'.format(init_lr))

        info['train_loss'].append(mse_meter.value()[0])
        info['train_mae'].append(mae_meter.value()[0])
        info['test_loss'].append(loss_test)
        info['test_mae'].append(mae_test)
        if args.use_tb:
            writer.add_scalar('testing_loss',loss_test,epoch)
            writer.add_scalar('testing_mae',mae_test,epoch)
    return info



def test(args, test_loader,model,device):
    loss_fn = nn.MSELoss()
    MAE_fn = nn.L1Loss()
    mse_meter = meter.AverageValueMeter()
    mae_meter = meter.AverageValueMeter()
    model.eval()
    with torch.no_grad():

        for idx, (mols, label) in enumerate(test_loader):
            g = dgl.batch([mol.ful_g for mol in mols])
            g.to(device)
            label = label.to(device)
            res = model.inference(g).squeeze()
            loss = loss_fn(res, label)
            mae = MAE_fn(res, label)
            mae_meter.add(mae.detach().item())
            mse_meter.add(loss.detach().item())

        return mse_meter.value()[0], mae_meter.value()[0]

if __name__ == "__main__":
    config = Global_Config()
    args = make_args()

    if args.use_default is False:
        args.epochs = 1200
        args.batchsize = 64
        args.lr = 1e-3
        args.use_tb = True
        args.dataset = 'qm9'
        args.device = 0
        args.save_model = True
        args.workers = 0
        args.shuffle = True
        args.multi_gpu = False
        args.prop_name = 'homo'
    print(args)

    train_data_num = 110000



    logs_path = config.PATH+'/datasets/logs'+ time.strftime('/%m%d_%H_%M')

    train_set, test_set = MoleDataset(prop_name=args.prop_name), MoleDataset(prop_name=args.prop_name)


    train_set.load_mol(config.train_pkl[args.dataset]), test_set.load_mol(config.test_pkl[args.dataset])


    # train_part
    train_set = MoleDataset(mols=random.sample(train_set.mols,train_data_num))


    device = torch.device('cuda:'+str(args.device) if torch.cuda.is_available() else 'cpu')
    # th.set_default_tensor_type(device)
    if args.use_tb:
        writer = SummaryWriter(log_dir=logs_path,comment='baseline_sch')
    else:
        writer = None


    # model = SchNetModel(dim=48,n_conv=4,cutoff=5.0,width=0.5,norm=True, output_dim=1)
    # model = MC_SchNetModel(dim=32,n_conv=4,cutoff=5.0,width=0.5,norm=True, output_dim=1)
    # model = SchNetModel(dim=128,n_conv=4,cutoff=30.0,width=0.1,norm=True, output_dim=1)
    model = MM_SchNetModel(dim=128,n_conv=4,cutoff=30.0,width=0.1,norm=True, output_dim=1,mask_rate=0.2)
    # model = WSchnet_R(dim=128,n_conv=4,cutoff=30.0,cls_dim=1000,width=0.1,norm=True, output_dim=1)


    # run a mini SchNet model

    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,momentum=0.5,nesterov=0.6)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(time.strftime('%m%d_%H_%M  model {}  optimizer {}'.format(str(model),str(optimizer))))

    if args.multi_gpu:
        model = DataParallel(model,device_ids=[i for i in range(torch.cuda.device_count())])

    info = train(args,train_set,test_set,model,optimizer, writer, device)

    if args.save_model:
        torch.save({'model_state_dict':model.state_dict(),
                    'optimizier_state_dict':optimizer.state_dict(),
                    'info':info,
                    },config.save_model_path(args.dataset))

