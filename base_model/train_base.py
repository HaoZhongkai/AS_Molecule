import argparse
import torch
import torch.nn as nn
from base_model.schmodel import SchNetModel
from torch.utils.data import DataLoader
from utils.funcs import *
from torchnet import meter
from tensorboardX import SummaryWriter
import time
import pickle
import os
import time
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
            res = model(g).squeeze()
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
            res = model(g).squeeze()
            loss = loss_fn(res, label)
            mae = MAE_fn(res, label)
            mae_meter.add(mae.detach().item())
            mse_meter.add(loss.detach().item())

        return mse_meter.value()[0], mae_meter.value()[0]

if __name__ == "__main__":
    config = Global_Config()
    args = make_args()

    if args.use_default is False:
        args.epochs = 60
        args.batchsize = 64
        args.use_tb = False
        args.dataset = 'qm9'
        args.device = 0
        args.save_model = True
        args.workers = 0
        args.shuffle = True
        args.multi_gpu = True
        args.prop_name = 'homo'
    print(args)


    # train_paths = ['/media/sdc/seg3d/anaconda3/pkgs/libsnn/datasets/OPV/OPV_dataset_train_2.pkl']
                  # '/media/sdc/seg3d/anaconda3/pkgs/libsnn/datasets/OPV/OPV_dataset_train_2.pkl',
                  # '/media/sdc/seg3d/anaconda3/pkgs/libsnn/datasets/OPV/OPV_dataset_train_3.pkl']
    # test_path = '/media/sdc/seg3d/anaconda3/pkgs/libsnn/datasets/OPV/OPV_dataset_test.pkl'
    # valid_path = '/media/sdc/seg3d/anaconda3/pkgs/libsnn/datasets/OPV/OPV_dataset_valid.pkl'
    # train_path = config.PATH+'/datasets/OPV/data_elem_train.pkl'
    # test_path =  config.PATH+'/datasets/OPV/data_elem_test.pkl'
    # valid_path = config.PATH+'/datasets/OPV/data_elem_valid.pkl'


    logs_path = config.PATH+'/datasets/logs'+ time.strftime('/%m%d_%H_%M')


    train_set, test_set = MoleDataset(prop_name=args.prop_name), MoleDataset(prop_name=args.prop_name)


    train_set.load_mol(config.test_pkl[args.dataset]), test_set.load_mol(config.test_pkl[args.dataset])

    device = torch.device('cuda:'+str(args.device) if torch.cuda.is_available() else 'cpu')
    # th.set_default_tensor_type(device)
    if args.use_tb:
        writer = SummaryWriter(log_dir=logs_path,comment='baseline_sch')
    else:
        writer = None
    model = SchNetModel(dim=48,n_conv=4,cutoff=5.0,width=0.5,norm=True, output_dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    if args.multi_gpu:
        model = DataParallel(model,device_ids=[i for i in range(torch.cuda.device_count())])

    info = train(args,train_set,test_set,model,optimizer, writer, device)

    if args.save_model:
        torch.save({'model_state_dict':model.state_dict(),
                    'optimizier_state_dict':optimizer.state_dict(),
                    'info':info,
                    },config.save_model_path(args.dataset))

