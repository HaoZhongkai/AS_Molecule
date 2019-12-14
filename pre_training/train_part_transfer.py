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
import pickle
import torch.nn.functional as F
import ot

sys.path.append('..')
from utils.funcs import *
from base_model.schmodel import SchNetModel
from config import *
import numpy as np
from pre_training.sch_embeddings import SchEmbedding
from pre_training.wsch import WSchnet_N, WSchnet_G, WSchnet, WSchnet_R, MM_WSchnet_R

'''
Train from a transfer model
'''


def train(args,settings,train_datset, test_dataset, model,optimizer, writer,device):
    print("start training with {} data".format(len(train_datset)))

    train_loader = DataLoader(dataset=train_datset,batch_size=args.batchsize,collate_fn=batcher,shuffle=args.shuffle,num_workers=args.workers)


    # prepare labels
    p_labels = train_datset.prop

    # p_labels = (train_datset.prop - train_datset.mean) / train_datset.std
    # p_labels = (1 + torch.erf(p_labels / 2 ** 0.5)) / 2  # transform it to (0,1), constant must be bigger than 1e-7
    # bin_gap = 1 / settings['prop_bins']
    # p_labels = (p_labels / (bin_gap+1e-7)).long()
    p_labels = p_labels.to(device)

    print(model)
    model.set_mean_std(train_datset.mean,train_datset.std)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    loss_r = nn.MSELoss()
    loss_mae = nn.L1Loss()
    # MAE_fn = nn.L1Loss()
    n_loss_meter = meter.AverageValueMeter()
    p_loss_meter = meter.AverageValueMeter()
    n_acc_meter = meter.ConfusionMeter(100)     # clustering num might be too big, do not use confusion matrix
    p_mae_meter = meter.AverageValueMeter()
    init_lr = args.lr


    regressor = nn.Linear(64, 1,bias=True)
    regressor.to(device)


    for epoch in range(args.epochs):
        n_loss_meter.reset()
        p_loss_meter.reset()
        n_acc_meter.reset()
        p_mae_meter.reset()

        model.train()



        for idx, (mols, labels)  in enumerate(train_loader):
            g = dgl.batch([mol.ful_g for mol in mols])
            g.to(device)
            labels = labels.to(device)

            # Mask node features
            mask = torch.randint(0,g.number_of_nodes(),[int(args.mask_n_ratio*g.number_of_nodes())])
            g.ndata['nodes'][mask] = 0


            _,_, prop_preds = model(g)



            # n_pred_cls = torch.argmax(atom_preds, dim=1)
            # c_pred_cls = torch.argmax(cls_preds, dim=1)
            # cls_logits = torch.log(F.softmax(cls_preds,dim=1))


            # n_loss = loss_fn(atom_preds[mask], n_label[mask])

            # compute c loss
            # c_loss = torch.sum( - cls_labels*cls_logits,dim=1).mean()


            p_loss = loss_r(prop_preds.squeeze(),labels)       # squeeze is really important
            p_mae = loss_mae(prop_preds.squeeze(),labels)

            loss = p_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # cls_log_prob[idx*args.batchsize:idx*args.batchsize+len(mols)] = - cls_logits.detach().cpu().numpy()

            # n_loss_meter.add(loss.detach().item())  # total loss
            # c_loss_meter.add(c_loss.detach().item())
            # n_acc_meter.add(n_pred_cls, n_label)
            # c_acc_meter.add(c_pred_cls, cls_labels)
            p_loss_meter.add(p_loss.detach().item())
            p_mae_meter.add(p_mae.detach().item())


            if idx%50 == 0 and args.use_tb:
                acc = 100*sum(n_acc_meter.value()[i,i] for i in range(10))/n_acc_meter.value().sum()
                writer.add_scalar('n_train_loss',n_loss_meter.value()[0],int((idx+1+epoch*len(train_loader))/50))
                writer.add_scalar('n_train_acc',acc,int((idx+1+epoch*len(train_loader))/50))
                print('training loss {} acc {}'.format(n_loss_meter.value()[0], acc))

        # n_loss_test, n_acc_test= test(args,test_loader,model,device)

        n_acc = 100 * sum(n_acc_meter.value()[i, i] for i in range(100)) / n_acc_meter.value().sum()
        test_loss, test_mae = test(args,test_dataset,model,device)
        # p_acc = 100 * sum(p_acc_meter.value()[i, i] for i in range(settings['prop_bins'])) / p_acc_meter.value().sum()
        # print("Epoch {:2d}, training: loss: {:.7f}, acc: {:.7f}  self-clustering: loss: {:.7f} acc: {:.7f}  props: loss {} mae {}".format(epoch, n_loss_meter.value()[0], n_acc, c_loss_meter.value()[0], 100 * c_acc_meter.value(), p_loss_meter.value()[0],p_mae_meter.value()[0]))
        print("Epoch {:2d}, training: loss: {:.7f}, acc: {:.7f}  props loss {}  mae {}".format(epoch, n_loss_meter.value()[0], n_acc, p_loss_meter.value()[0], p_mae_meter.value()[0]))
        print("Test loss {} mae {}".format(test_loss, test_mae))

        if (epoch + 1) % 100 == 0:
            init_lr = init_lr *0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = init_lr
            print('current learning rate: {}'.format(init_lr))
        #
        # info['n_loss'].append(n_loss_meter.value()[0])
        # info['n_acc'].append(n_acc)
        # # info['c_loss'].append(c_loss_meter.value()[0])
        # # info['c_acc'].append(100 * c_acc_meter.value())
        # info['p_loss'].append(p_loss_meter.value()[0])
        # info['p_mae'].append(p_mae_meter.value()[0])
    return


def test(args, test_dataset,model,device):
    test_loader= DataLoader(dataset=test_dataset,batch_size=args.batchsize,collate_fn=batcher,shuffle=args.shuffle,num_workers=args.workers)
    # labels = test_dataset.prop
    loss_fn = nn.MSELoss()
    MAE_fn = nn.L1Loss()
    mse_meter = meter.AverageValueMeter()
    mae_meter = meter.AverageValueMeter()
    model.eval()
    with torch.no_grad():

        for idx, (mols, labels) in enumerate(test_loader):
            g = dgl.batch([mol.ful_g for mol in mols])
            g.to(device)
            labels = labels.to(device)

            _,_,res = model(g)
            res = res.squeeze()

            loss = loss_fn(res, labels)
            mae = MAE_fn(res, labels)
            mae_meter.add(mae.detach().item())
            mse_meter.add(loss.detach().item())

        return mse_meter.value()[0], mae_meter.value()[0]

def get_preds(args,model,dataset,device):
    time0 = time.time()
    dataloader = DataLoader(dataset=dataset, batch_size=args.batchsize*5, collate_fn=batcher,shuffle=False, num_workers=args.workers)
    model.to(device)
    # model.set_mean_std(dataset.mean,dataset.std)
    embeddings = []
    with torch.no_grad():
        for idx,(mols,_) in enumerate(dataloader):
            g = dgl.batch([mol.ful_g for mol in mols])
            g.to(device)
            embedding = model.embed_g(g)
            embeddings.append(embedding)

    embeddings = torch.cat(embeddings,dim=0)
    print('inference {}'.format(time.time()-time0))

    return embeddings


if __name__ == "__main__":
    config = Global_Config()
    args = make_args()

    if args.use_default is False:
        args.epochs = 2000
        args.batchsize = 32
        args.use_tb = False
        args.dataset = 'qm9'
        args.device = 0
        args.save_model = True
        args.workers = 0
        args.shuffle = True
        args.multi_gpu = False
        args.prop_name = 'homo'
        args.mask_n_ratio = 0.2
        args.lr = 1e-3
    print(args)

    embed_method = 'sch_embedding'

    settings = {
        'cls_num':5000,
        'cls_epochs':3,     # clustering every 5epochs
        'iters': 10,
        'init_method':'random',
        'prop_bins':25
    }

    train_data_num = 5000
    # load_path = '/home/jeffzhu/AL/datasets/models/wsch_g_gauss_1205_19_43.pkl'
    load_path = '/home/jeffzhu/AL/datasets/models/wsch_g_ae_ot_1208_09_56.pkl'

    logs_path = config.PATH+'/datasets/logs'+ time.strftime('/%m%d_%H_%M')
    model_save_path = config.PATH+'/datasets/models'+ time.strftime('/wsch_g_%m%d_%H_%M.pkl')
    print('model save path {} load path {}'.format(model_save_path, load_path))
    train_mols, test_mols = pickle.load(open(config.train_pkl['qm9'],'rb')), pickle.load(open(config.test_pkl['qm9'],'rb'))

    train_dataset, test_dataset = MoleDataset(mols=train_mols), MoleDataset(mols=test_mols)

    # train_part

    if embed_method is 'sch_embedding':
        embedding_model = SchEmbedding(dim=96, n_conv=4, cutoff=30.0, width=0.1, norm=True, output_dim=1)    #64 dim
        mols_embeddings = get_preds(args,embedding_model,train_dataset,torch.device(args.device))

    elif embed_method is 'wsch_g':
        # embedding_model = WSchnet_N(dim=96, n_conv=4, cutoff=30.0, width=0.1, norm=True, output_dim=1)
        # embedding_model.load_state_dict(torch.load(pre_model_path))
        embedding_model = pickle.load(open(load_path, 'rb'))

        mols_embeddings = get_preds(args,embedding_model,train_dataset,torch.device(args.device))

    else:
        raise ValueError
    # mols_embeddings = mols_embeddings.cpu()
    # data_ids = k_medoid(mols_embeddings, train_data_num, 10, show_stats=True)
    # data_ids = k_center(mols_embeddings,train_data_num)
    # print('data selection finished')

    # train_subset = MoleDataset(mols=[train_dataset.mols[i] for i in data_ids])
    #
    #
    train_subset = MoleDataset(mols=random.sample(train_dataset.mols, train_data_num))

    device = torch.device('cuda:'+str(args.device) if torch.cuda.is_available() else 'cpu')
    # th.set_default_tensor_type(device)
    if args.use_tb:
        writer = SummaryWriter(log_dir=logs_path,comment='baseline_sch')
    else:
        writer = None


    # model = WSchnet_R(dim=128,n_conv=4,cutoff=30.0,cls_dim=settings['cls_num'],width=0.1,norm=True, output_dim=1,props_bins=settings['prop_bins'])
    # model = MM_WSchnet_R(dim=128,n_conv=4,cutoff=30.0,cls_dim=settings['cls_num'],width=0.1,norm=True, output_dim=1,mask_rate=0.2,props_bins=settings['prop_bins'])
    model = pickle.load(open(load_path,'rb'))


    # fix part of the model, first try the embedding layer
    for param in model.embedding_layer.parameters():
        param.requires_grad = False
    # for param in model.conv_layers[:1].parameters():
    #     param.requires_grad = False
    optimizer = torch.optim.Adam(filter(lambda p:p.requires_grad,model.parameters()), lr=args.lr)

    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.multi_gpu:
        model = DataParallel(model,device_ids=[i for i in range(torch.cuda.device_count())])

    info = train(args,settings,train_subset,test_dataset,model,optimizer, writer, device)


    pickle.dump(model,open(model_save_path,'wb'))
    if args.save_model:
        torch.save({'model_state_dict':model.state_dict(),
                    'optimizier_state_dict':optimizer.state_dict(),
                    'info':info,
                    },config.save_model_path(args.dataset))