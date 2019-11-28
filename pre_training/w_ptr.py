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

sys.path.append('..')
from utils.funcs import *
from base_model.schmodel import SchNetModel
from config import *
import numpy as np
from pre_training.wsch import WSchnet_N, WSchnet_G, WSchnet

'''Jointly self-training with node level masking and clustering center labeling'''


def train(args,settings,train_datset, model,optimizer, writer,device):
    print("start")

    train_loader = DataLoader(dataset=train_datset,batch_size=args.batchsize,collate_fn=batcher_g,shuffle=args.shuffle,num_workers=args.workers)
    # test_loader= DataLoader(dataset=test_dataset,batch_size=args.batchsize,collate_fn=batcher_g,shuffle=args.shuffle,num_workers=args.workers)


    # prepare labels
    p_labels = (train_datset.prop - train_datset.mean) / train_datset.std
    p_labels = (1 + torch.erf(p_labels / 2 ** 0.5)) / 2  # transform it to (0,1), constant must be bigger than 1e-7
    bin_gap = 1 / settings['prop_bins']
    p_labels = (p_labels / (bin_gap+1e-7)).long()
    p_labels = p_labels.to(device)

    print(model)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    # MAE_fn = nn.L1Loss()
    n_loss_meter = meter.AverageValueMeter()
    c_loss_meter = meter.AverageValueMeter()
    p_loss_meter = meter.AverageValueMeter()
    n_acc_meter = meter.ConfusionMeter(100)     # clustering num might be too big, do not use confusion matrix
    c_acc_meter = AccMeter(settings['cls_num'])
    p_acc_meter = meter.ConfusionMeter(settings['prop_bins'])

    init_lr = args.lr
    info = {'n_loss':[],
            'n_acc':[],
            'c_loss':[],
            'c_acc':[],
            'p_loss': [],
            'p_acc': []
            }
    cls_tags = 0
    for epoch in range(args.epochs):
        n_loss_meter.reset()
        c_loss_meter.reset()
        p_loss_meter.reset()
        n_acc_meter.reset()
        c_acc_meter.reset()
        p_acc_meter.reset()

        model.train()

        # prepare pesudo labels via k means
        if epoch % settings['cls_epochs'] == 0:
            feats_all = get_preds(args,model,train_datset,device)
            if epoch == 0:
                cls_tags = k_means(feats_all.cpu(),settings['cls_num'],settings['iters'],inits=settings['init_method'],show_stats=True)
            else:
                cls_tags = k_means(feats_all.cpu(),settings['cls_num'],settings['iters'],inits=cls_tags,show_stats=True)


        for idx, (mols, n_label, ids)  in enumerate(train_loader):
            g = dgl.batch([mol.ful_g for mol in mols])
            g.to(device)
            n_label = n_label.to(device)

            # Mask node features
            mask = torch.randint(0,g.number_of_nodes(),[int(args.mask_n_ratio*g.number_of_nodes())])
            g.ndata['nodes'][mask] = 0

            # make pesudo labels vis k means
            cls_labels = cls_tags[list(ids)].to(device)

            atom_preds, cls_preds, prop_preds = model(g)


            n_pred_cls = torch.argmax(atom_preds, dim=1)
            c_pred_cls = torch.argmax(cls_preds, dim=1)
            p_pred_cls = torch.argmax(prop_preds, dim=1)

            n_loss = loss_fn(atom_preds[mask], n_label[mask])
            c_loss = loss_fn(cls_preds,cls_labels)
            p_loss = loss_fn(prop_preds, p_labels[list(ids)])


            loss = c_loss + n_loss + p_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            n_loss_meter.add(n_loss.detach().item())
            c_loss_meter.add(c_loss.detach().item())
            n_acc_meter.add(n_pred_cls, n_label)
            c_acc_meter.add(c_pred_cls, cls_labels)
            p_loss_meter.add(p_loss.detach().item())
            p_acc_meter.add(p_pred_cls, p_labels[list(ids)])

            if idx%50 == 0 and args.use_tb:
                acc = 100*sum(n_acc_meter.value()[i,i] for i in range(10))/n_acc_meter.value().sum()
                writer.add_scalar('n_train_loss',n_loss_meter.value()[0],int((idx+1+epoch*len(train_loader))/50))
                writer.add_scalar('n_train_acc',acc,int((idx+1+epoch*len(train_loader))/50))
                print('training loss {} acc {}'.format(n_loss_meter.value()[0], acc))

        # n_loss_test, n_acc_test= test(args,test_loader,model,device)

        n_acc = 100 * sum(n_acc_meter.value()[i, i] for i in range(100)) / n_acc_meter.value().sum()
        p_acc = 100 * sum(p_acc_meter.value()[i, i] for i in range(settings['prop_bins'])) / p_acc_meter.value().sum()
        print(
            "Epoch {:2d}, training: loss: {:.7f}, acc: {:.7f}  self-clustering: loss: {:.7f} acc: {:.7f}  props: loss {} acc {}".format(
                epoch, n_loss_meter.value()[0], n_acc, c_loss_meter.value()[0], 100 * c_acc_meter.value(), p_loss_meter.value()[0],p_acc))
        if (epoch + 1) % 100 == 0:
            init_lr = init_lr / 1
            for param_group in optimizer.param_groups:
                param_group['lr'] = init_lr
            print('current learning rate: {}'.format(init_lr))

        info['n_loss'].append(n_loss_meter.value()[0])
        info['n_acc'].append(n_acc)
        info['c_loss'].append(c_loss_meter.value()[0])
        info['c_acc'].append(100 * c_acc_meter.value())
        info['p_loss'].append(p_loss_meter.value()[0])
        info['p_acc'].append(p_acc)
    return info


def get_preds(args,model,dataset,device):
    time0 = time.time()
    dataloader = DataLoader(dataset=dataset, batch_size=args.batchsize*5, collate_fn=batcher_g,shuffle=False, num_workers=args.workers)
    model.to(device)
    # model.set_mean_std(dataset.mean,dataset.std)
    embeddings = []
    with torch.no_grad():
        for idx,(mols,_,_) in enumerate(dataloader):
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
        args.epochs = 200
        args.batchsize = 64
        args.use_tb = False
        args.dataset = 'qm9'
        args.device = 0
        args.save_model = True
        args.workers = 0
        args.shuffle = True
        args.multi_gpu = False
        args.prop_name = 'homo'
        args.mask_n_ratio = 0.2
    print(args)


    settings = {
        'cls_num':5000,
        'cls_epochs':1,     # clustering every 5epochs
        'iters': 10,
        'init_method':'k_center',
        'prop_bins':25
    }

    print(time.strftime('%m%d_%H_%M'))
    logs_path = config.PATH+'/datasets/logs'+ time.strftime('/%m%d_%H_%M')
    model_save_path = config.PATH+'/datasets/models'+ time.strftime('/wsch_g_%m%d_%H_%M.pkl')

    train_mols, test_mols = pickle.load(open(config.train_pkl['qm9'],'rb')), pickle.load(open(config.test_pkl['qm9'],'rb'))

    train_dataset, test_dataset = SelfMolDataSet(mols=train_mols,level='g'), SelfMolDataSet(mols=test_mols,level='g')



    device = torch.device('cuda:'+str(args.device) if torch.cuda.is_available() else 'cpu')
    # th.set_default_tensor_type(device)
    if args.use_tb:
        writer = SummaryWriter(log_dir=logs_path,comment='baseline_sch')
    else:
        writer = None


    model = WSchnet(dim=256,n_conv=4,cutoff=30.0,cls_dim=settings['cls_num'],width=0.1,norm=True, output_dim=1,props_bins=settings['prop_bins'])

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    if args.multi_gpu:
        model = DataParallel(model,device_ids=[i for i in range(torch.cuda.device_count())])

    info = train(args,settings,train_dataset,model,optimizer, writer, device)


    pickle.dump(model,open(model_save_path,'wb'))
    if args.save_model:
        torch.save({'model_state_dict':model.state_dict(),
                    'optimizier_state_dict':optimizer.state_dict(),
                    'info':info,
                    },config.save_model_path(args.dataset))