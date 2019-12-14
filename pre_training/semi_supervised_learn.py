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
from pre_training.wsch import WSchnet_N, WSchnet_G, WSchnet, WSchnet_R, Semi_Schnet

'''provide pre-trained model for transfer
    A AE like model
    first mask some nodes to reconstruct the features of the atoms
    then sample some edges to reconstruct the distance (divide to bins) to simulate a AE
    the edges number are n^2, however the degree of freedom is 3*n, we sample alpha* sqrt(n)
 '''


def train(args,settings,train_datset, test_dataset, model, data_ids,optimizer, writer,device):
    print("start")

    train_loader = DataLoader(dataset=train_datset,batch_size=args.batchsize,collate_fn=batcher_g,shuffle=args.shuffle,num_workers=args.workers)


    # prepare labels
    p_labels = train_datset.prop
    p_labels = p_labels.to(device)

    print(model)
    model.set_mean_std(train_datset.mean,train_datset.std)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    loss_r = nn.MSELoss()
    loss_mae = nn.L1Loss()
    loss_meter = meter.AverageValueMeter()
    n_loss_meter = meter.AverageValueMeter()
    c_loss_meter = meter.AverageValueMeter()
    p_loss_meter = meter.AverageValueMeter()
    n_acc_meter = meter.ConfusionMeter(100)     # clustering num might be too big, do not use confusion matrix
    c_acc_meter = AccMeter(settings['cls_num'])
    p_mae_meter = meter.AverageValueMeter()
    e_acc_meter = meter.ConfusionMeter(150)
    e_loss_meter = meter.AverageValueMeter()
    init_lr = args.lr
    info = {'n_loss':[],
            'n_acc':[],
            'c_loss':[],
            'c_acc':[],
            'p_loss': [],
            'p_mae': []
            }

    # TODO: For edge labeling (transform a edge to discrete label,  use W|h1 - h2|)
    edge_bins = torch.linspace(0,30,150).to(device)    # 0.2 per bin
    # node_classifier, edge_classifier = nn.Linear(64,100), nn.Linear(64,150)
    # node_classifier.to(device), edge_classifier.to(device)
    # 

    # optimal transport setup
    Q = 0
    K = settings['cls_num']
    N = len(train_datset)
    # uniform distribution
    q = torch.ones(K) / K
    p = torch.ones(N) / N

    C = np.ones([N, K]) * np.log(K) / N  # prob_tensor  (cost function)
    Q = np.ones([N, K]) / (K * N)  # the tag is a prob distribution
    cls_tags = torch.Tensor(np.argmax(Q,axis=1)).to(device)
    # # TODO: For Test
    # # Now I replace it by a normal distribution 4 is decided by 100000*Gauss(4)~10
    # q = np.exp(-(np.linspace(-4,4,K)**2)/2)/(np.sqrt(2*np.pi))
    # q = q / q.sum()
    # p = torch.ones(N) / N
    #
    # C = np.ones([N, K])* np.log(K) / N   # cost matrix
    # Q = np.copy(np.tile(q,(N, 1))) / N   # joint distribution




    for epoch in range(args.epochs):
        loss_meter.reset()
        n_loss_meter.reset()
        c_loss_meter.reset()
        p_loss_meter.reset()
        n_acc_meter.reset()
        c_acc_meter.reset()
        p_mae_meter.reset()
        e_acc_meter.reset()
        e_loss_meter.reset()


        model.train()

        # prepare pesudo labels via k means
        if epoch % settings['cls_epochs'] == 1 and settings['ot_loss']:

            time0 = time.time()
            Q = ot.sinkhorn(p, q, C, 0.04)  # shape dataset_num*cls_num ...takes 40s~250s on cpu
            print('optimal transport solved: {}'.format(time.time() - time0))

        if epoch % settings['pesudo_epochs'] == 0:
            p_labels = get_pesudo_labels(args,model,train_datset,data_ids,device)


        for idx, (mols, n_label, ids)  in enumerate(train_loader):
            g = dgl.batch([mol.ful_g for mol in mols])
            g.to(device)
            n_label = n_label.to(device)

            atom, atom_preds, edge_preds, (src, dst, edge_ids), cls_preds, embeddings_g, prop_preds = model(g)

            # sampling edges  Now it is confused whether to mask these edges or simply reconstruct them
            edge_dist = torch.clone(g.edata['distance'][edge_ids]).requires_grad_(False)
            edge_labels = torch.argmin(torch.abs(edge_dist-edge_bins),dim=1).long()
            node_labels = n_label[src]


            # make pesudo labels vis optimal transport
            cls_labels = torch.tensor(np.argmax(Q[list(ids)],axis=1), requires_grad=False).to(device).long()



            n_pred_cls = torch.argmax(atom_preds, dim=1)
            e_pred_cls = torch.argmax(edge_preds,dim=1)
            c_pred_cls = torch.argmax(cls_preds, dim=1)
            cls_logits = torch.log(F.softmax(cls_preds,dim=1))




            n_loss = loss_fn(atom_preds, node_labels)
            e_loss = loss_fn(edge_preds,edge_labels)
            c_loss = loss_fn(cls_preds, cls_labels)


            p_loss = loss_r(prop_preds.squeeze(), p_labels[list(ids)])      # squeeze is really important
            p_mae = loss_mae(prop_preds.squeeze(),p_labels[list(ids)])

            loss = n_loss + e_loss + 1e4 * p_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            C[idx*args.batchsize:idx*args.batchsize+len(mols)] = - cls_logits.detach().cpu().numpy()

            loss_meter.add(loss.detach().item())
            n_loss_meter.add(n_loss.detach().item())  # total loss
            c_loss_meter.add(c_loss.detach().item())
            n_acc_meter.add(n_pred_cls, node_labels)
            c_acc_meter.add(c_pred_cls, cls_labels)
            p_loss_meter.add(p_loss.detach().item())
            p_mae_meter.add(p_mae.detach().item())
            e_acc_meter.add(e_pred_cls,edge_labels)
            e_loss_meter.add(e_loss.detach().item())

            if idx%50 == 0 and args.use_tb:
                acc = 100*sum(n_acc_meter.value()[i,i] for i in range(10))/n_acc_meter.value().sum()
                writer.add_scalar('n_train_loss',n_loss_meter.value()[0],int((idx+1+epoch*len(train_loader))/50))
                writer.add_scalar('n_train_acc',acc,int((idx+1+epoch*len(train_loader))/50))
                print('training loss {} acc {}'.format(n_loss_meter.value()[0], acc))

        # n_loss_test, n_acc_test= test(args,test_loader,model,device)

        n_acc = 100 * sum(n_acc_meter.value()[i, i] for i in range(100)) / n_acc_meter.value().sum()
        e_acc = 100 * sum(e_acc_meter.value()[i, i] for i in range(150)) / e_acc_meter.value().sum()
        # c_acc = 100 * sum(c_acc_meter.value()[i, i] for i in range(150)) / c_acc_meter.value().sum()

        test_loss, test_mae = test(args,test_dataset,model,device)
        print("Epoch {:2d}, training: loss: {:.7f}, node loss {}  acc: {:.7f} edge loss {} acc {}  self-clustering: loss: {:.7f} props loss {}  mae {}".format(epoch, loss_meter.value()[0],n_loss_meter.value()[0], n_acc,e_loss_meter.value()[0] ,e_acc,c_loss_meter.value()[0],p_loss_meter.value()[0], p_mae_meter.value()[0]))
        print("Test loss {} mae {}".format(test_loss, test_mae))

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
        info['p_mae'].append(p_mae_meter.value()[0])
    return info


def test(args, test_dataset,model,device):
    test_loader= DataLoader(dataset=test_dataset,batch_size=args.batchsize,collate_fn=batcher_g,shuffle=args.shuffle,num_workers=args.workers)
    labels = test_dataset.prop
    loss_fn = nn.MSELoss()
    MAE_fn = nn.L1Loss()
    mse_meter = meter.AverageValueMeter()
    mae_meter = meter.AverageValueMeter()
    model.eval()
    with torch.no_grad():

        for idx, (mols, n_label, ids) in enumerate(test_loader):
            g = dgl.batch([mol.ful_g for mol in mols])
            g.to(device)
            label = labels[list(ids)].to(device)

            output = model(g)
            res = output[-1].squeeze()

            loss = loss_fn(res, label)
            mae = MAE_fn(res, label)
            mae_meter.add(mae.detach().item())
            mse_meter.add(loss.detach().item())

        return mse_meter.value()[0], mae_meter.value()[0]

def get_pesudo_labels(args,model,dataset,data_ids,device):
    time0 = time.time()
    dataloader = DataLoader(dataset=dataset, batch_size=args.batchsize*5, collate_fn=batcher_g,shuffle=False, num_workers=args.workers)
    model.to(device)
    model.set_mean_std(dataset.prop[data_ids].mean(),dataset.prop[data_ids].std())
    p_labels = []
    with torch.no_grad():
        for idx,(mols,_,_) in enumerate(dataloader):
            g = dgl.batch([mol.ful_g for mol in mols])
            g.to(device)
            output = model(g)
            p_labels.append(output[-1].squeeze())

    p_labels = torch.cat(p_labels,dim=0)
    p_labels[data_ids] = dataset.prop[data_ids].to(device)
    print('get pesudo labels {}'.format(time.time()-time0))

    return p_labels


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
        args.epochs = 40
        args.batchsize = 64
        args.use_tb = False
        args.dataset = 'qm9'
        args.device = 1
        args.save_model = True
        args.workers = 0
        args.shuffle = True
        args.multi_gpu = False
        args.prop_name = 'homo'
        args.mask_n_ratio = 0.2
    print(args)



    settings = {
        'cls_num':2000,
        'cls_epochs':4,     # clustering every 5epochs
        'iters': 10,
        'init_method':'random',
        'prop_bins':25,
        'edge_sampling_rate':0.2,
        'ot_loss':False,
        'pesudo_epochs':2,
    }

    train_data_num = 10000
    embed_method = 'other'


    # load_path = '/home/jeffzhu/AL/datasets/models/wsch_g_gauss_1205_19_43.pkl'
    load_path = '/home/jeffzhu/AL/datasets/models/wsch_g_ae_ot_1207_11_52.pkl'

    logs_path = config.PATH+'/datasets/logs'+ time.strftime('/%m%d_%H_%M')
    model_save_path = config.PATH+'/datasets/models'+ time.strftime('/wsch_g_ae_ot_%m%d_%H_%M.pkl')
    print('model save path {}'.format(model_save_path))
    train_mols, test_mols = pickle.load(open(config.train_pkl['qm9'],'rb')), pickle.load(open(config.test_pkl['qm9'],'rb'))

    train_dataset, test_dataset = SelfMolDataSet(mols=train_mols,level='g'), SelfMolDataSet(mols=test_mols,level='g')

    # train_part
    # train_dataset = SelfMolDataSet(mols=random.sample(train_dataset.mols, train_data_num),level='g')

    # train_part

    if embed_method is 'sch_embedding':
        embedding_model = SchEmbedding(dim=96, n_conv=4, cutoff=30.0, width=0.1, norm=True, output_dim=1)  # 64 dim
        mols_embeddings = get_preds(args, embedding_model, train_dataset, torch.device(args.device))

    elif embed_method is 'wsch_g':
        # embedding_model = WSchnet_N(dim=96, n_conv=4, cutoff=30.0, width=0.1, norm=True, output_dim=1)
        # embedding_model.load_state_dict(torch.load(pre_model_path))
        embedding_model = pickle.load(open(load_path, 'rb'))

        mols_embeddings = get_preds(args, embedding_model, train_dataset, torch.device(args.device))


    else:
        pass

    # mols_embeddings = mols_embeddings.cpu()
    # data_ids = k_medoid(mols_embeddings, train_data_num, 10, show_stats=True)
    # data_ids = k_center(mols_embeddings,train_data_num)
    # print('data selection finished')

    data_ids = random.sample(range(len(train_dataset)),train_data_num)


    device = torch.device('cuda:'+str(args.device) if torch.cuda.is_available() else 'cpu')
    # th.set_default_tensor_type(device)
    if args.use_tb:
        writer = SummaryWriter(log_dir=logs_path,comment='baseline_sch')
    else:
        writer = None


    # model = WSchnet_R(dim=128,n_conv=4,cutoff=30.0,cls_dim=settings['cls_num'],width=0.1,norm=True, output_dim=1,props_bins=settings['prop_bins'])
    model = Semi_Schnet(dim=128,n_conv=4,cutoff=30.0,cls_dim=settings['cls_num'],width=0.1,norm=True, output_dim=1,edge_bins=150,mask_n_ratio=args.mask_n_ratio,mask_msg_ratio=0,props_bins=settings['prop_bins'])

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    if args.multi_gpu:
        model = DataParallel(model,device_ids=[i for i in range(torch.cuda.device_count())])

    info = train(args,settings,train_dataset,test_dataset,model,data_ids,optimizer, writer, device)


    pickle.dump(model,open(model_save_path,'wb'))
    if args.save_model:
        torch.save({'model_state_dict':model.state_dict(),
                    'optimizier_state_dict':optimizer.state_dict(),
                    'info':info,
                    },config.save_model_path(args.dataset))