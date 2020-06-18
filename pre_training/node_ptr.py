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
from pre_training.wsch import WSchnet_N


def train(args, train_datset, test_dataset, model, optimizer, writer, device):
    print("start")

    train_loader = DataLoader(dataset=train_datset,
                              batch_size=args.batchsize,
                              collate_fn=batcher_n,
                              shuffle=args.shuffle,
                              num_workers=args.workers)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.batchsize,
                             collate_fn=batcher_n,
                             shuffle=args.shuffle,
                             num_workers=args.workers)

    print(model)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    # MAE_fn = nn.L1Loss()
    n_loss_meter = meter.AverageValueMeter()
    n_acc_meter = meter.ConfusionMeter(100)
    init_lr = args.lr
    info = {'n_loss': [], 'n_acc': []}
    for epoch in range(args.epochs):
        n_loss_meter.reset()
        n_acc_meter.reset()
        model.train()
        for idx, (mols, label) in enumerate(train_loader):
            g = dgl.batch([mol.ful_g for mol in mols])
            g.to(device)
            label = label.to(device)

            # Mask node features
            mask = torch.randint(
                0, g.number_of_nodes(),
                [int(args.mask_n_ratio * g.number_of_nodes())])
            g.ndata['nodes'][mask] = 0

            res = model(g).squeeze()

            n_pred_cls = torch.argmax(res, dim=1)
            n_loss = loss_fn(res[mask], label[mask])

            optimizer.zero_grad()
            n_loss.backward()
            optimizer.step()

            n_loss_meter.add(n_loss.detach().item())
            n_acc_meter.add(n_pred_cls, label)
            if idx % 50 == 0 and args.use_tb:
                acc = 100 * sum(
                    n_acc_meter.value()[i, i]
                    for i in range(10)) / n_acc_meter.value().sum()
                writer.add_scalar(
                    'n_train_loss',
                    n_loss_meter.value()[0],
                    int((idx + 1 + epoch * len(train_loader)) / 50))
                writer.add_scalar(
                    'n_train_acc', acc,
                    int((idx + 1 + epoch * len(train_loader)) / 50))
                print('training loss {} acc {}'.format(n_loss_meter.value()[0],
                                                       acc))

        n_loss_test, n_acc_test = test(args, test_loader, model, device)

        acc = 100 * sum(n_acc_meter.value()[i, i]
                        for i in range(10)) / n_acc_meter.value().sum()
        print(
            "Epoch {:2d}, training: loss: {:.7f}, acc: {:.7f}  test: loss: {:.7f} acc: {:.7f}"
            .format(epoch,
                    n_loss_meter.value()[0], acc, n_loss_test, n_acc_test))
        if (epoch + 1) % 100 == 0:
            init_lr = init_lr / 1
            for param_group in optimizer.param_groups:
                param_group['lr'] = init_lr
            print('current learning rate: {}'.format(init_lr))

        info['n_loss'].append(n_loss_meter.value()[0])
        info['n_acc'].append(acc)
    return info


def test(args, test_loader, model, device):
    model.eval()
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    # MAE_fn = nn.L1Loss()
    n_loss_meter = meter.AverageValueMeter()
    n_acc_meter = meter.ConfusionMeter(100)
    with torch.no_grad():
        for idx, (mols, label) in enumerate(test_loader):
            g = dgl.batch([mol.ful_g for mol in mols])
            g.to(device)
            label = label.to(device)

            # Mask node features
            mask = torch.randint(
                0, g.number_of_nodes(),
                [int(args.mask_n_ratio * g.number_of_nodes())])
            g.ndata['nodes'][mask] = 0

            res = model(g).squeeze()

            n_pred_cls = torch.argmax(res, dim=1)
            n_loss = loss_fn(res[mask], label[mask])

            n_loss_meter.add(n_loss.detach().item())
            n_acc_meter.add(n_pred_cls, label)

        n_loss_test = n_loss_meter.value()[0]
        n_acc_test = 100 * sum(n_acc_meter.value()[i, i]
                               for i in range(10)) / n_acc_meter.value().sum()

    return n_loss_test, n_acc_test


if __name__ == "__main__":
    config = Global_Config()
    args = make_args()

    if args.use_default is False:
        args.epochs = 20
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

    logs_path = config.PATH + '/datasets/logs' + time.strftime('/%m%d_%H_%M')
    model_save_path = config.PATH + '/datasets/models' + time.strftime(
        '/wsch_n_%m%d_%H_M.pkl')

    train_mols, test_mols = pickle.load(open(config.train_pkl['qm9'],
                                             'rb')), pickle.load(
                                                 open(config.test_pkl['qm9'],
                                                      'rb'))

    train_dataset, test_dataset = SelfMolDataSet(
        mols=train_mols, level='n'), SelfMolDataSet(mols=test_mols, level='n')

    device = torch.device(
        'cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu')
    # th.set_default_tensor_type(device)
    if args.use_tb:
        writer = SummaryWriter(log_dir=logs_path, comment='baseline_sch')
    else:
        writer = None

    model = WSchnet_N(dim=96,
                      n_conv=4,
                      cutoff=30.0,
                      width=0.1,
                      norm=True,
                      output_dim=1)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    if args.multi_gpu:
        model = DataParallel(
            model, device_ids=[i for i in range(torch.cuda.device_count())])

    info = train(args, train_dataset, test_dataset, model, optimizer, writer,
                 device)

    pickle.dump(model, open(model_save_path, 'wb'))
    if args.save_model:
        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'optimizier_state_dict': optimizer.state_dict(),
                'info': info,
            }, config.save_model_path(args.dataset))
