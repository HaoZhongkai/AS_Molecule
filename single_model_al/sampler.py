import torch
import numpy as np
import random
import torch.nn as nn
import time
from copy import deepcopy
import math
from torch.utils.data import DataLoader
from torchnet import meter
from torch.optim import Adam
import pandas as pd
import sys
import ot
import torch.nn.functional as F
import copy

sys.path.append('..')
from base_model.schmodel import SchNetModel

from utils.funcs import *


class AL_sampler(object):
    '''A unified sampler class for all single model based active learning algorithms
        __init__----------
            args:args
            total_data_num:         total_data_num of the dataset
            init_ids:               the initial training data when AL starts
            batch_data_num:         num of datas to label each iteration
            method:                 AL sampling method in 'random','k_center','bayes'
        query-------------
            inputs: the output of the inference process (torch: Tensor)
            output: the data ids of new batch data
    '''
    def __init__(self,args,total_data_num,batch_data_num,init_ids=None,method='random'):
        self.args = args
        self.total_data_num = total_data_num
        self.batch_data_num = batch_data_num
        self.data_mix = args.data_mix
        self.data_mixing_rate = args.data_mixing_rate
        self.label_ids = init_ids if init_ids != None else []   # labelled ids
        self.data_ids = np.delete(np.arange(self.total_data_num,dtype=int),init_ids)  #data unselected
        self.al_method = method

        if method == 'k_center':
            self.core_ids = init_ids



    def get_label_ids(self):
        return self.label_ids


    def get_unlabeled_ids(self):
        return self.data_ids

    # inputs: tensor of embeddings
    def query(self, inputs):
        new_batch_ids = []
        if self.al_method == 'random':
            new_batch_ids = self._random_query()

        elif self.al_method == 'k_center':
            new_batch_ids = self._k_center_query(inputs)

        elif self.al_method == 'bayes':
            new_batch_ids = self._bayes_query(inputs)

        elif self.al_method == 'msg_mask':

            new_batch_ids = self._msg_mask_query(inputs)

        elif self.al_method == 'k_medroids':

            new_batch_ids = self._k_medroids_query(inputs)

        else:
            raise ValueError

        # add the new batch ids to label_ids
        self.label_ids.extend(new_batch_ids)

        return new_batch_ids


    # def generate_subset(self,new_batch_ids):
    #     if self.data_mix:
    #         subset_ids = deepcopy(random.sample(self.label_ids,int(self.data_mixing_rate*len(self.label_ids))))
    #         subset_ids.extend(list(new_batch_ids))
    #     else:
    #         subset_ids = deepcopy(self.label_ids)
    #         subset_ids.extend(list(new_batch_ids))
    #     return subset_ids


    def _random_query(self):
        query_ids = random.sample(range(len(self.data_ids)), self.batch_data_num)
        new_batch_ids = self.data_ids[query_ids]
        self.data_ids = np.delete(self.data_ids, query_ids)
        return new_batch_ids





    # accelerated k center
    def _k_center_query(self, inputs):
        time0 = time.time()

        new_batch_ids_ = []
        new_batch_ids = []
        # calculate the minimum dist using a chunked way
        un_embeddings = inputs[self.data_ids]
        core_embeddings = inputs[self.core_ids]  # core point is the data already choosen
        min_dist = 1e5*torch.ones(self.total_data_num).to(un_embeddings.device)
        min_dist[self.core_ids] = 0
        chunk_ids = chunks(range(un_embeddings.size(0)), int(math.sqrt(un_embeddings.size(0))))
        un_ebd_a = torch.sum(un_embeddings ** 2, dim=1)
        c_ebd_b = torch.sum(core_embeddings ** 2, dim=1)
        for i in range(len(chunk_ids)):
            min_dist[self.data_ids[chunk_ids[i]]] = un_ebd_a[chunk_ids[i]] + torch.min(c_ebd_b - 2 * un_embeddings[chunk_ids[i]] @ core_embeddings.t(), dim=1)[0]
        for id in range(self.batch_data_num):
            new_point_id_ = int(torch.argmax(min_dist[self.data_ids]))  # id relative to query_data_ids
            new_point_id = self.data_ids[new_point_id_]
            new_batch_ids_.append(new_point_id_)
            new_batch_ids.append(new_point_id)
            distance_new = torch.sum((inputs[new_point_id]-inputs)**2,dim=1)
            min_dist = torch.min(torch.stack([min_dist,distance_new],dim=0),dim=0)[0]
            # print(id)
        self.core_ids = np.sort(np.concatenate([self.core_ids, new_batch_ids]))
        self.data_ids = np.delete(self.data_ids, new_batch_ids_)
        print('query new data {}'.format(time.time() - time0))
        return new_batch_ids


    def _k_medroids_query(self, inputs):
        time0 = time.time()
        un_embeddings = inputs[self.data_ids]

        new_batch_ids_ = k_medoids_pp(un_embeddings, self.batch_data_num, 10, show_stats=True)
        new_batch_ids = self.data_ids[new_batch_ids_]

        self.data_ids = np.delete(self.data_ids,new_batch_ids_)
        print('query new data {}'.format(time.time()-time0))

        return new_batch_ids




    def _variance_query(self,preds):
        time0 = time.time()
        preds_unlabeled = preds[self.data_ids]
        variance = torch.std(preds_unlabeled, dim=1).cpu().numpy().squeeze()
        vars_ids = np.stack([variance, np.arange(0, len(variance))], axis=0)
        queries = vars_ids[:, vars_ids[0].argsort()]
        query_ids = queries[1].astype(int)[-self.batch_data_num:]  # query id according to new dataset
        # query_data_ids = queries[2].astype(int)[-self.batchnum:]       #query id according to origin dataset(whole)
        new_batch_ids = self.data_ids[query_ids]
        self.data_ids = np.delete(self.data_ids, query_ids)  # del from unlabeled
        print('query new data  {}'.format(time.time() - time0))
        return new_batch_ids




    def _bayes_query(self, inputs):
        return self._variance_query(inputs)


    def _msg_mask_query(self, inputs):
        return self._variance_query(inputs)




class Inferencer(object):
    '''A unified class for inference the model before active learning sampling
        __init__------------
            method:         AL method in 'random' 'bayes' 'k_center'

        run------------
            model:          the inputs model like Schnet or MC_schnet
            dataset:     the data set( class inherited from torch: Dataset)
            device:         device

            Output: the inputs for AL sampling(eg. variance for Bayes AL method)
    '''
    def __init__(self,args,method='random'):
        self.args = args
        self.method = method


    def run(self,model,dataset,device):
        output = []
        if self.method == 'random':
            output = self._random_inference()
        elif self.method == 'k_center':
            output = self._k_center_inference(model,dataset,device)
        elif self.method == 'bayes':
            output = self._bayes_inference(model,dataset, device)
        elif self.method in ['dropout','msg_mask']:
            output = self._perbulence_inference(model,dataset,device)
        else:
            raise ValueError
        return output


    # random method does not need inference
    def _random_inference(self):
        pass


    def _k_center_inference(self, model, dataset, device):
        time0 = time.time()
        dataloader = DataLoader(dataset=dataset, batch_size=self.args.batchsize, collate_fn=batcher, shuffle=False,
                                num_workers=self.args.workers)
        model.to(device)
        model.set_mean_std(dataset.mean, dataset.std)
        scores = []
        with torch.no_grad():
            for idx, (mols, _) in enumerate(dataloader):
                g = dgl.batch([mol.ful_g for mol in mols])
                g.to(device)
                res = model.inference(g)
                scores.append(res)

        scores = torch.cat(scores, dim=0)
        print('inference {}'.format(time.time() - time0))

        return scores



    def _bayes_inference(self,model, dataset, device):
        time0 = time.time()
        dataloader = DataLoader(dataset=dataset, batch_size=self.args.batchsize*13, collate_fn=batcher, shuffle=False,
                                num_workers=self.args.workers)
        model.to(device)
        model.train()
        model.set_mean_std(dataset.mean, dataset.std)
        preds = []
        with torch.no_grad():
            for idx, (mols, _) in enumerate(dataloader):
                pred = torch.zeros(len(mols), self.args.mc_sampling_num)
                g = dgl.batch([mol.ful_g for mol in mols])
                g.to(device)
                for i in range(self.args.mc_sampling_num):
                    pred[:, i] = model(g).squeeze()
                preds.append(pred)
        preds = torch.cat(preds, dim=0)  # torch tensor
        print('inference {}'.format(time.time() - time0))
        return preds


    def _perbulence_inference(self,model, dataset, device):
        time0 = time.time()
        dataloader = DataLoader(dataset=dataset, batch_size=self.args.batchsize * 13, collate_fn=batcher, shuffle=False,
                                num_workers=self.args.workers)
        model.to(device)
        model.train()
        model.set_mean_std(dataset.mean, dataset.std)
        preds = []
        with torch.no_grad():
            for idx, (mols, _) in enumerate(dataloader):
                pred = torch.zeros(len(mols), self.args.mc_sampling_num)
                g = dgl.batch([mol.ful_g for mol in mols])
                g.to(device)
                for i in range(self.args.mc_sampling_num):
                    pred[:, i] = model.inference(g).squeeze()
                preds.append(pred)
        preds = torch.cat(preds, dim=0)  # torch tensor
        print('inference {}'.format(time.time() - time0))
        return preds




class Trainer(object):
    '''class for finetuning the neural network
        method: how the finetuning method stops
            1. fixed_epochs: finetune only fixed epochs
            2. varying_epochs: finetune epochs are decided by an inputs list
            3. by_valid: finetune stops when MAE on validation set increases
    '''
    def __init__(self,args,total_iters,method='fixed_epochs',ft_epochs=None):
        self.args = args
        self.method = method
        self.total_iters = total_iters
        self.iter = 0
        self.total_epochs = 0

        if self.method == 'fixed_epochs':
            if type(ft_epochs) is not int:
                print('fixed epochs finetuning requires a parameter ft_epochs: int')
                raise ValueError
            else:
                self.ft_epochs = [ft_epochs]* self.total_iters


        elif self.method == 'varying_epochs':
            if type(ft_epochs) is not list:
                print('varying epochs finetuning requires a parameter ft_epochs: list')
                raise ValueError
            elif len(ft_epochs) != self.total_iters:
                print('epochs list size not match')
                raise ValueError
            else:
                self.ft_epochs = ft_epochs

        elif self.method == 'by_valid':
            pass

        else:
            print('method not exists')
            raise ValueError

    # augments:
    #   info:          a dict contains the training information
    #   ft_epochs:     list of ft_epochs
    def finetune(self, inputs):
    # def finetune(self,train_dataset,model,optimizer,writer,info,device):
        train_dataset = inputs['train_dataset']
        model = inputs['model']
        optimizer = inputs['optimizer']
        writer = inputs['writer']
        info = inputs['info']
        device = inputs['device']

        if 'val_dataset' in inputs.keys():
            val_dataset = inputs['val_dataset']
        else:
            val_dataset = None

        if self.method  in ('fixed_epochs','varying_epochs'):
            info = self._ft_with_known_epochs(train_dataset,model,optimizer,writer,info,device)

        elif self.method in ('by_valid',):
            if val_dataset is None:
                raise ValueError
            info = self._ft_by_valid_datas(train_dataset,val_dataset,model,optimizer,writer,info,device)
        else:
            raise ValueError

        return info

    def test(self,test_dataset, model, device):
        loss_fn = nn.MSELoss()
        MAE_fn = nn.L1Loss()
        mse_meter = meter.AverageValueMeter()
        mae_meter = meter.AverageValueMeter()
        model.eval()
        model.to(device)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.args.batchsize, collate_fn=batcher, shuffle=self.args.shuffle,
                                 num_workers=self.args.workers)

        model.set_mean_std(test_dataset.mean, test_dataset.std)
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



    def _ft_with_known_epochs(self,train_dataset,model,optimizer,writer,info,device):
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.args.batchsize, collate_fn=batcher,
                                  shuffle=self.args.shuffle, num_workers=self.args.workers)
        print('start finetuning with label numbers {} at iteration {}'.format(len(train_dataset),self.iter))
        print('dataset mean {} std {}'.format(train_dataset.mean.item(), train_dataset.std.item()))
        model.set_mean_std(train_dataset.mean, train_dataset.std)
        model.to(device)

        # optimizer = optimizer_(model.parameters(), lr=args.lr)

        ft_epochs = self.ft_epochs[self.iter]
        self.total_epochs += ft_epochs
        self.iter += 1
        loss_fn = nn.MSELoss()
        MAE_fn = nn.L1Loss()
        mse_meter = meter.AverageValueMeter()
        mae_meter = meter.AverageValueMeter()

        for epoch in range(ft_epochs):
            mse_meter.reset()
            mae_meter.reset()
            model.train()
            for idx, (mols, label) in enumerate(train_loader):
                g = dgl.batch([mol.ful_g for mol in mols])
                g.to(device)

                label = label.to(device)
                res = model(g).squeeze()  # use SchEmbedding model
                loss = loss_fn(res, label)
                mae = MAE_fn(res, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                mae_meter.add(mae.detach().item())
                mse_meter.add(loss.detach().item())
            print("Epoch {:2d}/{:2d}, training: loss: {:.7f}, mae: {:.7f}".format(epoch, self.total_epochs,
                                                                                  mse_meter.value()[0],
                                                                                  mae_meter.value()[0]))

            info['train_loss'].append(mse_meter.value()[0])
            info['train_mae'].append(mae_meter.value()[0])
            info['total_epochs'].append(self.total_epochs)
            if self.args.use_tb:
                writer.add_scalar('train_loss', mse_meter.value()[0], self.total_epochs)
                writer.add_scalar('train_mae', mae_meter.value()[0], self.total_epochs)

        return info



    def _ft_by_valid_datas(self,train_dataset,val_dataset,model,optimizer,writer,info,device):
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.args.batchsize, collate_fn=batcher,
                                  shuffle=self.args.shuffle, num_workers=self.args.workers)
        print('start finetuning with label numbers {} at iteration {}'.format(len(train_dataset), self.iter))
        print('dataset mean {} std {}'.format(train_dataset.mean.item(), train_dataset.std.item()))
        model.set_mean_std(train_dataset.mean, train_dataset.std)
        model.to(device)
        # optimizer = optimizer_(model.parameters(), lr=args.lr)

        epoch = 0
        self.iter += 1
        loss_fn = nn.MSELoss()
        MAE_fn = nn.L1Loss()
        mse_meter = meter.AverageValueMeter()
        mae_meter = meter.AverageValueMeter()
        mae_valid = [1e10]    #inits

        while True:
            # test or valid
            _, valid_mae = self.test(val_dataset,model,device)

            if valid_mae>mae_valid[-1]:
                break
            else:
                self.total_epochs += 1
                mae_valid.append(valid_mae)
                epoch += 1


            mse_meter.reset()
            mae_meter.reset()
            model.train()
            for idx, (mols, label) in enumerate(train_loader):
                g = dgl.batch([mol.ful_g for mol in mols])
                g.to(device)
                label = label.to(device)
                res = model(g).squeeze()  # use SchEmbedding model
                loss = loss_fn(res, label)
                mae = MAE_fn(res, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                mae_meter.add(mae.detach().item())
                mse_meter.add(loss.detach().item())
            print("Epoch {:2d}/{:2d}, training: loss: {:.7f}, mae: {:.7f}  validing mae {:.7f}".format(epoch, self.total_epochs,
                                                                                  mse_meter.value()[0],
                                                                                  mae_meter.value()[0],
                                                                                  valid_mae))

            info['train_loss'].append(mse_meter.value()[0])
            info['train_mae'].append(mae_meter.value()[0])
            info['total_epochs'].append(self.total_epochs)
            if self.args.use_tb:
                writer.add_scalar('train_loss', mse_meter.value()[0], self.total_epochs)
                writer.add_scalar('train_mae', mae_meter.value()[0], self.total_epochs)

        return info


class Weakly_Supervised_Trainer(object):
    def __init__(self,args,wal_settings):
        self.args =  args
        self.wal_settings = wal_settings
        self.cls_tags = 0
        self.iters = 0
        self.method = wal_settings['cls_method']


    def run(self,model,dataset,optimizer,device,writer=None,p_labels=None,level='g'):
        if self.method == 'k_means':
            self._run_kmeans(model,dataset,optimizer,device,writer,p_labels,level)
        elif self.method == 'ot':
            self._run_ot(model,dataset,optimizer,device,writer,p_labels,level)
        else:
            raise ValueError


    def _run_kmeans(self,model,dataset,optimizer,device,writer=None,p_labels=None,level='g'):
        settings = self.wal_settings
        train_loader = DataLoader(dataset=dataset, batch_size=self.args.batchsize, collate_fn=batcher_g,shuffle=self.args.shuffle, num_workers=self.args.workers)
        model.to(device)
        if p_labels is not None:
            p_labels = p_labels.to(device)
        loss_fn = nn.CrossEntropyLoss()
        # MAE_fn = nn.L1Loss()
        n_loss_meter = meter.AverageValueMeter()
        c_loss_meter = meter.AverageValueMeter()
        p_loss_meter = meter.AverageValueMeter()
        n_acc_meter = meter.ConfusionMeter(100)  # clustering num might be too big, do not use confusion matrix
        p_acc_meter = meter.ConfusionMeter(settings['prop_bins'])
        c_acc_meter = AccMeter(settings['cls_num'])
        init_lr = self.args.lr
        info = {'n_loss': [],
                'n_acc': [],
                'c_loss': [],
                'c_acc': [],
                'p_loss':[],
                'p_acc':[]
                }
        cls_tags = 0
        for epoch in range(self.args.ft_epochs):
            n_loss_meter.reset()
            c_loss_meter.reset()
            p_loss_meter.reset()
            n_acc_meter.reset()
            c_acc_meter.reset()
            p_acc_meter.reset()
            model.train()

            # prepare pesudo labels via k means
            if epoch % settings['cls_epochs'] == 0:
                feats_all = get_preds_w(self.args, model, dataset, device)
                if self.iters == 0:
                    cls_tags = k_means(feats_all.cpu(), settings['cls_num'], settings['iters'],inits=settings['init_method'], show_stats=True)
                    self.cls_tags = cls_tags

                else:
                    cls_tags = k_means(feats_all.cpu(),settings['cls_num'],settings['iters'],inits='random',show_stats=True)   #******

                    self.cls_tags = cls_tags
                model.re_init_head()
            for idx, (mols, n_label, ids) in enumerate(train_loader):
                g = dgl.batch([mol.ful_g for mol in mols])
                g.to(device)
                n_label = n_label.to(device)

                # Mask node features
                mask = torch.randint(0, g.number_of_nodes(), [int(self.args.mask_n_ratio * g.number_of_nodes())])
                g.ndata['nodes'][mask] = 0

                # make pesudo labels vis k means
                cls_labels = cls_tags[list(ids)].to(device)

                atom_preds, cls_preds, prop_preds = model(g)

                n_pred_cls = torch.argmax(atom_preds, dim=1)
                c_pred_cls = torch.argmax(cls_preds, dim=1)
                p_pred_cls = torch.argmax(prop_preds, dim=1)

                n_loss = loss_fn(atom_preds[mask], n_label[mask])
                c_loss = loss_fn(cls_preds, cls_labels)
                p_loss = torch.Tensor([0.]).to(device)
                if level == 'w':
                    p_loss = loss_fn(prop_preds,p_labels[list(ids)])

                loss = c_loss + n_loss + p_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                n_loss_meter.add(n_loss.detach().item())
                c_loss_meter.add(c_loss.detach().item())
                n_acc_meter.add(n_pred_cls, n_label)
                c_acc_meter.add(c_pred_cls, cls_labels)
                p_loss_meter.add(p_loss.detach().item())
                p_acc_meter.add(p_pred_cls,p_labels[list(ids)]) if p_labels is not None else p_acc_meter.add(p_pred_cls,torch.zeros_like(p_pred_cls).long())



                if idx % 50 == 0 and self.args.use_tb:
                    acc = 100 * sum(n_acc_meter.value()[i, i] for i in range(10)) / n_acc_meter.value().sum()
                    writer.add_scalar('n_train_loss', n_loss_meter.value()[0],
                                      int((idx + 1 + epoch * len(train_loader)) / 50))
                    writer.add_scalar('n_train_acc', acc, int((idx + 1 + epoch * len(train_loader)) / 50))
                    print('training loss {} acc {}'.format(n_loss_meter.value()[0], acc))

            # n_loss_test, n_acc_test= test(args,test_loader,model,device)

            n_acc = 100 * sum(n_acc_meter.value()[i, i] for i in range(100)) / n_acc_meter.value().sum()
            p_acc = 100*sum(p_acc_meter.value()[i, i] for i in range(settings['prop_bins'])) / p_acc_meter.value().sum()
            print("Epoch {:2d}, training: loss: {:.7f}, acc: {:.7f}  self-clustering: loss: {:.7f} acc: {:.7f}  props: loss {} acc {} level {}".format(epoch, n_loss_meter.value()[0], n_acc, c_loss_meter.value()[0], 100 * c_acc_meter.value(), p_loss_meter.value()[0], p_acc,level))
            if (epoch + 1) % 100 == 0:
                init_lr = init_lr *0.75
                for param_group in optimizer.param_groups:
                    param_group['lr'] = init_lr
                print('current learning rate: {}'.format(init_lr))

            info['n_loss'].append(n_loss_meter.value()[0])
            info['n_acc'].append(n_acc)
            info['c_loss'].append(c_loss_meter.value()[0])
            info['c_acc'].append(100 * c_acc_meter.value())
            info['p_loss'].append(p_loss_meter.value()[0])
            info['p_acc'].append(p_acc)
            self.iters += 1
        return info


    def _run_ot(self,model,dataset,optimizer,device,writer=None,p_labels=None,level='g'):
        settings = self.wal_settings
        train_loader = DataLoader(dataset=dataset, batch_size=self.args.batchsize, collate_fn=batcher_g,
                                  shuffle=self.args.shuffle, num_workers=self.args.workers)
        model.to(device)
        if p_labels is not None:
            p_labels = p_labels.to(device)
        loss_fn = nn.CrossEntropyLoss()
        MSE_fn = nn.MSELoss()
        MAE_fn = nn.L1Loss()
        loss_meter = meter.AverageValueMeter()
        n_loss_meter = meter.AverageValueMeter()
        e_loss_meter = meter.AverageValueMeter()
        c_loss_meter = meter.AverageValueMeter()
        p_loss_meter = meter.AverageValueMeter()
        n_acc_meter = meter.ConfusionMeter(100)  # clustering num might be too big, do not use confusion matrix
        e_acc_meter = meter.ConfusionMeter(150)
        p_mae_meter = meter.AverageValueMeter()
        c_acc_meter = AccMeter(settings['cls_num'])
        init_lr = self.args.lr
        info = {'n_loss': [],
                'n_acc': [],
                'c_loss': [],
                'c_acc': [],
                'p_loss': [],
                'p_mae': []
                }
        cls_tags = 0
        edge_bins = torch.linspace(0, 30, 150).to(device)  # 0.2 per bin
        K = settings['cls_num']
        N = len(dataset)

        # q = np.ones(K)/K     # cls distribution
        # p = np.ones(N)/N     # instance distribution

        # C = np.ones([N, K]) * np.log(K) / N  # prob_tensor  (cost function)
        # Q = np.ones([N, K]) / (K * N)  # the tag is a prob distribution

        # # Now I replace it by a normal distribution 4 is decided by 100000*Gauss(4)~10
        q = np.exp(-(np.linspace(-4,4,K)**2)/2)/(np.sqrt(2*np.pi))
        q = q / q.sum()
        p = torch.ones(N) / N
        #
        C = np.ones([N, K])* np.log(K) / N   # cost matrix
        Q = np.copy(np.tile(q,(N, 1))) / N   # joint distribution

        model.set_mean_std(torch.zeros([1]), torch.ones([1]))
        for epoch in range(self.args.ft_epochs):
            loss_meter.reset()
            n_loss_meter.reset()
            e_loss_meter.reset()
            c_loss_meter.reset()
            p_loss_meter.reset()
            n_acc_meter.reset()
            e_acc_meter.reset()
            c_acc_meter.reset()
            p_mae_meter.reset()
            model.train()

            # prepare pesudo labels via optimal transport
            if epoch % settings['cls_epochs'] == 1:
                time0 = time.time()
                Q = ot.sinkhorn(p, q, C, 0.04)
                print('optimal transport finished {}'.format(time.time() -time0))
            for idx, (mols, n_label, ids) in enumerate(train_loader):
                g = dgl.batch([mol.ful_g for mol in mols])
                g.to(device)
                n_label = n_label.to(device)

                # make pesudo labels vis optimal transport
                cls_labels = torch.tensor(np.argmax(Q[list(ids)], axis=1), requires_grad=False).to(device).long()


                atom, atom_preds, edge_preds, (src, dst, edge_ids), cls_preds, embeddings_g, prop_preds = model(g)

                edge_dist = torch.clone(g.edata['distance'][edge_ids]).requires_grad_(False)
                edge_labels = torch.argmin(torch.abs(edge_dist - edge_bins), dim=1).long()
                node_labels = n_label[src]


                n_pred_cls = torch.argmax(atom_preds, dim=1)
                e_pred_cls = torch.argmax(edge_preds, dim=1)
                c_pred_cls = torch.argmax(cls_preds, dim=1)
                cls_logits = torch.log(F.softmax(cls_preds, dim=1))

                n_loss = loss_fn(atom_preds, node_labels)
                e_loss = loss_fn(edge_preds, edge_labels)
                c_loss = loss_fn(cls_preds, cls_labels)

                p_loss = torch.Tensor([0.]).to(device)
                if level == 'w':
                    p_loss = MSE_fn(prop_preds, p_labels[list(ids)])
                    p_mae = MAE_fn(prop_preds, p_labels[list(ids)])

                # loss = c_loss + n_loss + e_loss + p_loss* 5e4
                # For AB study
                loss = c_loss + p_loss* 1e4

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                C[idx * self.args.batchsize:idx * self.args.batchsize + len(mols)] = - cls_logits.detach().cpu().numpy()

                loss_meter.add(loss.detach().item())
                n_loss_meter.add(n_loss.detach().item())
                e_loss_meter.add(e_loss.detach().item())
                c_loss_meter.add(c_loss.detach().item())
                n_acc_meter.add(n_pred_cls, node_labels)
                e_acc_meter.add(e_pred_cls,edge_labels)
                c_acc_meter.add(c_pred_cls, cls_labels)
                p_loss_meter.add(p_loss.detach().item())
                p_mae_meter.add(p_mae.detach().item()) if p_labels is not None else p_mae_meter.add(0)


            # n_loss_test, n_acc_test= test(args,test_loader,model,device)
            n_acc = 100 * sum(n_acc_meter.value()[i, i] for i in range(100)) / n_acc_meter.value().sum()
            e_acc = 100 * sum(e_acc_meter.value()[i, i] for i in range(150)) / e_acc_meter.value().sum()
            print("Epoch {:2d}, training: loss: {:.7f}, node {:.4f} acc: {:.4f} edge {:.4f} acc {:.4f} clustering: loss: {:.4f} acc {:.4f} props: loss {:.5f}  mae {:.5f} level {}".format(
                epoch, loss_meter.value()[0], n_loss_meter.value()[0], n_acc, e_loss_meter.value()[0], e_acc, c_loss_meter.value()[0], c_acc_meter.value(), p_loss_meter.value()[0], p_mae_meter.value()[0], level))
            if (epoch + 1) % 100 == 0:
                init_lr = init_lr / 1
                for param_group in optimizer.param_groups:
                    param_group['lr'] = init_lr
                print('current learning rate: {}'.format(init_lr))

            info['n_loss'].append(n_loss_meter.value()[0])
            info['n_acc'].append(n_acc)
            info['c_loss'].append(c_loss_meter.value()[0])
            info['p_loss'].append(p_loss_meter.value()[0])
            info['p_mae'].append(p_mae_meter.value()[0])
            self.iters += 1
        return info



    # add ground truth label for labeled data, others by the prediction of model_h
    def generate_p_labels(self,model_h,train_dataset, label_ids,un_labeled_ids, prop_name,device):
        time0 = time.time()
        un_dataset = MoleDataset(mols=[train_dataset.mols[i] for i in un_labeled_ids], prop_name=prop_name)
        dataloader = DataLoader(dataset=un_dataset, batch_size=self.args.batchsize * 5, collate_fn=batcher, shuffle=False,num_workers=self.args.workers)
        model_h.to(device)
        # model.set_mean_std(dataset.mean,dataset.std)
        p_labels = torch.zeros(len(train_dataset)).to(device)
        scores = []
        with torch.no_grad():
            for idx, (mols, _) in enumerate(dataloader):
                g = dgl.batch([mol.ful_g for mol in mols])
                g.to(device)
                score = model_h(g).squeeze()
                scores.append(score)

        scores = torch.cat(scores, dim=0)
        p_labels[label_ids] = train_dataset.prop[label_ids].to(device)
        p_labels[un_labeled_ids] = scores
        # p_labels = (p_labels.contiguous()-model_h.mean_per_atom) / model_h.std_per_atom
        # p_labels = (1+torch.erf(p_labels/2**0.5))/2   #transform it to (0,1), when bins are big, value might overflow
        # bin_gap = 1/self.wal_settings['prop_bins']
        # p_labels = (p_labels/(bin_gap+1e-7)).long()

        print('pesudo label generation {}'.format(time.time() - time0))
        return p_labels




# replace the inference when using w-schnet
def get_preds_w(args,model,dataset,device):
    time0 = time.time()
    level = dataset.get_level()
    if level == 'n':
        batcher_ = batcher_n
    else:
        batcher_ = batcher_g
    dataloader = DataLoader(dataset=dataset, batch_size=args.batchsize*5, collate_fn=batcher_,shuffle=False, num_workers=args.workers)
    model.to(device)
    # model.set_mean_std(dataset.mean,dataset.std)
    embeddings = []
    with torch.no_grad():
        for idx,datas in enumerate(dataloader):
            mols = datas[0]
            g = dgl.batch([mol.ful_g for mol in mols])
            g.to(device)
            embedding = model.embed_g(g)
            embeddings.append(embedding)

    embeddings = torch.cat(embeddings,dim=0)
    print('inference {}'.format(time.time()-time0))

    return embeddings


def check_point_test(settings,train_dataset,test_dataset, teacher_model,max_epochs,device):

    dim, cutoff, output_dim, width, n_conv, norm, atom_ref, pre_train = settings['dim'], settings['cutoff'], settings['output_dim'], settings['width'], settings['n_conv'], settings['norm'], settings['atom_ref'], settings['pre_train']
    lr, epochs, batch_size, n_patience = settings['lr'], settings['epochs'], settings['batch_size'], settings['n_patience']
    model = SchNetModel(dim=dim, cutoff=cutoff, output_dim=output_dim,width= width, n_conv=n_conv, norm=norm, atom_ref=atom_ref, pre_train=pre_train)
    model.load_state_dict(copy.deepcopy(teacher_model.state_dict()),strict=False)
    optimizer = Adam(model.parameters(),lr=lr)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, collate_fn=batcher,
                              shuffle=True, num_workers=0)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, collate_fn=batcher,
                              shuffle=True, num_workers=0)
    print('Start checkpoint testing  label num {}'.format(len(train_dataset)))
    print('dataset mean {} std {}'.format(train_dataset.mean.item(), train_dataset.std.item()))
    model.set_mean_std(train_dataset.mean, train_dataset.std)
    model.to(device)
    init_lr = lr
    # optimizer = optimizer_(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    MAE_fn = nn.L1Loss()
    mse_meter = meter.AverageValueMeter()
    mae_meter = meter.AverageValueMeter()
    test_mae_meter = meter.AverageValueMeter()
    train_mae = []
    test_mae = []
    best_test_mae = 1e8
    patience = 0
    epoch = 0
    for i in range(max_epochs):
        mse_meter.reset()
        mae_meter.reset()
        model.train()
        for idx, (mols, label) in enumerate(train_loader):
            g = dgl.batch([mol.ful_g for mol in mols])
            g.to(device)

            label = label.to(device)
            res = model(g).squeeze()  # use SchEmbedding model
            loss = loss_fn(res, label)
            mae = MAE_fn(res, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mae_meter.add(mae.detach().item())
            mse_meter.add(loss.detach().item())
        print("Epoch {:2d}, training: loss: {:.7f}, mae: {:.7f}".format(epoch,mse_meter.value()[0],mae_meter.value()[0]))
        train_mae.append(mae_meter.value()[0])
        epoch += 1
        with torch.no_grad():
            test_mae_meter.reset()
            for idx, (mols, label) in enumerate(test_loader):
                g = dgl.batch([mol.ful_g for mol in mols])
                g.to(device)
                label = label.to(device)
                res = model(g).squeeze()  # use SchEmbedding model
                mae = MAE_fn(res, label)
                test_mae_meter.add(mae.detach().item())
        test_mae.append(test_mae_meter.value()[0])
        if (epoch + 1) % 100 == 0:
            init_lr = init_lr * 0.75
            for param_group in optimizer.param_groups:
                param_group['lr'] = init_lr
            print('current learning rate: {}'.format(init_lr))
        if test_mae[-1]>best_test_mae:
            patience += 1
        else:
            best_test_mae = test_mae[-1]
            patience = 0
        print('checkpoint test mae {} patience {}'.format(test_mae_meter.value()[0], patience))

    return model, train_mae, test_mae





def save_cpt_xlsx(cpk_path,cpk_datas,train_mae, test_mae):
    df1s = []
    df2s = []
    t_mae_min = []
    for i in range(len(train_mae)):
        df1s.append(pd.DataFrame({'data'+str(cpk_datas[i]):train_mae[i]}))
        df2s.append(pd.DataFrame({'data'+str(cpk_datas[i]):test_mae[i]}))
        t_mae_min.append(np.min(test_mae[i]))


    df1 = pd.concat(df1s,ignore_index=False,axis=1)
    df2 = pd.concat(df2s,ignore_index=False,axis=1)
    df3 = pd.DataFrame({'test_mae':t_mae_min})
    # train_mae = {'data'+str(cpk_datas[i]):train_mae[i] for i in range(len(train_mae))}
    # test_mae = {'data'+str(cpk_datas[i]):test_mae[i] for i in range(len(test_mae))}
    # df1 = pd.DataFrame(train_mae)
    # df2 = pd.DataFrame(test_mae)

    writer = pd.ExcelWriter(cpk_path,engine='xlsxwriter')

    df1.to_excel(writer,sheet_name='train_mae')
    df2.to_excel(writer,sheet_name='test_mae')
    df3.to_excel(writer,sheet_name='t_min_mae')
    writer.save()

    # def _k_center_query(self, input):
    #     time0 = time.time()
    #
    #     new_batch_ids = []
    #     # pool = mp.Pool(process_num)
    #     for id in range(self.batch_data_num):
    #         un_embeddings = input[self.data_ids]
    #         core_embeddings = input[self.core_ids]
    #         minimal_cover_dist = torch.zeros(len(self.data_ids)).to(un_embeddings.device)
    #         chunk_ids = chunks(range(un_embeddings.size(0)), int(math.sqrt(un_embeddings.size(0))))
    #         un_ebd_a = torch.sum(un_embeddings ** 2, dim=1)
    #         c_ebd_b = torch.sum(core_embeddings ** 2, dim=1)
    #         for i in range(len(chunk_ids)):
    #             # minimal_cover_dist[i] = torch.min(torch.norm(un_embeddings[i]-core_embeddings,p=2,dim=1,keepdim=False))
    #             minimal_cover_dist[chunk_ids[i]] = \
    #                 torch.min(c_ebd_b - 2 * un_embeddings[chunk_ids[i]] @ core_embeddings.t(), dim=1)[0]
    #
    #         core_point_id = torch.argmax(minimal_cover_dist + un_ebd_a).cpu().numpy()  # id in data_ids
    #         new_batch_ids.append(self.data_ids[core_point_id])
    #         self.data_ids = np.delete(self.data_ids, core_point_id)
    #         # print(id)
    #     self.core_ids = np.sort(np.concatenate([self.core_ids, new_batch_ids]))
    #     print('query new data {}'.format(time.time() - time0))
    #     return new_batch_ids




 # def _run_ot(self,model,dataset,optimizer,device,writer=None,p_labels=None,level='g'):
 #        settings = self.wal_settings
 #        train_loader = DataLoader(dataset=dataset, batch_size=self.args.batchsize, collate_fn=batcher_g,
 #                                  shuffle=self.args.shuffle, num_workers=self.args.workers)
 #        model.to(device)
 #        if p_labels is not None:
 #            p_labels = p_labels.to(device)
 #        loss_fn = nn.CrossEntropyLoss()
 #        MSE_fn = nn.MSELoss()
 #        MAE_fn = nn.L1Loss()
 #        n_loss_meter = meter.AverageValueMeter()
 #        c_loss_meter = meter.AverageValueMeter()
 #        p_loss_meter = meter.AverageValueMeter()
 #        n_acc_meter = meter.ConfusionMeter(100)  # clustering num might be too big, do not use confusion matrix
 #        p_mae_meter = meter.AverageValueMeter()
 #        c_acc_meter = AccMeter(settings['cls_num'])
 #        init_lr = self.args.lr
 #        info = {'n_loss': [],
 #                'n_acc': [],
 #                'c_loss': [],
 #                'c_acc': [],
 #                'p_loss': [],
 #                'p_mae': []
 #                }
 #        cls_tags = 0
 #        K = settings['cls_num']
 #        N = len(dataset)
 #        q = np.ones(K)/K     # cls distribution
 #        p = np.ones(N)/N     # instance distribution
 #
 #        C = np.ones([N, K]) * np.log(K) / N  # prob_tensor  (cost function)
 #        P = np.ones([N, K]) / (K * N)  # the tag is a prob distribution
 #
 #        for epoch in range(self.args.ft_epochs):
 #            n_loss_meter.reset()
 #            c_loss_meter.reset()
 #            p_loss_meter.reset()
 #            n_acc_meter.reset()
 #            c_acc_meter.reset()
 #            p_mae_meter.reset()
 #            model.train()
 #
 #            # prepare pesudo labels via optimal transport
 #            if epoch % settings['cls_epochs'] == 1:
 #                time0 = time.time()
 #                P = ot.sinkhorn(p, q, C, 0.04)
 #                print('optimal transport finished {}'.format(time.time() -time0))
 #            for idx, (mols, n_label, ids) in enumerate(train_loader):
 #                g = dgl.batch([mol.ful_g for mol in mols])
 #                g.to(device)
 #                n_label = n_label.to(device)
 #
 #                # Mask node features
 #                mask = torch.randint(0, g.number_of_nodes(), [int(self.args.mask_n_ratio * g.number_of_nodes())])
 #                g.ndata['nodes'][mask] = 0
 #
 #                # make pesudo labels vis k means
 #                cls_labels = N * torch.tensor(P[list(ids)],requires_grad=False).to(device).float()
 #
 #                atom_preds, cls_preds, prop_preds = model(g)
 #                cls_logits = torch.log(F.softmax(cls_preds, dim=1))
 #
 #                n_pred_cls = torch.argmax(atom_preds, dim=1)
 #                p_pred_cls = torch.argmax(prop_preds, dim=1)
 #
 #                n_loss = loss_fn(atom_preds[mask], n_label[mask])
 #                c_loss = torch.sum(- cls_labels * cls_logits, dim=1).mean()
 #
 #                p_loss = torch.Tensor([0.]).to(device)
 #                if level == 'w':
 #                    p_loss = loss_fn(prop_preds, p_labels[list(ids)])
 #
 #                loss = c_loss + n_loss + p_loss
 #
 #                optimizer.zero_grad()
 #                loss.backward()
 #                optimizer.step()
 #
 #                C[idx * self.args.batchsize:idx * self.args.batchsize + len(mols)] = - cls_logits.detach().cpu().numpy()
 #
 #                n_loss_meter.add(n_loss.detach().item())
 #                c_loss_meter.add(c_loss.detach().item())
 #                n_acc_meter.add(n_pred_cls, n_label)
 #                # c_acc_meter.add(c_pred_cls, cls_labels)
 #                p_loss_meter.add(p_loss.detach().item())
 #                p_acc_meter.add(p_pred_cls, p_labels[list(ids)]) if p_labels is not None else p_acc_meter.add(
 #                    p_pred_cls, torch.zeros_like(p_pred_cls).long())
 #
 #                if idx % 50 == 0 and self.args.use_tb:
 #                    acc = 100 * sum(n_acc_meter.value()[i, i] for i in range(10)) / n_acc_meter.value().sum()
 #                    writer.add_scalar('n_train_loss', n_loss_meter.value()[0],
 #                                      int((idx + 1 + epoch * len(train_loader)) / 50))
 #                    writer.add_scalar('n_train_acc', acc, int((idx + 1 + epoch * len(train_loader)) / 50))
 #                    print('training loss {} acc {}'.format(n_loss_meter.value()[0], acc))
 #
 #            # n_loss_test, n_acc_test= test(args,test_loader,model,device)
 #
 #            n_acc = 100 * sum(n_acc_meter.value()[i, i] for i in range(100)) / n_acc_meter.value().sum()
 #            p_acc = 100 * sum(
 #                p_acc_meter.value()[i, i] for i in range(settings['prop_bins'])) / p_acc_meter.value().sum()
 #            print(
 #                "Epoch {:2d}, training: loss: {:.7f}, acc: {:.7f}  self-clustering: loss: {:.7f}  props: loss {} acc {} level {}".format(
 #                    epoch, n_loss_meter.value()[0], n_acc, c_loss_meter.value()[0], p_loss_meter.value()[0], p_acc, level))
 #            if (epoch + 1) % 100 == 0:
 #                init_lr = init_lr / 1
 #                for param_group in optimizer.param_groups:
 #                    param_group['lr'] = init_lr
 #                print('current learning rate: {}'.format(init_lr))
 #
 #            info['n_loss'].append(n_loss_meter.value()[0])
 #            info['n_acc'].append(n_acc)
 #            info['c_loss'].append(c_loss_meter.value()[0])
 #            # info['c_acc'].append(100 * c_acc_meter.value())
 #            info['p_loss'].append(p_loss_meter.value()[0])
 #            info['p_acc'].append(p_acc)
 #            self.iters += 1
 #        return info