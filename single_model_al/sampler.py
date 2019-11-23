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
    def __init__(self,args,total_data_num,batch_data_num,init_ids,method='random'):
        self.args = args
        self.total_data_num = total_data_num
        self.batch_data_num = batch_data_num
        self.data_mix = args.data_mix
        self.data_mixing_rate = args.data_mixing_rate
        self.label_ids = init_ids
        self.data_ids = np.delete(np.arange(self.total_data_num,dtype=int),init_ids)  #data unselected
        self.al_method = method

        if method == 'k_center':
            self.core_ids = init_ids



    def get_label_ids(self):
        return self.label_ids


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

        else:
            raise ValueError

        # add the new batch ids to label_ids
        self.label_ids.extend(new_batch_ids)

        return new_batch_ids


    def generate_subset(self,new_batch_ids):
        if self.data_mix:
            subset_ids = deepcopy(random.sample(self.label_ids,int(self.data_mixing_rate*len(self.label_ids))))
            subset_ids.extend(list(new_batch_ids))
        else:
            subset_ids = deepcopy(self.label_ids)
            subset_ids.extend(list(new_batch_ids))
        return subset_ids


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
            print(id)
        self.core_ids = np.sort(np.concatenate([self.core_ids, new_batch_ids]))
        self.data_ids = np.delete(self.data_ids, new_batch_ids_)
        print('query new data {}'.format(time.time() - time0))
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






def check_point_test(settings,train_dataset,test_dataset,device):

    dim, cutoff, output_dim, width, n_conv, norm, atom_ref, pre_train = settings['dim'], settings['cutoff'], settings['output_dim'], settings['width'], settings['n_conv'], settings['norm'], settings['atom_ref'], settings['pre_train']
    lr, epochs, batch_size = settings['lr'], settings['epochs'], settings['batch_size']
    model = SchNetModel(dim=dim, cutoff=cutoff, output_dim=output_dim,width= width, n_conv=n_conv, norm=norm, atom_ref=atom_ref, pre_train=pre_train)
    optimizer = Adam(model.parameters(),lr=lr)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, collate_fn=batcher,
                              shuffle=True, num_workers=0)
    train_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, collate_fn=batcher,
                              shuffle=True, num_workers=0)
    print('Start checkpoint testing ')
    print('dataset mean {} std {}'.format(train_dataset.mean.item(), train_dataset.std.item()))
    model.set_mean_std(train_dataset.mean, train_dataset.std)
    model.to(device)

    # optimizer = optimizer_(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    MAE_fn = nn.L1Loss()
    mse_meter = meter.AverageValueMeter()
    mae_meter = meter.AverageValueMeter()
    test_mae_meter = meter.AverageValueMeter()
    train_mae = []
    test_mae = []
    for epoch in range(epochs):
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
        print("Epoch {:2d}/{:2d}, training: loss: {:.7f}, mae: {:.7f}".format(epoch, epochs,mse_meter.value()[0],mae_meter.value()[0]))
        train_mae.append(mae_meter.value()[0])
        if (epoch+1)% 10 ==0 :
            with torch.no_grad():
                test_mae_meter.reset()
                for idx, (mols, label) in enumerate(train_loader):
                    g = dgl.batch([mol.ful_g for mol in mols])
                    g.to(device)
                    label = label.to(device)
                    res = model(g).squeeze()  # use SchEmbedding model
                    mae = MAE_fn(res, label)
                    test_mae_meter.add(mae.detach().item())
            print('checkpoint test mae {}'.format(test_mae_meter.value()[0]))
            test_mae.append(test_mae_meter.value()[0])
    return train_mae, test_mae





def save_cpt_xlsx(cpk_path,cpk_datas,train_mae, test_mae):
    train_mae = {cpk_datas[i]:train_mae[i] for i in range(len(train_mae))}
    test_mae = {cpk_datas[i]:test_mae[i] for i in range(len(test_mae))}
    df1 = pd.DataFrame(train_mae)
    df2 = pd.DataFrame(test_mae)

    writer = pd.ExcelWriter(cpk_path,engine='xlsxwriter')

    df1.to_excel(writer,sheet_name='train_mae')
    df2.to_excel(writer,sheet_name='test_mae')

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




