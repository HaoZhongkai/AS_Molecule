import torch
from utils.funcs import *
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
from torchnet import meter
from copy import deepcopy
import torch.multiprocessing as mp
# from torch.multiprocessing import Manager,Queue,Process,Pipe,Lock
from config import *
import time
# from base_model.sch import SchNetModel
from qbc_learn.model import SchNetModel
from tensorboardX import SummaryWriter

# torch.multiprocessing.set_start_method('spawn')
torch.multiprocessing.set_sharing_strategy('file_system')
import sys
import torch
from torch.utils.data import dataloader
from torch.multiprocessing import reductions
from multiprocessing.reduction import ForkingPickler

default_collate_func = dataloader.default_collate


def default_collate_override(batch):
  dataloader._use_shared_memory = False
  return default_collate_func(batch)

setattr(dataloader, 'default_collate', default_collate_override)

for t in torch._storage_classes:
  if sys.version_info[0] == 2:
    if t in ForkingPickler.dispatch:
        del ForkingPickler.dispatch[t]
  else:
    if t in ForkingPickler._extra_reducers:
        del ForkingPickler._extra_reducers[t]

class Commitee():
    def __init__(self,args,dataset_dir,model_num, batch_data_num,dataset_num):
        self.model_num = model_num
        self.batchnum = batch_data_num  #different from batchsize
        self.data_ids = np.arange(0,dataset_num,dtype=int)  #now unlabeled data
        self.args = args
        self.dataset_dir = dataset_dir
        self.prop_name = args.prop_name

    def query_dataset(self):
        time0 = time.time()
        query_dataset = FMolDataSet(self.dataset_dir,self.prop_name,self.data_ids)
        print('get query set time {}'.format(time.time()-time0))
        return query_dataset

    # build query datasets for vars, mols: the whole training dataset
    def query_ids(self,preds):
        # selection
        time0 = time.time()
        vars = np.std(preds, axis=0).squeeze()
        vars_ids = np.stack([vars,np.arange(0,len(vars))],axis=0)
        queries = vars_ids[:,vars_ids[0].argsort()]
        query_ids = queries[1].astype(int)[-self.batchnum:]  #query id according to new dataset
        # query_data_ids = queries[2].astype(int)[-self.batchnum:]       #query id according to origin dataset(whole)
        query_data_ids = self.data_ids[query_ids]
        self.data_ids = np.delete(self.data_ids, query_ids)    #del from unlabeled
        print('query new data  {}'.format(time.time()-time0))
        return query_data_ids

#inference all data with mp models
#shares {'args':..'dataset':..,'preds':..}
#queue {'model':.,'model_id':..}
def get_preds(args,dataset,queue,rt_queue,device):
    torch.cuda.set_device(device)
    time0 = time.time()
    dataloader = DataLoader(dataset=dataset, batch_size=args.batchsize, collate_fn=batcher,shuffle=False, num_workers=args.workers)
    print('building subprocess dataset {}'.format(time.time()-time0))
    time0 = time.time()
    while not queue.empty():
        target = queue.get()
        model, model_id = target['model'], target['model_id']
        model.to(device)
        model.set_mean_std(dataset.mean,dataset.std)
        preds = []
        with torch.no_grad():
            for idx, (mols,_) in enumerate(dataloader):
                pred = model(mols,device)
                preds.append(pred.cpu().numpy())
        rt_queue[model_id] = np.concatenate(preds,axis=0)
    print('inferencing {}'.format(time.time()-time0))
    return

#finetune with multiprocessing
#iter: iteration id choosing batch; m_id: model id
#shares {'dataset':..,'args'....}
#tar_queue({ 'model':..,'info':corresponds to model_id,'model_id}) return_dict[{'model':[],'info':[]}]
def finetune(args,train_dataset,tar_queue,return_models,optimizer_,device,iter):
    # torch.cuda.set_device(device)
    while not tar_queue.empty():
        target = tar_queue.get()
        m_id = target['model_id']
        model = target['model']
        info = target['info']
        train_loader = DataLoader(dataset=train_dataset,batch_size=args.batchsize,collate_fn=batcher,shuffle=args.shuffle,num_workers=args.workers)
        total_epochs = iter*args.qbc_ft_epochs
        if m_id == 0:
            print('Finetuning with label numbers {}'.format(len(train_dataset)))
            print('Iter {} Total epochs {}'.format(iter,total_epochs) )
            print('dataset mean {} std {}'.format(train_dataset.mean.item(), train_dataset.std.item()))
        model.set_mean_std(train_dataset.mean, train_dataset.std)
        model.to(device)
        optimizer = optimizer_(model.parameters(),lr=args.lr)
        loss_fn = nn.MSELoss()
        MAE_fn = nn.L1Loss()
        mse_meter = meter.AverageValueMeter()
        mae_meter = meter.AverageValueMeter()
        for epoch in range(args.qbc_ft_epochs):
            mse_meter.reset()
            mae_meter.reset()
            model.train()
            for idx, (mols, label)  in enumerate(train_loader):
                label = label.to(device)
                res = model(mols,device).squeeze()
                loss = loss_fn(res, label)
                mae = MAE_fn(res, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                mae_meter.add(mae.detach().item())
                mse_meter.add(loss.detach().item())
            if m_id == 0:
                print("Epoch {:2d}/ {}, training: loss: {:.7f}, mae: {:.7f} cuda {}".format(epoch,epoch+total_epochs,mse_meter.value()[0],mae_meter.value()[0],device))
            info['total_epoch'].append(epoch+total_epochs)
            info['train_loss'].append(mse_meter.value()[0])
            info['train_mae'].append(mae_meter.value()[0])
        #return finetuned model
        model.to(torch.device('cpu'))
        return_models[m_id] = {'model':model,'info':info}
    return

def qbc_test(args, test_set, models, device,use_all):
    loss_fn = nn.MSELoss()
    MAE_fn = nn.L1Loss()
    test_models = models if use_all else [models[0]]
    test_loader = DataLoader(dataset=test_set,batch_size=args.batchsize,collate_fn=batcher,shuffle=args.shuffle,num_workers=args.workers)
    loss_mat = np.zeros([len(test_models),len(test_loader)])
    mae_mat = np.zeros([len(test_models),len(test_loader)])
    with torch.no_grad():
        for i in range(len(test_models)):
            test_models[i].to(device)
            test_models[i].eval()
            for idx, (mols, label) in enumerate(test_loader):
                label = label.to(device)
                res = test_models[i](mols,device).squeeze()
                loss = loss_fn(res, label)
                mae = MAE_fn(res, label)
                loss_mat[i,idx*args.batchsize:idx*args.batchsize+len(mols)] = loss.cpu().numpy()
                mae_mat[i,idx*args.batchsize:idx*args.batchsize+len(mols)] = mae.cpu().numpy()
    mse = loss_mat.mean()
    mae = mae_mat.mean()
    return mse, mae

#models:[model]
#results:{'data num':[],'mae'[]}
def qbc_active_learning(args,config,train_set,test_set,models,optimizer_,writer,process_num,test_freq):
    devices = [torch.device('cuda:'+str(i) if torch.cuda.is_available() else 'cpu') for i in range(process_num)]
    commitee = Commitee(args,train_set.dir,len(models),args.batch_data_num,len(train_set))
    model_num = len(models)
    traindata_num = len(train_set)
    t_iterations = int(len(train_set)/args.batch_data_num)  #discard tails
    train_ids = []  #currently not use inits
    manager = mp.Manager()

    return_models = manager.list([{'model':[],'info':{'total_epoch':[],'train_loss':[],'train_mae':[]}} for _ in range(model_num)])
    ac_results = {'label_rate':[],'data_num':[],'test_mae':[]}
    print('start active learning QBC')
    for iter in range(t_iterations):
        #inference
        print('getting query dataset...')
        time0 = time.time()
        query_subset = commitee.query_dataset()
        preds = manager.list([[] for _ in range(model_num)])
        print('building share objects {}  datasetid {}'.format(time.time()-time0,id(query_subset)))
        queue_q = mp.Queue()
        for i in range(model_num):
            queue_q.put({'model':models[i],'model_id':i})
        processes_q = []
        print('building inference process...{}'.format(time.time()-time0))
        for i in range(process_num):
            time0 = time.time()
            p = mp.Process(target=get_preds,args=(args,query_subset,queue_q,preds,devices[i]))
            p.start()
            processes_q.append(p)
            print('subprocess build {}'.format(time.time()-time0))
        for p in processes_q:
            p.join()
        preds = np.stack(preds,axis=0)
        #query
        print('quering new labeled data...')
        query_ids = commitee.query_ids(preds)
        train_ids.extend(query_ids)

        print(len(set(list(train_ids))))

        # print(set('training idddddd num{}'.format(len(set(train_ids)))))


        train_subset = FMolDataSet(train_set.dir,args.prop_name,train_ids)
        data_num = len(train_subset)
        label_rate = data_num / traindata_num
        #finetuning
        queue_t = mp.Queue()
        for i in range(model_num):  #put models to queue
            info = {'total_epoch':[],'train_loss':[],'train_mae':[]}
            queue_t.put({'model':models[i],'info':info,'model_id':i})
        processes_t = []
        print('building finetuning process...')
        for i in range(process_num):
            p = mp.Process(target=finetune,args=(args,train_subset,queue_t,return_models,optimizer_,devices[i],iter))
            p.start()
            processes_t.append(p)
        for p in processes_t:
            p.join()
        models = [return_models[i]['model'] for i in range(model_num)]
        print('finetuning finish with {} data, label rate {}'.format(len(train_subset),len(train_subset)/traindata_num))
        if args.use_tb:
            total_epochs = return_models[0]['info']['total_epoch'][-1]
            writer.add_scalar('training_loss', return_models[0]['info']['train_loss'][-1], total_epochs)
            writer.add_scalar('training_mae', return_models[0]['info']['train_mae'][-1], total_epochs)
        if (iter+1)%test_freq == 0: #save the model after test
            _, test_mae = qbc_test(args,test_set,models,devices[0],use_all=args.test_use_all)
            ac_results['data_num'].append(data_num)
            ac_results['label_rate'].append(label_rate)
            ac_results['test_mae'].append(test_mae)
            print('test mae {} train data num {} label rate {}'.format(test_mae,data_num,label_rate))
            if args.use_tb:
                writer.add_scalar('test_mae',test_mae,label_rate)
            if args.save_model: #currently no training info
                torch.save({'test_mae':test_mae,'models_state_dict':[model.state_dict() for model in models]},
                           config.save_model_path(args.dataset+'qbc_ac'))
    return ac_results

if __name__ =="__main__":
    config = Global_Config()
    args = make_args()

    if args.use_default is False:
        args.batchsize = 64
        args.epochs = 300
        args.use_tb = False
        args.dataset = 'qm9'
        args.device = 1
        args.save_model = True
        args.workers = 0
        args.shuffle = True
        args.multi_gpu = False
        args.prop_name = 'homo'
        args.lr = 1e-3
        args.workers = 10
        args.qbc_ft_epochs = 6


        args.batch_data_num = 200



        args.model_num = 1
        args.process_num = 1
        args.test_freq = 10

    if args.process_num>torch.cuda.device_count():
        print('process can not be more than gpu num {}'.format(torch.cuda.device_count()))
        args.process_num = torch.cuda.device_count()
    print(args)
    logs_path = config.PATH + '/datasets/logs' + time.strftime('/%m%d_%H_%M')
    result_path = config.PATH + '/datasets/qbc/' + args.dataset + time.strftime('_%m%d_%H_%M.txt')
    train_set, test_set = FMolDataSet(config.mols_dir['qm9'],prop_name=args.prop_name), MoleDataset(prop_name=args.prop_name)

    # train_set.load_mol(config.train_pkl[args.dataset]), test_set.load_mol(config.test_pkl[args.dataset])
    test_set.load_mol(config.test_pkl[args.dataset])


    print('loading data success')
    mp = mp.get_context('forkserver')
    #init models

    models = []
    for i in range(args.model_num):
        models.append(SchNetModel(dim=32, n_conv=4,cutoff=5.0, width=0.5, norm=True, output_dim=1))
        print(models[-1])

    optimizer_ = torch.optim.Adam

    if args.use_tb:
        writer = SummaryWriter(log_dir=logs_path, comment='baseline_sch')
    else:
        writer = None


    results = qbc_active_learning(args,config,train_set,test_set,models,optimizer_,writer,process_num=args.process_num,test_freq=args.test_freq)

    with open(result_path,'w') as fp:
        for i in range(len(results['data_num'])):
            fp.write(str(results['data_num'][i])+'\t'+str(results['test_mae'][i])+'\n')
        fp.write(str(args))
    print('test success')







# class Commitee():
#     def __init__(self,args,model_num, batch_data_num,dataset_num):
#         self.model_num = model_num
#         self.batchnum = batch_data_num  #different from batchsize
#         self.data_ids = np.arange(0,dataset_num,dtype=int)  #now unlabeled data
#         self.args = args
#
#     def query_dataset(self,mols):
#         time0 = time.time()
#         query_dataset = MoleDataset(mols=[mols[i] for i in self.data_ids])
#         print('get query set time {}'.format(time.time()-time0))
#         return query_dataset
#
#     # build query datasets for vars, mols: the whole training dataset
#     def query_ids(self,mols,preds):
#         # selection
#         time0 = time.time()
#         vars = np.std(preds, axis=0).squeeze()
#         vars_ids = np.stack([vars,np.arange(0,len(vars)),deepcopy(self.data_ids)],axis=0)
#
#         queries = vars_ids[:,vars_ids[0].argsort()]
#         query_ids = queries[1].astype(int)[-self.batchnum:]  #query id according to new dataset
#         query_data_ids = queries[2].astype(int)[-self.batchnum:]       #query id according to origin dataset(whole)
#         self.data_ids = np.delete(self.data_ids, query_data_ids)    #del from unlabeled
#         # print('query_id',query_ids)
#         # print('data id',self.data_ids)
#         mols_query = [mols[i] for i in query_ids] # newly added mols
#         print('query new data  {}'.format(time.time()-time0))
#         return mols_query







# #finetune with multiprocessing
# #iter: iteration id choosing batch; m_id: model id
# #shares {'dataset':..,'args'....}
# #tar_queue({ 'model':..,'info':corresponds to model_id,'model_id}) return_dict[{'model':[],'info':[]}]
# def finetune(shares,tar_queue,return_models,optimizer_,device,iter):
#     # torch.cuda.set_device(device)
#     args = shares['args']
#     train_dataset = shares['dataset']
#     while not tar_queue.empty():
#         target = tar_queue.get()
#         m_id = target['model_id']
#         model = target['model']
#         info = target['info']
#         train_loader = DataLoader(dataset=train_dataset,batch_size=args.batchsize,collate_fn=batcher,shuffle=args.shuffle,num_workers=args.workers)
#         total_epochs = iter*args.qbc_ft_epochs
#         if m_id == 0:
#             print('Finetuning with label numbers {}'.format(len(train_dataset)))
#             print('Iter {} Total epochs {}'.format(iter,total_epochs) )
#             print('dataset mean {} std {}'.format(train_dataset.mean.item(), train_dataset.std.item()))
#         model.set_mean_std(train_dataset.mean, train_dataset.std)
#         model.to(device)
#         optimizer = optimizer_(model.parameters(),lr=args.lr)
#         loss_fn = nn.MSELoss()
#         MAE_fn = nn.L1Loss()
#         mse_meter = meter.AverageValueMeter()
#         mae_meter = meter.AverageValueMeter()
#         for epoch in range(args.qbc_ft_epochs):
#             mse_meter.reset()
#             mae_meter.reset()
#             model.train()
#             for idx, (mols, label)  in enumerate(train_loader):
#                 label = label.to(device)
#                 res = model(mols,device).squeeze()
#                 loss = loss_fn(res, label)
#                 mae = MAE_fn(res, label)
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#                 mae_meter.add(mae.detach().item())
#                 mse_meter.add(loss.detach().item())
#             if m_id == 0:
#                 print("Epoch {:2d}/ {}, training: loss: {:.7f}, mae: {:.7f} cuda {}".format(epoch,epoch+total_epochs,mse_meter.value()[0],mae_meter.value()[0],device))
#             info['total_epoch'].append(epoch+total_epochs)
#             info['train_loss'].append(mse_meter.value()[0])
#             info['train_mae'].append(mae_meter.value()[0])
#         #return finetuned model
#         model.to(torch.device('cpu'))
#         return_models[m_id] = {'model':model,'info':info}
#     return








# #models:[model]
# #results:{'data num':[],'mae'[]}
# def qbc_active_learning(args,config,train_set,test_set,models,optimizer_,writer,process_num,test_freq):
#     devices = [torch.device('cuda:'+str(i) if torch.cuda.is_available() else 'cpu') for i in range(process_num)]
#
#
#     commitee = Commitee(args,train_set.dir,len(models),args.batch_data_num,len(train_set))
#     model_num = len(models)
#     traindata_num = len(train_set)
#     t_iterations = int(len(train_set)/args.batch_data_num)  #discard tails
#     train_mols = []  #currently not use inits
#     manager = mp.Manager()
#
#     return_models = manager.list([{'model':[],'info':{'total_epoch':[],'train_loss':[],'train_mae':[]}} for _ in range(model_num)])
#     ac_results = {'label_rate':[],'data_num':[],'test_mae':[]}
#     print('start active learning QBC')
#     for iter in range(t_iterations):
#         #inference
#         print('getting query dataset...')
#         time0 = time.time()
#         query_subset = commitee.query_dataset()
#         preds = manager.list([[] for _ in range(model_num)])
#         # query_loader = DataLoader(dataset=query_subset, batch_size=args.batchsize, collate_fn=batcher, shuffle=args.shuffle,num_workers=args.workers)
#
#         # sharing_vars_q = manager.dict({'dataset':query_subset,'args':args})   #59s on 10000 data
#         # sharing_vars_q = manager.list(query_subset.mols)              #30s on 10000 data
#         # sharing_vars_q = manager.dict({'args':args})
#         print('building share objects {}  datasetid {}'.format(time.time()-time0,id(query_subset)))
#         queue_q = mp.Queue()
#         for i in range(model_num):
#             queue_q.put({'model':models[i],'model_id':i})
#         processes_q = []
#         print('building inference process...{}'.format(time.time()-time0))
#         for i in range(process_num):
#             time0 = time.time()
#             p = mp.Process(target=get_preds,args=(args,query_subset,queue_q,preds,devices[i],query_subset))
#             p.start()
#             processes_q.append(p)
#             print('subprocess build {}'.format(time.time()-time0))
#         for p in processes_q:
#             p.join()
#         preds = np.stack(preds,axis=0)
#         #query
#         print('quering new labeled data...')
#         query_mols = commitee.query_ids(train_set.mols,preds)
#         train_mols.extend(query_mols)
#         train_subset = MoleDataset(mols=train_mols)
#         data_num = len(train_subset)
#         label_rate = data_num / traindata_num
#         #finetuning
#         sharing_vars_t = manager.dict({'dataset': train_subset,'args':args})
#         queue_t = mp.Queue()
#         for i in range(model_num):  #put models to queue
#             info = {'total_epoch':[],'train_loss':[],'train_mae':[]}
#             queue_t.put({'model':models[i],'info':info,'model_id':i})
#         processes_t = []
#         print('building finetuning process...')
#         for i in range(process_num):
#             p = mp.Process(target=finetune,args=(sharing_vars_t,queue_t,return_models,optimizer_,devices[i],iter))
#             p.start()
#             processes_t.append(p)
#             print('Finetuning on device id {}'.format(i))
#         for p in processes_t:
#             p.join()
#         models = [return_models[i]['model'] for i in range(model_num)]
#         print('finetuning finish with {} data, label rate {}'.format(len(train_subset),len(train_subset)/traindata_num))
#
#         if (iter+1)%test_freq == 0: #save the model after test
#             _, test_mae = qbc_test(args,test_set,models,devices[0],use_all=args.test_use_all)
#             ac_results['data_num'].append(data_num)
#             ac_results['label_rate'].append(label_rate)
#             ac_results['test_mae'].append(test_mae)
#             print('test mae {} train data num {} label rate {}'.format(test_mae,data_num,label_rate))
#             if args.use_tb:
#                 total_epochs = return_models[0]['info']['total_epoch'][-1]
#                 writer.add_scalar('test_mae',test_mae,label_rate)
#                 writer.add_scalar('training_loss',return_models[0]['info']['train_loss'][-1],total_epochs)
#                 writer.add_scalar('training_mae',return_models[0]['info']['train_mae'][-1],total_epochs)
#             if args.save_model: #currently no training info
#                 torch.save({
#                     'test_mae':test_mae,
#                     'models_state_dict':[model.state_dict() for model in models]
#                 },config.save_model_path(args.dataset+'qbc_ac'))
#     return ac_results