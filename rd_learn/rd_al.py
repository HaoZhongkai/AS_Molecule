from utils.funcs import *
import numpy as np
import random
import torch.nn as nn
from torchnet import meter
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from config import *
from base_model.sch import SchNetModel
from copy import deepcopy
def random_data_sampler(MaDataset,label_rate):
    rd_index = random.sample(range(len(MaDataset)),int(label_rate*len(MaDataset)))
    subdataset = None
    if MaDataset.mols:
        new_mols = [MaDataset.mols[i] for i in rd_index]
        subdataset = MoleDataset(mols=new_mols)
    elif MaDataset.datas:
        new_datas = [MaDataset.datas[i] for i in rd_index]
        subdataset = MoleDataset(datas=new_datas)
        subdataset.build()
    else:
        assert 'Not initialized dataset'

    return subdataset


def test(args, test_set,model,device):
    loss_fn = nn.MSELoss()
    MAE_fn = nn.L1Loss()
    mse_meter = meter.AverageValueMeter()
    mae_meter = meter.AverageValueMeter()
    model.eval()
    model.to(device)
    test_loader = DataLoader(dataset=test_set,batch_size=args.batchsize,collate_fn=batcher,shuffle=args.shuffle,num_workers=args.workers)

    model.set_mean_std(test_set.mean,test_set.std)
    with torch.no_grad():

        for idx, (mols, label) in enumerate(test_loader):

            label = label.to(device)

            res = model(mols).squeeze()
            loss = loss_fn(res, label)
            mae = MAE_fn(res, label)
            mae_meter.add(mae.detach().item())
            mse_meter.add(loss.detach().item())

        return mse_meter.value()[0], mae_meter.value()[0]

def train(args,train_dataset, model,optimizer_, writer,device=torch.device('cpu')):
    train_loader = DataLoader(dataset=train_dataset,batch_size=args.batchsize,collate_fn=batcher,shuffle=args.shuffle,num_workers=args.workers)
    print('start training with label numbers {}'.format(len(train_dataset)))
    print('mean {} std {}'.format(train_dataset.mean.item(), train_dataset.std.item()))
    # if model.name in ["MGCN", "SchNet"]:
    model.set_mean_std(train_dataset.mean, train_dataset.std)
    model.to(device)
    optimizer = optimizer_(model.parameters(),lr=args.lr)
    loss_fn = nn.MSELoss()
    MAE_fn = nn.L1Loss()
    mse_meter = meter.AverageValueMeter()
    mae_meter = meter.AverageValueMeter()
    info = {'train_loss':[],
            'train_mae':[]}
    for epoch in range(args.epochs):
        mse_meter.reset()
        mae_meter.reset()
        model.train()
        for idx, (mols, label)  in enumerate(train_loader):
            label = label.to(device)
            res = model(mols).squeeze()
            loss = loss_fn(res, label)
            mae = MAE_fn(res, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mae_meter.add(mae.detach().item())
            mse_meter.add(loss.detach().item())
        print("Epoch {:2d}, training: loss: {:.7f}, mae: {:.7f}".format(epoch, mse_meter.value()[0], mae_meter.value()[0]))
        info['train_loss'].append(mse_meter.value()[0])
        info['train_mae'].append(mae_meter.value()[0])
        # if args.use_tb:
        #     writer.add_scalar('testing_loss',loss_test,epoch)
        #     writer.add_scalar('testing_mae',mae_test,epoch)
    return info

def rd_active_learning(args,config,train_set,test_set,model,optimizer_,label_rates,writer,device):

    # label_rates = np.arange(0,100,step=5)/100

    ac_info = []     #[tuple(train mse, train mae, test mse, test mae)]

    #zero
    model_learner = deepcopy(model)
    test_mse, test_mae = test(args,test_set,model_learner,device)
    ac_info.append((0.0,0.0,test_mse,test_mae))
    print('test with no training {}'.format(test_mae))

    for i in range(1,len(label_rates)):
        model_learner = deepcopy(model)
        train_subset = random_data_sampler(train_set,label_rates[i])
        train_info = train(args,train_subset,model_learner,optimizer_,writer,device)
        test_mse, test_mae = test(args,test_set,model_learner,device)
        ac_info.append((train_info['train_loss'][-1],train_info['train_mae'][-1],test_mse,test_mae))
        print('labels number {}  test mae {}'.format(len(train_subset), test_mae))
        if args.use_tb:
            writer.add_scalar('test_mae',test_mae,i)
        if args.save_model:
            torch.save({
                'info_train':train_info,
                'test_mae':test_mae,
                'model':model_learner.state_dict()
            },config.save_model_path(args.dataset+'rd_ac'))

    ac_result = dict(zip(label_rates,ac_info))
    return ac_result

if __name__ == '__main__':
    config = Global_Config()
    args = make_args()

    if args.use_default is False:
        args.batchsize = 64
        args.epochs = 300
        args.use_tb = True
        args.dataset = 'qm9'
        args.device = 1
        args.save_model = True
        args.workers = 0
        args.shuffle = True
        args.multi_gpu = False
        args.prop_name = 'homo'
        args.lr = 1e-3
    print(args)

    logs_path = config.PATH + '/datasets/logs' + time.strftime('/%m%d_%H_%M')
    result_path = config.PATH + '/datasets/rd/'+args.dataset+time.strftime('_%m%d_%H_%M.txt')
    train_set, test_set = MoleDataset(prop_name=args.prop_name), MoleDataset(prop_name=args.prop_name)

    train_set.load_mol(config.train_pkl[args.dataset]), test_set.load_mol(config.test_pkl[args.dataset])

    device = torch.device('cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu')
    # th.set_default_tensor_type(device)
    if args.use_tb:
        writer = SummaryWriter(log_dir=logs_path, comment='baseline_sch')
    else:
        writer = None
    model = SchNetModel(dim=32, n_conv=4,cutoff=5.0, width=0.5, norm=True, output_dim=1, device=device)
    print(model)
    optimizer = torch.optim.Adam

    label_rates = np.arange(75,105,5)/100   #the first will not be trained
    results = rd_active_learning(args,config,train_set,test_set,model,optimizer,label_rates,writer,device)

    with open(result_path,'w') as fp:
        for key in results.keys():
            fp.write(str(key)+'\t'+ ''.join([str(i)+'\t' for i in results[key]])+'\n')

    print('test success')














