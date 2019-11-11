import torch
import torch.nn as nn
from geo_al.embedding_model import SchEmbedding
from geo_al.k_center import K_center
from config import *
from utils.funcs import *
from torch.utils.data import DataLoader
import time
import random
from torchnet import meter
from tensorboardX import SummaryWriter

# K_center AL
#load the whole dataset

def get_preds(args,model,dataset,device):
    time0 = time.time()
    dataloader = DataLoader(dataset=dataset, batch_size=args.batchsize, collate_fn=batcher,shuffle=False, num_workers=args.workers)
    model.to(device)
    model.set_mean_std(dataset.mean,dataset.std)
    embeddings = []
    with torch.no_grad():
        for idx,(mols,_) in enumerate(dataloader):
            _,embedding = model(mols,device)
            embeddings.append(embedding)

    embeddings = torch.cat(embeddings,dim=0)
    print('inference {}'.format(time.time()-time0))

    return embeddings


def finetune(args,train_dataset,model,optimizer,writer,info,device,iter):
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batchsize, collate_fn=batcher,shuffle=args.shuffle, num_workers=args.workers)
    print('start finetuning with label numbers {}'.format(len(train_dataset)))
    print('dataset mean {} std {}'.format(train_dataset.mean.item(), train_dataset.std.item()))
    # if model.name in ["MGCN", "SchNet"]:
    model.set_mean_std(train_dataset.mean, train_dataset.std)
    model.to(device)
    # optimizer = optimizer_(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    MAE_fn = nn.L1Loss()
    mse_meter = meter.AverageValueMeter()
    mae_meter = meter.AverageValueMeter()
    total_epochs = iter*args.k_center_ft_epochs
    # info = {'train_loss': [],
    #         'train_mae': [],
    #         'total_epochs':[]}
    for epoch in range(args.k_center_ft_epochs):
        mse_meter.reset()
        mae_meter.reset()
        model.train()
        for idx, (mols, label) in enumerate(train_loader):
            label = label.to(device)
            res = model(mols,device)[0].squeeze()   # use SchEmbedding model
            loss = loss_fn(res, label)
            mae = MAE_fn(res, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mae_meter.add(mae.detach().item())
            mse_meter.add(loss.detach().item())
        print("Epoch {:2d}/{:2d}, training: loss: {:.7f}, mae: {:.7f}".format(epoch,total_epochs+epoch, mse_meter.value()[0],mae_meter.value()[0]))

        info['train_loss'].append(mse_meter.value()[0])
        info['train_mae'].append(mae_meter.value()[0])
        info['total_epochs'].append(total_epochs+epoch)
        if args.use_tb:
            writer.add_scalar('train_loss', mse_meter.value()[0],total_epochs+epoch)
            writer.add_scalar('train_mae', mae_meter.value()[0],total_epochs+epoch)


        # if args.use_tb:
        #     writer.add_scalar('testing_loss',loss_test,epoch)
        #     writer.add_scalar('testing_mae',mae_test,epoch)
    return info

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

            res = model(mols,device)[0].squeeze()
            loss = loss_fn(res, label)
            mae = MAE_fn(res, label)
            mae_meter.add(mae.detach().item())
            mse_meter.add(loss.detach().item())

        return mse_meter.value()[0], mae_meter.value()[0]


def k_center_learn(args,config,train_dataset,test_dataset,model,optimizer,writer,device,test_freq):
    ac_info = []
    label_rates = []
    print('start k-center active learning ')
    t_iterations = int((len(train_dataset)-args.init_data_num)/args.batch_data_num)  #discard tail data
    total_data_num = t_iterations*args.batch_data_num+args.init_data_num
    train_ids = random.sample(range(total_data_num),args.init_data_num)

    K_center_sampler = K_center(args,total_data_num,args.batch_data_num,train_ids)
    train_info = {'total_epochs':[],
            'train_loss':[],
            'train_mae':[]}

    # initialization training
    train_mols = [train_dataset.mols[i]  for i in train_ids]
    train_subset = MoleDataset(mols=train_mols)
    train_info = finetune(args,train_subset,model,optimizer,writer,train_info,device,0)
    for iter in range(1,t_iterations+1):
        embeddings = get_preds(args,model,train_dataset,device)
        query_ids = K_center_sampler.query(embeddings)
        train_mols.extend([train_dataset.mols[i] for i in query_ids])
        train_subset = MoleDataset(mols=train_mols)
        label_rate = len(train_subset)/total_data_num
        label_rates.append(label_rate)
        #finetuning
        train_info = finetune(args,train_subset,model,optimizer,writer,train_info,device,iter)

        if iter % args.test_freq == 0:
            testing_mse, testing_mae = test(args,test_dataset,model,device)
            print('labels ratio {} number {}  test mae {}'.format(label_rate,len(train_subset), testing_mae))
            ac_info.append((train_info['train_loss'][-1],train_info['train_mae'][-1],testing_mse,testing_mae))

            if args.use_tb:
                writer.add_scalar('test_mae',testing_mae,label_rate)
            if args.save_model:
                torch.save({
                    'info_train':train_info,
                    'testing_mae':testing_mae,
                    'model':model.state_dict(),
                    'data_ids':K_center_sampler.data_ids
                },config.save_model_path(args.dataset+'k_center_ac'))

    ac_results = dict(zip(label_rates,ac_info))
    return ac_results
if __name__ == "__main__":
    config = Global_Config()
    args = make_args()

    if args.use_default is False:
        args.batchsize = 64
        args.epochs = 300
        args.use_tb = False
        args.dataset = 'qm9'
        args.device = 0
        args.save_model = True
        args.workers = 0
        args.shuffle = True
        args.multi_gpu = False
        args.prop_name = 'homo'
        args.lr = 1e-3

        args.init_data_num = 5000
        args.k_center_ft_epochs = 4
        args.batch_data_num = 100
        args.test_freq = 10


    print(args)
    logs_path = config.PATH + '/datasets/logs' + time.strftime('/%m%d_%H_%M')
    result_path = config.PATH + '/datasets/k_center/' + args.dataset + time.strftime('_%m%d_%H_%M.txt')
    train_set, test_set = MoleDataset(prop_name=args.prop_name), MoleDataset(prop_name=args.prop_name)

    train_set.load_mol(config.train_pkl[args.dataset]), test_set.load_mol(config.test_pkl[args.dataset])

    device = torch.device('cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu')
    # th.set_default_tensor_type(device)
    if args.use_tb:
        writer = SummaryWriter(log_dir=logs_path, comment='baseline_sch')
    else:
        writer = None
    model = SchEmbedding(dim=32, n_conv=4, cutoff=5.0, width=0.5, norm=True, output_dim=1)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)

    # label_rates = np.arange(75, 105, 5) / 100  # the first will not be trained
    results = k_center_learn(args, config, train_set, test_set, model, optimizer, writer, device,args.test_freq)

    with open(result_path, 'w') as fp:
        for key in results.keys():
            fp.write(str(key) + '\t' + ''.join([str(i) + '\t' for i in results[key]]) + '\n')

    print('test success')






