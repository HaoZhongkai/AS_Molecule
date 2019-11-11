import torch.nn as nn
import torch
from geo_al.k_center import K_center
from config import *
from utils.funcs import *
from torch.utils.data import DataLoader
import torch.utils as utils
import time
import random
from torchnet import meter
from tensorboardX import SummaryWriter

class CNN_Cifar(nn.Module):
    def __init__(self):
        super(CNN_Cifar,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=30,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                30,40,5,1,2
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                40,80,4,2,1
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.linear = nn.Linear(80*2*2,100)
        self.out = nn.Linear(100,10)

    def forward(self,x):
        x = x.view(-1,3,32,32)
        x = self.conv1(x)               #10*16*16
        x = self.conv2(x)               #20*8*8
        x = self.conv3(x)               # 2*2
        x = x.view(x.size(0),-1)
        x = self.linear(x)              #100
        x = self.out(x)                 #10
        return x





# K_center AL
#load the whole dataset


# slightly different from qm9/opv
def get_preds(args,model,dataset,device):
    time0 = time.time()
    dataloader = DataLoader(dataset=dataset, batch_size=args.batchsize,shuffle=False, num_workers=args.workers)
    model.to(device)
    # model.set_mean_std(dataset.mean,dataset.std)  #ignore when using cifar10
    embeddings = []
    with torch.no_grad():
        for idx,(data,_) in enumerate(dataloader):
            data = data.to(device)
            embedding = model(data)
            embeddings.append(embedding)

    embeddings = torch.cat(embeddings,dim=0)
    print('inference {}'.format(time.time()-time0))
    return embeddings

#difference:
# mae--> acc
#mols-->data or img
# model has no attributes set mean std
#loss Cross entropy
def finetune(args,train_dataset,model,optimizer,writer,info,device,iter):
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batchsize,shuffle=args.shuffle, num_workers=args.workers)#collate no need
    print('start finetuning with label numbers {}'.format(len(train_dataset)))
    # print('dataset mean {} std {}'.format(train_dataset.mean.item(), train_dataset.std.item()))
    # if model.name in ["MGCN", "SchNet"]:
    # model.set_mean_std(train_dataset.mean, train_dataset.std)
    model.to(device)
    # optimizer = optimizer_(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()  # loss_fn = nn.MSELoss()    # MAE_fn = nn.L1Loss()
    mse_meter = meter.AverageValueMeter()# mae_meter = meter.AverageValueMeter()
    acc_meter = meter.ConfusionMeter(10)
    total_epochs = iter*args.k_center_ft_epochs
    for epoch in range(args.k_center_ft_epochs):
        mse_meter.reset()
        acc_meter.reset()
        model.train()
        for idx, (datas, label) in enumerate(train_loader):
            datas = datas.to(device)
            label = label.to(device)
            scores = model(datas).squeeze()   # use CNN_Cifar
            out_classes = torch.argmax(scores, 1)
            target_digit = torch.argmax(label, 1)
            loss = loss_fn(scores, target_digit)    # mae = MAE_fn(res, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc_meter.add(out_classes, target_digit)
            mse_meter.add(loss.detach().item())
        acc = 100*sum(acc_meter.value()[i,i] for i in range(10))/acc_meter.value().sum()# mae_meter.add(mae.detach().item())
        print("Epoch {:2d}/{:2d}, training: loss: {:.7f}, acc: {:.7f}".format(epoch,total_epochs+epoch, mse_meter.value()[0],acc))
        info['train_loss'].append(mse_meter.value()[0])
        info['train_acc'].append(acc)
        info['total_epochs'].append(total_epochs+epoch)
        if args.use_tb:
            writer.add_scalar('train_loss', mse_meter.value()[0],total_epochs+epoch)
            writer.add_scalar('train_acc', acc,total_epochs+epoch)
        # if args.use_tb:
        #     writer.add_scalar('testing_loss',loss_test,epoch)
        #     writer.add_scalar('testing_mae',mae_test,epoch)
    return info

def test(args, test_set,model,device):
    loss_fn = nn.CrossEntropyLoss()
    # MAE_fn = nn.L1Loss()
    mse_meter = meter.AverageValueMeter()# mae_meter = meter.AverageValueMeter()
    acc_meter = meter.ConfusionMeter(10)
    model.eval()
    model.to(device)
    test_loader = DataLoader(dataset=test_set,batch_size=args.batchsize,shuffle=args.shuffle,num_workers=args.workers)
    # model.set_mean_std(test_set.mean,test_set.std)
    with torch.no_grad():
        for idx, (datas, label) in enumerate(test_loader):
            label = label.to(device)
            datas = datas.to(device)
            scores = model(datas).squeeze()
            out_classes = torch.argmax(scores, 1)
            target_digit = torch.argmax(label, 1)
            loss = loss_fn(scores, target_digit)# acc_meter.add(mae.detach().item())
            mse_meter.add(loss.detach().item())
            acc_meter.add(out_classes,target_digit)
        acc = 100*sum(acc_meter.value()[i,i] for i in range(10))/acc_meter.value().sum()
        return mse_meter.value()[0], acc

# Cifar 10 is small and allow us to retraining the whole NN
def k_center_learn(args,config,train_datas,test_datas,model,optimizer_,writer,device):
    ac_info = []
    label_rates = []
    print('start k-center active learning ')
    t_iterations = int((train_datas[0].shape[0]-args.init_data_num)/args.batch_data_num)  #discard tail data
    total_data_num = t_iterations*args.batch_data_num+args.init_data_num
    train_ids = random.sample(range(total_data_num),args.init_data_num)

    K_center_sampler = K_center(args,total_data_num,args.batch_data_num,train_ids)
    train_info = {'total_epochs':[],
            'train_loss':[],
            'train_acc':[]}
    train_dataset, test_dataset = Cifar(*train_datas), Cifar(*test_datas)
    # initialization training
    train_subdatas = train_datas[0][train_ids], train_datas[1][train_ids]
    train_subset = Cifar(*train_subdatas)
    optimizer = optimizer_(model.parameters(), lr=args.lr)
    train_info = finetune(args,train_subset,model,optimizer,writer,train_info,device,0)
    for iter in range(1,t_iterations+1):
        embeddings = get_preds(args,model,train_dataset,device)

        if args.query_method is 'k_center':
            query_ids = K_center_sampler.query(embeddings)
        else:
            query_ids = K_center_sampler.random_query()
        # train_subdatas.extend([train_dataset.mols[i] for i in query_ids])
        train_subdatas = torch.cat([train_subdatas[0],train_datas[0][query_ids]]), torch.cat([train_subdatas[1],train_datas[1][query_ids]])
        train_subset = Cifar(*train_subdatas)
        label_rate = len(train_subset)/total_data_num
        label_rates.append(label_rate)
        #finetuning
        if args.init_model:  #completely reinitialize the model
            model = CNN_Cifar()
            optimizer = optimizer_(model.parameters(), lr=args.lr)
        train_info = finetune(args,train_subset,model,optimizer,writer,train_info,device,iter)

        if iter % args.test_freq == 0:
            testing_mse, testing_acc = test(args,test_dataset,model,device)
            print('labels ratio {} number {}  test acc {}'.format(label_rate,len(train_subset), testing_acc))
            ac_info.append((train_info['train_loss'][-1],train_info['train_acc'][-1],testing_mse,testing_acc))

            if args.use_tb:
                writer.add_scalar('test_acc',testing_acc,label_rate)
            if args.save_model:
                torch.save({
                    'info_train':train_info,
                    'testing_acc':testing_acc,
                    'model':model.state_dict(),
                    'data_ids':K_center_sampler.data_ids
                },config.save_model_path(args.dataset+'k_center_ac'))

    ac_results = dict(zip(label_rates,ac_info))
    return ac_results
if __name__ == "__main__":
    config = Global_Config()
    args = make_args()

    # for cifar10 configuration 50000:10000
    if args.use_default is False:
        args.batchsize = 50
        args.epochs = 300
        args.use_tb = False
        args.dataset = 'cifar10'
        args.device = 1
        args.save_model = False
        args.workers = 0
        args.shuffle = True
        args.multi_gpu = False
        args.prop_name = 'homo'
        args.lr = 1e-3

        args.init_data_num = 5000
        args.k_center_ft_epochs = 50
        args.batch_data_num = 100
        args.test_freq = 2
        args.init_model = True
        args.query_method = 'k_center'


    optimizer_ = torch.optim.Adam
    print(args)
    logs_path = config.PATH + '/datasets/logs' + time.strftime('/%m%d_%H_%M')
    result_path_k_center = config.PATH + '/datasets/k_center/' + args.dataset + time.strftime('_%m%d_%H_%M.txt')
    result_path_random = config.PATH + '/datasets/rd' + args.dataset + time.strftime('_%m%d_%H_%M.txt')
    # train_set, test_set = MoleDataset(prop_name=args.prop_name), MoleDataset(prop_name=args.prop_name)
    # train_set.load_mol(config.train_pkl[args.dataset]), test_set.load_mol(config.test_pkl[args.dataset])
    train_imgs, test_imgs = pickle.load(open(config.train_pkl['cifar10'],'rb')), pickle.load(open(config.test_pkl['cifar10'],'rb'))

    device = torch.device('cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu')
    # th.set_default_tensor_type(device)
    if args.use_tb:
        writer = SummaryWriter(log_dir=logs_path, comment='cifar10')
    else:
        writer = None
    # model = SchEmbedding(dim=32, n_conv=4, cutoff=5.0, width=0.5, norm=True, output_dim=1)
    model = CNN_Cifar()
    print(model)
    # optimizer = optimizer_(model.parameters(),lr=args.lr)

    # label_rates = np.arange(75, 105, 5) / 100  # the first will not be trained
    print('start k center active learning')
    results_k_center = k_center_learn(args, config, train_imgs, test_imgs, model, optimizer_, writer, device)   #notice the optimizer_
    print('start')
    results_random = k_center_learn(args, config, train_imgs, test_imgs, model, optimizer_, writer, device)

    with open(result_path_k_center, 'w') as fp:
        for key in results_k_center.keys():
            fp.write(str(key) + '\t' + ''.join([str(i) + '\t' for i in results_k_center[key]]) + '\n')
    with open(result_path_random,'w') as fp:
        for key in results_random.keys():
            fp.write(str(key) + '\t' + ''.join([str(i) + '\t' for i in results_random[key]]) + '\n')
    print('test success')


