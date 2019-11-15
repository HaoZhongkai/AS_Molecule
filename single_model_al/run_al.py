import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import random
import sys
from tensorboardX import SummaryWriter
sys.path.append('..')

from bayes_al.mc_sch import MC_SchNetModel
from base_model.schmodel import SchNetModel
from single_model_al.sampler import AL_sampler, Trainer, Inferencer
from utils.funcs import MoleDataset
from config import *


def active_learning(input):
    args = input['args']
    config = input['config']
    train_dataset = input['train_dataset']
    test_dataset = input['test_dataset']
    # val_dataset = input['val_dataset']
    model = input['model']
    # optimizer = input['optimizer']
    writer = input['writer']
    device = input['device']
    # test_freq = input['test_freq']
    al_method = input['al_method']
    ft_method = input['ft_method']
    ft_epochs = input['ft_epochs']

    print('start {} active learning'.format(al_method))

    ac_info = []
    label_rates = []
    train_info = {'total_epochs': [],
                  'train_loss': [],
                  'train_mae': []}
    t_iterations = int((len(train_dataset)-args.init_data_num)/args.batch_data_num) + 1  #discard tail data
    total_data_num = (t_iterations-1)*args.batch_data_num+args.init_data_num
    train_ids = random.sample(range(total_data_num),args.init_data_num)


    al_sampler = AL_sampler(args,total_data_num,args.batch_data_num,train_ids,al_method)
    al_inferencer = Inferencer(args,al_method)
    al_trainer = Trainer(args,t_iterations,method=ft_method,ft_epochs=ft_epochs)

    # initialization training
    train_mols = [train_dataset.mols[i] for i in train_ids]
    train_subset = MoleDataset(mols=train_mols)
    input['train_dataset'] = train_subset
    input['info'] = train_info
    train_info =  al_trainer.finetune(input)

    # due to the initial iteration, the showed iteration will be 1 smaller than the actual iteration
    for iters in range(1,t_iterations):

        preds = al_inferencer.run(model,train_dataset,device)
        new_batch_ids = al_sampler.query(preds)

        train_mols.extend([train_dataset.mols[i] for i in new_batch_ids])
        train_subset = MoleDataset(mols=train_mols)
        label_rate = len(train_subset) / total_data_num
        label_rates.append(label_rate)

        #finetuning
        # renew train_dataset
        input['train_dataset'] = train_subset
        input['info'] = train_info
        train_info = al_trainer.finetune(input)

        # record results
        if iters % args.test_freq == 0:
            testing_mse, testing_mae = al_trainer.test(test_dataset, model, device)
            print('labels ratio {} number {}  test mae {}'.format(label_rate, len(train_subset), testing_mae))
            ac_info.append((train_info['train_loss'][-1], train_info['train_mae'][-1], testing_mse, testing_mae))

            if args.use_tb:
                writer.add_scalar('test_mae', testing_mae, label_rate)
            if args.save_model:
                torch.save({
                    'info_train': train_info,
                    'testing_mae': testing_mae,
                    'model': model.state_dict(),
                    'data_ids': al_sampler.data_ids
                }, config.save_model_path(args.dataset+al_method+'_'+ft_method))

    ac_results = dict(zip(label_rates, ac_info))
    return ac_results



if __name__ == "__main__":
    config = Global_Config()
    args = make_args()

    al_method = 'k_center'
    ft_method = 'by_valid'
    ft_epochs = 2


    print(args)
    logs_path = config.PATH + '/datasets/logs' + time.strftime('/%m%d_%H_%M')
    result_path = config.PATH + '/datasets/s_al/' + args.dataset+'_'+ al_method+'_'+ft_method + time.strftime('_%m%d_%H_%M.txt')
    train_set, valid_set, test_set = MoleDataset(prop_name=args.prop_name), MoleDataset(prop_name=args.prop_name), MoleDataset(prop_name=args.prop_name)

    train_set.load_mol(config.train_pkl[args.dataset]), valid_set.load_mol(config.valid_pkl[args.dataset]),test_set.load_mol(config.test_pkl[args.dataset])

    device = torch.device('cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu')

    if args.use_tb:
        writer = SummaryWriter(log_dir=logs_path, comment='baseline_sch')
    else:
        writer = None

    if al_method is 'random' or 'k_center':
        model = SchNetModel(dim=32, n_conv=4, cutoff=5.0, width=0.5, norm=True, output_dim=1)
    elif al_method is 'bayes':
        model = MC_SchNetModel(dim=32, n_conv=4, cutoff=5.0, width=0.5, norm=True, output_dim=1)
    else:
        raise ValueError



    print(model)
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)

    input = {
        'args':args,
        'config':config,
        'train_dataset':train_set,
        'test_dataset':test_set,
        'val_dataset':valid_set,
        'model':model,
        'optimizer':optimizer,
        'writer':writer,
        'device':device,
        'test_freq':args.test_freq,
        'al_method':al_method,
        'ft_method':ft_method,
        'ft_epochs':ft_epochs

    }
    results = active_learning(input)

    '''result file description:
        label_rate  train_loss  train_mae   test_mae
    '''
    with open(result_path, 'w') as fp:
        for key in results.keys():
            fp.write(str(key) + '\t' + ''.join([str(i) + '\t' for i in results[key]]) + '\n')

    print('test success')