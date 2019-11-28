import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import random
import numpy as np
import sys
from tensorboardX import SummaryWriter
sys.path.append('..')

from bayes_al.mc_sch import MC_SchNetModel
from bayes_al.mm_sch import MM_SchNetModel
from base_model.schmodel import SchNetModel
from geo_al.embedding_model import SchEmbedding
from single_model_al.sampler import AL_sampler, Trainer, Inferencer, check_point_test, save_cpt_xlsx
from utils.funcs import MoleDataset
from config import *


def active_learning(input):
    args           =  input['args']
    config         =  input['config']
    train_dataset  =  input['train_dataset']
    test_dataset   =  input['test_dataset']
    model          =  input['model']
    writer         =  input['writer']
    device         =  input['device']
    al_method      =  input['al_method']
    ft_method      =  input['ft_method']
    ft_epochs      =  input['ft_epochs']
    cpt_data_nums  =  input['checkpoint_data_num']
    cpt_settings   =  input['cpt_settings']
    result_path    =  input['result_path']
    cpt_path       =  input['cpt_path']
    # val_dataset = input['val_dataset']
    # test_freq = input['test_freq']



    print('start {} active learning'.format(al_method))

    ac_info = []
    ac_results = []
    cpk_train_mae = []
    cpk_test_mae = []
    label_rates = []
    train_info = {'total_epochs': [],
                  'train_loss': [],
                  'train_mae': []}
    t_iterations = int((len(train_dataset)-args.init_data_num)/args.batch_data_num) + 1  #discard tail data
    total_data_num = (t_iterations-1)*args.batch_data_num+args.init_data_num
    train_ids = random.sample(range(len(train_dataset)),args.init_data_num)     #record total data not discard


    al_sampler = AL_sampler(args,len(train_dataset),args.batch_data_num,train_ids,al_method)
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
        train_subset_ids = al_sampler.generate_subset(new_batch_ids)
        # train_mols.extend([train_dataset.mols[i] for i in new_batch_ids])
        train_subset = MoleDataset(mols=[train_dataset.mols[i] for i in train_subset_ids])

        expect_data_num = args.init_data_num + iters*args.batch_data_num
        label_rate = expect_data_num / total_data_num

        #finetuning
        # renew train_dataset
        input['train_dataset'] = train_subset
        input['info'] = train_info
        train_info = al_trainer.finetune(input)






        # record results
        if iters % args.test_freq == 0:
            testing_mse, testing_mae = al_trainer.test(test_dataset, model, device)
            print('labels ratio {} number {}  test mae {}'.format(label_rate, len(train_subset), testing_mae))
            label_rates.append(label_rate)
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

            '''result file description:
                    label_rate  train_loss  train_mae   test_mae
                '''
            ac_results = dict(zip(label_rates, ac_info))

            with open(result_path, 'w') as fp:
                for key in ac_results.keys():
                    fp.write(str(key) + '\t' + ''.join([str(i) + '\t' for i in ac_results[key]]) + '\n')
            print('save success')

        # Do checkpoint_test
        # tune hyperparameters of checkpoint model outside !
        if args.test_checkpoint and (expect_data_num in cpt_data_nums):
            labeled_ids = al_sampler.get_label_ids()
            train_ckpset = MoleDataset(mols=[train_dataset.mols[i] for i in labeled_ids])
            _ ,cpk_mae_train, cpk_mae_test = check_point_test(cpt_settings, train_ckpset, test_dataset, device)
            cpk_train_mae.append(cpk_mae_train)
            cpk_test_mae.append(cpk_mae_test)
            save_cpt_xlsx(cpt_path, cpt_data_nums, cpk_train_mae, cpk_test_mae)
            print('checkpoint test record save success')

            # exit when the maximum checkpoint data number is reached
            if expect_data_num >= np.max(cpt_data_nums):
                return ac_results



    return ac_results



if __name__ == "__main__":
    config = Global_Config()
    args = make_args()
    #
    # al_method = 'random'
    # ft_method = 'by_valid'
    # ft_epochs = 2
    al_method = args.al_method
    ft_method = args.ft_method
    ft_epochs = args.ft_epochs

    checkpoint_data_num = [10000,20000,30000,40000,60000]

    cpt_settings = {
        'dim':96,
        'n_conv':4,
        'cutoff':30.0,
        'width':0.1,
        'norm':True,
        'output_dim':1,
        'atom_ref':None,
        'pre_train':None,

        'lr':1e-4,
        'epochs':150,
        'batch_size':64,
        'n_patience':30

    }

    print(args)
    logs_path = config.PATH + '/datasets/logs' + time.strftime('/%m%d_%H_%M')
    result_path = config.PATH + '/datasets/s_al/' + args.dataset+'_'+ al_method+'_'+ft_method + time.strftime('_%m%d_%H_%M.txt')
    checkpoint_test_path = config.PATH + '/datasets/s_al/' + args.dataset + '_' +al_method + time.strftime('_%m%d_%H_%M_cpt.xlsx')
    train_set, valid_set, test_set = MoleDataset(prop_name=args.prop_name), MoleDataset(prop_name=args.prop_name), MoleDataset(prop_name=args.prop_name)

    train_set.load_mol(config.train_pkl[args.dataset]), valid_set.load_mol(config.valid_pkl[args.dataset]),test_set.load_mol(config.test_pkl[args.dataset])

    device = torch.device('cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu')

    if args.use_tb:
        writer = SummaryWriter(log_dir=logs_path, comment='baseline_sch')
    else:
        writer = None


    print(al_method)
    if al_method ==  'random':
        model = SchNetModel(dim=96, n_conv=4, cutoff=30.0, width=0.1, norm=True, output_dim=1)
    elif al_method == 'k_center':
        model = SchEmbedding(dim=96, n_conv=4, cutoff=30.0, width=0.1, norm=True, output_dim=1)
    elif al_method == 'bayes':
        model = MC_SchNetModel(dim=96, n_conv=4, cutoff=30.0, width=0.1, norm=True, output_dim=1)
    elif al_method == 'msg_mask':
        model = MM_SchNetModel(dim=96, n_conv=4, cutoff=30.0, width=0.1, norm=True, output_dim=1)
    else:
        raise ValueError



    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
    print(model)
    print(optimizer)
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
        'ft_epochs':ft_epochs,
        'checkpoint_data_num': checkpoint_data_num,
        'cpt_settings':cpt_settings,
        'result_path': result_path,
        'cpt_path':checkpoint_test_path
    }
    results = active_learning(input)



    print('test success')