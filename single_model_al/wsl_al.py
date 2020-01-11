import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import random
import numpy as np
import sys
from tensorboardX import SummaryWriter
sys.path.append('..')

from pre_training.wsch import WSchnet, Semi_Schnet
from single_model_al.sampler import AL_sampler, Inferencer, check_point_test, save_cpt_xlsx, Weakly_Supervised_Trainer, get_preds_w
from utils.funcs import MoleDataset, k_center, SelfMolDataSet, k_medoid, get_atom_ref
from config import Global_Config, make_args


'''Weakly supervised learning for active learning on graphs'''


def active_learning(input):
    args = input['args']
    config = input['config']
    train_dataset = input['train_dataset']
    test_dataset = input['test_dataset']
    model_l = input['model']
    writer = input['writer']
    device = input['device']
    ft_method = input['ft_method']
    cpt_data_nums = input['checkpoint_data_num']
    al_settings = input['al_settings']
    result_path = input['result_path']
    cpt_path = input['cpt_path']
    checkpoint_epochs_num = input['checkpoint_epochs_num']

    # al_method = input['al_method']
    # val_dataset = input['val_dataset']
    # test_freq = input['test_freq']
    # ft_epochs = input['ft_epochs']

    print('start weakly supervised active learning')
    al_method = 'k_center'
    cpk_test_iter = 0
    ac_info = []
    ac_results = []
    cpk_train_mae = []
    cpk_test_mae = []
    label_rates = []
    train_info = {'total_epochs': [],
                  'train_loss': [],
                  'train_mae': []}
    p_labels = torch.zeros(len(train_dataset)).long()

    t_iterations = int((len(train_dataset) - args.init_data_num) / args.batch_data_num) + 1  # discard tail data
    total_data_num = (t_iterations - 1) * args.batch_data_num + args.init_data_num
    # train_ids = random.sample(range(len(train_dataset)), args.init_data_num)  # record total data not discard

    dataset_s = SelfMolDataSet(mols = train_dataset.mols,level='w',prop_name=args.prop_name)
    # al_inferencer = Inferencer(args,al_method)
    al_trainer = Weakly_Supervised_Trainer(args,al_settings)
    # al_trainer = Trainer(args, t_iterations, method=ft_method, ft_epochs=ft_epochs)
    al_trainer.run(model_l,dataset_s,optimizer,device,writer,None,level='g')
    preds = get_preds_w(args,model_l,dataset_s,device)
    # train_ids = k_center(preds.cpu(),args.init_data_num)
    train_ids = k_medoid(preds.cpu(),args.init_data_num,al_settings['iters'],None,show_stats=True)
    al_sampler = AL_sampler(args, len(train_dataset), args.batch_data_num, train_ids, al_method)

    # initialization training
    # train_mols = [train_dataset.mols[i] for i in train_ids]
    # train_subset = MoleDataset(mols=train_mols)
    # input['train_dataset'] = train_subset
    input['info'] = train_info

    for iters in range(0, t_iterations):

        expect_data_num = args.init_data_num + iters * args.batch_data_num
        label_rate = expect_data_num / total_data_num
        labeled_ids = al_sampler.get_label_ids()
        unlabeled_ids = al_sampler.get_unlabeled_ids()
        # Do checkpoint_test
        # tune hyperparameters of checkpoint model outside !
        if expect_data_num in cpt_data_nums :
            train_ckpset = MoleDataset(mols=[train_dataset.mols[i] for i in labeled_ids],prop_name=args.prop_name)
            print('start checkpoint testing iter {} with labels {}'.format(cpk_test_iter,len(train_ckpset)))
            model_h, cpk_mae_train, cpk_mae_test = check_point_test(al_settings, train_ckpset, test_dataset, model_l,checkpoint_epochs_num[cpk_test_iter], device)
            cpk_train_mae.append(cpk_mae_train)
            cpk_test_mae.append(cpk_mae_test)
            save_cpt_xlsx(cpt_path, cpt_data_nums, cpk_train_mae, cpk_test_mae)
            print('checkpoint test record save success')
            # exit when the maximum checkpoint data number is reached
            if expect_data_num >= np.max(cpt_data_nums):
                return ac_results
            else:
                # generate pesudo labels for next iteration
                p_labels = al_trainer.generate_p_labels(model_h, train_dataset,labeled_ids,unlabeled_ids, args.prop_name, device)

            cpk_test_iter += 1

        train_info = al_trainer.run(model_l,dataset_s,optimizer,device,writer,p_labels,level='w')
        preds = get_preds_w(args,model_l,dataset_s,device)
        new_batch_ids = al_sampler.query(preds)
        # train_subset_ids = al_sampler.generate_subset(new_batch_ids)
        # train_subset = MoleDataset(mols=[train_dataset.mols[i] for i in train_subset_ids])


        # input['train_dataset'] = train_subset
        input['info'] = train_info

        # record results
        if iters % args.test_freq == 0:
            # # testing_mse, testing_mae = al_trainer.test(test_dataset, model, device)
            print('labels ratio {} '.format(label_rate))
            # label_rates.append(label_rate)
            # ac_info.append((train_info['train_loss'][-1], train_info['train_mae'][-1], testing_mse, testing_mae))

            # if args.use_tb:
            #     writer.add_scalar('test_mae', testing_mae, label_rate)
            if args.save_model:
                torch.save({
                    'info_train': train_info,
                    # 'testing_mae': testing_mae,
                    'model': model.state_dict(),
                    'data_ids': al_sampler.get_label_ids()
                }, config.save_model_path(args.dataset + al_method + '_' + ft_method))

            '''result file description:
                    label_rate  train_loss  train_mae   test_mae
                '''
            ac_results = dict(zip(label_rates, ac_info))

            with open(result_path, 'w') as fp:
                for key in ac_results.keys():
                    fp.write(str(key) + '\t' + ''.join([str(i) + '\t' for i in ac_results[key]]) + '\n')
            print('save success')



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

    checkpoint_data_num =   [5000,10000,20000,30000,40000,50000,60000,70000]
    # checkpoint_epochs_num = [1500, 1000 ,1000 ,1000 ,600  ,400  ,300  ,300]
    checkpoint_epochs_num = [800, 500 ,500 ,500 ,400  ,200  ,200  ,200]

    # checkpoint_epochs_num = [2,2,2,2,2,2,2,2]
    al_settings = {
        'dim':96,
        'n_conv':4,
        'cutoff':30.0,
        'width':0.1,
        'norm':False,
        'output_dim':1,
        'atom_ref':get_atom_ref(args.prop_name),
        'pre_train':None,

        'lr':1e-3,
        'epochs':150,
        'batch_size':64,
        'n_patience':55,

        'cls_method':'ot',
        'prop_bins':25,
        'cls_num':100,
        'cls_epochs':6,
        'iters':6,
        'init_method':'k_medroids',



    }

    # Attention we found normalize will hurt the performance of optimal transport ???*****

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


    # atom_ref = get_atom_ref(args.prop_name)
    # model = WSchnet(dim=256, n_conv=4, cutoff=30.0, width=0.1, norm=True, output_dim=1,props_bins=al_settings['prop_bins'],cls_dim=al_settings['cls_num'])
    model = Semi_Schnet(dim=96,n_conv=4,cutoff=30.0,cls_dim=al_settings['cls_num'],width=0.1,norm=False, output_dim=1,edge_bins=150,mask_n_ratio=args.mask_n_ratio,mask_msg_ratio=0,props_bins=al_settings['prop_bins'],atom_ref=al_settings['atom_ref'])


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
        'checkpoint_epochs_num':checkpoint_epochs_num,
        'al_settings':al_settings,
        'result_path': result_path,
        'cpt_path':checkpoint_test_path
    }
    results = active_learning(input)



    print('test success')