import os
import time
from argparse import ArgumentParser
class Global_Config():
    def __init__(self):
        self.PATH = os.path.dirname(__file__)
        self.DATASET_PATH = {'OPV':self.PATH+'/datasets/OPV',
                             'qm9':self.PATH+'/datasets/qm9',
                             'cifar10':self.PATH+'/datasets/cifar10'}

        self.train_pkl = {'OPV':[self.DATASET_PATH['OPV']+'/opv_mol_train{}.pkl'.format(i) for i in range(1,6)],
                          # 'qm9':[self.DATASET_PATH['qm9']+'/qm9_mol_train{}.pkl'.format(i) for i in range(1,6)]
                          'qm9':self.DATASET_PATH['qm9']+'/qm9_mol_train.pkl',
                          'cifar10':self.DATASET_PATH['cifar10']+'/Train.pkl'
                          }

        self.test_pkl = {'OPV':self.DATASET_PATH['OPV']+'/opv_mol_test.pkl',
                          'qm9':self.DATASET_PATH['qm9']+'/qm9_mol_test.pkl',
                         'cifar10':self.DATASET_PATH['cifar10']+'/Test.pkl'}


        self.save_model_path = lambda comment:self.PATH+'/datasets/models'+time.strftime('/%m%d_%H_%M')+comment+'.tar'

        self.mols_dir = {'OPV':self.DATASET_PATH['OPV']+'/opv_mol',
                         'qm9':self.DATASET_PATH['qm9']+'/qm9_mol'}



def make_args():
    parser = ArgumentParser()
    parser.add_argument('--batchsize', type=int, default=64,
                        help='batch size')
    parser.add_argument("--epochs", type=int, default=200,
                        help="number of epochs")
    parser.add_argument('--use_tb', type=bool, default=True,
                        help='whether use tensorboard for logs')
    parser.add_argument('--device', type=int, default=1,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--dataset', type=str, default='qm9',
                        help='which dataset to use')
    parser.add_argument('--save_model', default=True,
                        help='whether save the model')
    parser.add_argument('--workers', default=0,
                        help='number of workers to load data')
    parser.add_argument('--shuffle', default=True,
                        help='whether shuffle data before training')
    parser.add_argument('--multi_gpu', default=False,
                        help='use multi gpu for training')
    parser.add_argument('--use_default',default=False,
                        help='whether use augments in args')
    parser.add_argument('--lr',default=1e-3,
                        help='learning rate')


    # prediction
    parser.add_argument('--prop_name', default='homo',
                      help='which property to predict')


    # universal active learning settings
    parser.add_argument('--batch_data_num', default=200)
    parser.add_argument('--test_freq',default=10)


    #qbc settings
    parser.add_argument('--qbc_ft_epochs', default=5)
    parser.add_argument('--model_num',default=4)
    parser.add_argument('--test_use_all',default=False,
                        help='whether use all models when testing')


    # k-center settings
    parser.add_argument('--init_data_num',default=1000,
                        help='initial data size')
    parser.add_argument('--k_center_ft_epochs',default=10,
                               help='finetuning epochs for k center method')
    args = parser.parse_args()

    return args
