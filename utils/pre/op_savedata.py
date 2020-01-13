import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(),"../..")))

from utils.funcs import Molecule
import numpy as np
from multiprocessing import Manager, Process
import torch
import dgl
from utils.funcs import MoleDataset
from config import Global_Config as Config
config = Config()
def get_mol(data):
    pos, atoms, edges, smi, prop, dists = data
    return Molecule(pos, atoms, edges, smi, prop, distance=dists, loc=False, glob=True)

class M():
    L = 1
    def __init__(self,num):
        self.num = num
        self.ar = np.array([5])
        self.th = torch.Tensor([num,num])
        self.g = dgl.DGLGraph()
        self.build()

    def build(self):
        self.g.add_nodes(self.num)



def get_M(num):
    return M(num)



if __name__ == '__main__':
    # num = range(10000)
    # mp = _mp.get_context('spawn')
    # cnt = 0
    # pool = mp.Pool(10)
    # for _ in pool.imap(get_M, num):
    #     sys.stdout.write('id{}\n'.format(cnt))
    #     cnt+=1
    # manager = Manager()


    # path = config.PATH+'/datasets/OPV/data_elem_train.pkl'
    path = config.PATH+'/datasets/OPV/data_elem_test.pkl'

    # save_path = [config.PATH+'/datasets/OPV/opv_mol_train1.pkl',
    #              config.PATH+'/datasets/OPV/opv_mol_train2.pkl',
    #              config.PATH+'/datasets/OPV/opv_mol_train3.pkl',
    #              config.PATH+'/datasets/OPV/opv_mol_train4.pkl',
    #              config.PATH+'/datasets/OPV/opv_mol_train5.pkl',
    #
    #              ]
    save_path = config.PATH+'/datasets/OPV/opv_mol_test.pkl'

    dataset = MoleDataset(path=path,prop_name='homo',loc=False,glob=True)
    dataset.build()
    dataset.save_mol(save_path)
    print('ok')




