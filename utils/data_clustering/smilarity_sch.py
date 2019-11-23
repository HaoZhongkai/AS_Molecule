#!/usr/bin/env python  
#-*- coding:utf-8 _*-

import pickle
from utils import *
import rdkit
from rdkit import Chem
from rdkit import DataStructs
import numpy as np
from rdkit.Chem import rdMolDescriptors
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import dgl
import torch

from utils.funcs import batcher, MoleDataset, k_medoid, k_medoids_pp
from config import Global_Config, make_args
import random
from pre_training.sch_embeddings import SchEmbedding
config = Global_Config()
args = make_args()


'''观察由随机的(或者预训练的)schnet输出的graph embedding的pca'''
n_bit = 512
def smi2vec(smi):
    mol = Chem.MolFromSmiles(smi)
    bit_vec = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 4, nBits=n_bit)
    vec = [bit_vec[i] for i in range(n_bit)]

    return vec

def get_preds(args,model,dataset,device):
    time0 = time.time()
    dataloader = DataLoader(dataset=dataset, batch_size=args.batchsize*5, collate_fn=batcher,shuffle=False, num_workers=args.workers)
    model.to(device)
    model.set_mean_std(dataset.mean,dataset.std)
    embeddings = []
    with torch.no_grad():
        for idx,(mols,_) in enumerate(dataloader):
            g = dgl.batch([mol.ful_g for mol in mols])
            g.to(device)
            embedding = model.embed_g(g)
            embeddings.append(embedding)

    embeddings = torch.cat(embeddings,dim=0)
    print('inference {}'.format(time.time()-time0))

    return embeddings


qm9_data_path = config.train_pkl['qm9']
qm9_data_path_te = config.test_pkl['qm9']

mols = pickle.load(open(qm9_data_path,'rb'))
mols_te = pickle.load(open(qm9_data_path_te,'rb'))
# smis = [mol.smi for mol in mols]

model = SchEmbedding(dim=48, n_conv=4, cutoff=5.0, width=0.5, norm=True, output_dim=1)
dataset = MoleDataset(mols=mols,prop_name='homo')
dataset_te = MoleDataset(mols=mols_te,prop_name='homo')

embeddings = get_preds(args,model,dataset,torch.device(args.device))

embeddings_te = get_preds(args,model,dataset_te,torch.device(args.device))


# embeddings = embeddings.cpu()
# embeddings_te = embeddings_te.cpu()
center_ids = k_medoids_pp(embeddings,5000,10,show_stats=True)
# center_ids = random.sample(range(embeddings.shape[0]),5000)


embeddings = embeddings.numpy()
embeddings_te = embeddings_te.numpy()
# fingerprints = [smi2vec(smi) for smi in smis]



n_components = 2

time0 = time.time()

pca = PCA(n_components=n_components)
pca.fit(embeddings)

qm9_pca = pca.transform(embeddings)

qm9_pca_te =  pca.transform(embeddings_te)


print('time {}'.format(time.time()-time0))
plt.scatter(qm9_pca[:,0],qm9_pca[:,1],marker='.',color='b',label='training data')
plt.scatter(qm9_pca[center_ids,0],qm9_pca[center_ids,1],marker='.',color='r',label='cluster_centers')
# plt.scatter(qm9_pca_te[:,0],qm9_pca_te[:,1],marker='.',color='g',label='test data')
plt.legend()
plt.savefig('qm9_pca_sch.png')

# qm9_pca_t = torch.Tensor(qm9_pca)

# save_path = config.DATASET_PATH['qm9']+'/qm9_fingerprint_'+str(n_components)+'.pkl'
save_path = config.DATASET_PATH['qm9']+'/sch_ebd.pkl'

pickle.dump(qm9_pca,open(save_path,'wb'))