import pickle
from utils import *
import rdkit
from rdkit import Chem
from rdkit import DataStructs
import numpy as np
from rdkit.Chem import rdMolDescriptors
from config import Global_Config
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import time
import torch
config = Global_Config()

# zinc_data = pickle.load(open('zinc_clean_smi.pkl','rb'))
# chemge_data = pickle.load(open('chemge_mol_100000.pkl','rb'))
# cgam_data = pickle.load(open('max_logp_total_pool.pkl','rb'))

#
# def get_top100(dataset, name):
#     length = len(dataset)
#     logp_set = []
#
#     for i in range(length):
#         logp = calc_score(Chem.MolFromSmiles(dataset[i]))
#         logp_set.append(logp)
#         print(i)
#
#     tuple = [(dataset[i],logp_set[i]) for i in range(length)]
#     sorted_tuple = sorted(tuple, key=lambda x:x[1], reverse=True)
#
#     print(sorted_tuple[:100])
#     pickle.dump(sorted_tuple[:100], open(name,'wb'))
#
# get_top100(cgam_data, 'cgam_top100.pkl')

# zinc_top100_name = 'zinc_top100.pkl'
# chemge_top100_name = 'chemge_top100.pkl'
# cgam_top100_name = 'cgam_top100.pkl'
#
# data_len = 100
#
# zinc_data = pickle.load(open(zinc_top100_name,'rb'))
# zinc_smi = [zinc_data[i][0] for i in range(data_len)]
#
# chemge_data = pickle.load(open(chemge_top100_name,'rb'))
# chemge_smi = [chemge_data[i][0] for i in range(data_len)]
#
# cgam_data = pickle.load(open(cgam_top100_name,'rb'))
# cgam_smi = [cgam_data[i][0] for i in range(data_len)]
#
# print(zinc_smi)
# print(chemge_smi)
# print(cgam_smi)
#
# def smilarity_between_two_mols(smi1, smi2):
#     mol1, mol2 = Chem.MolFromSmiles(smi1), Chem.MolFromSmiles(smi2)
#
#     vec1 = Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol1, 4, nBits=2048)
#     vec2 = Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol2, 4, nBits=2048)
#
#     tani = DataStructs.TanimotoSimilarity(vec1, vec2)
#     return tani

# res1 = []
# for i in range(100):
#     g_smi = chemge_smi[i]
#     pool = [smilarity_between_two_mols(g_smi, zinc_smi[j]) for j in range(100)]
#     max_smilarity = max(pool)
#     #print(max_smilarity)
#     res1.append(max_smilarity)
# print(np.mean(res1))
#
# res2 = []
# for i in range(100):
#     g_smi = cgam_smi[i]
#     pool = [smilarity_between_two_mols(g_smi, zinc_smi[j]) for j in range(100)]
#     max_smilarity = max(pool)
#     #print(max_smilarity)
#     res2.append(max_smilarity)
# print(np.mean(res2))

n_bit = 512
def smi2vec(smi):
    mol = Chem.MolFromSmiles(smi)
    bit_vec = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 4, nBits=n_bit)
    vec = [bit_vec[i] for i in range(n_bit)]

    return vec



qm9_data_path = config.train_pkl['qm9']
mols = pickle.load(open(qm9_data_path,'rb'))
smis = [mol.smi for mol in mols]

time0 = time.time()
fingerprints = [smi2vec(smi) for smi in smis]
n_components = 20

print('time {}'.format(time.time() - time0))
time0 = time.time()

pca = PCA(n_components=n_components)
pca.fit(fingerprints)

qm9_pca = pca.transform(fingerprints)





print('time {}'.format(time.time()-time0))
plt.scatter(qm9_pca[:,0],qm9_pca[:,1],marker='.')
plt.savefig('qm9_pca.png')

qm9_pca_t = torch.Tensor(qm9_pca)
save_path = config.DATASET_PATH['qm9']+'/qm9_fingerprint_'+str(n_components)+'.pkl'

pickle.dump(qm9_pca_t,open(save_path,'wb'))







# zinc_data = pickle.load(open('zinc_clean_smi.pkl','rb'))
#
# zinc_top100_name = 'zinc_top100.pkl'
# chemge_top100_name = 'chemge_top100.pkl'
# cgam_top100_name = 'cgam_top100.pkl'
#
# zinc_top100_smi = [pickle.load(open(zinc_top100_name,'rb'))[i][0] for i in range(100)]
# chemvae_top100_smi = [pickle.load(open(chemge_top100_name,'rb'))[i][0] for i in range(100)]
# cgam_top100_smi = [pickle.load(open(cgam_top100_name,'rb'))[i][0] for i in range(100)]
#
# zinc_data_len = len(zinc_data)
# selected_index = np.random.choice(zinc_data_len, 20000, replace=False)
#
# train_data = [smi2vec(zinc_data[i]) for i in selected_index]
#
# zinc_top_data = [smi2vec(zinc_top100_smi[i]) for i in range(100)]
# chemvae_top_data = [smi2vec(chemvae_top100_smi[i]) for  i in range(100)]
# cgam_top_data = [smi2vec(cgam_top100_smi[i]) for i in range(100)]
#
#
# pca = PCA(n_components=2)
# pca.fit(train_data)
#
# zinc_pca = pca.transform(zinc_top_data)
# chemvae_pca = pca.transform(chemvae_top_data)
# cgam_pca = pca.transform(cgam_top_data)
#
# df_zinc = pd.DataFrame(np.transpose((zinc_pca[:,0],zinc_pca[:,1])))
# df_zinc.columns = ['x','y']
#
# plt.scatter(x=df_zinc['x'], y=df_zinc['y'], c = 'y',
#             cmap= 'viridis', marker='.',
#             s=100,alpha=0.5, edgecolors='none',label='ZINC')
#
# df_chemvae = pd.DataFrame(np.transpose((chemvae_pca[:,0],chemvae_pca[:,1])))
# df_chemvae.columns = ['x','y']
#
# plt.scatter(x=df_chemvae['x'], y=df_chemvae['y'], c = 'b',
#             cmap= 'viridis', marker='.',
#             s=100,alpha=0.5, edgecolors='none',label='ChemVAE')
#
# df_cgam = pd.DataFrame(np.transpose((cgam_pca[:,0],cgam_pca[:,1])))
# df_cgam.columns = ['x','y']
#
# plt.scatter(x=df_cgam['x'], y=df_cgam['y'], c = 'r',
#             cmap= 'viridis', marker='.',
#             s=100,alpha=0.5, edgecolors='none',label='CGAM')
#
# plt.legend(loc='upper right')
# plt.savefig('comparison_top100.png')