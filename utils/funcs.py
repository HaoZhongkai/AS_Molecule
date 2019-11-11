import networkx as nx
from rdkit import Chem
import numpy as np
import torch
import dgl
from torch.utils.data import Dataset
import pickle
from torch.nn import DataParallel
import os
import torch.utils.data as data
# import torch.multiprocessing as mp
# torch.multiprocessing.set_sharing_strategy('file_system')


def get_mol(data):
    return Molecule(*data)

# PROPID = {'optical_lumo':0,'gap':1,'homo':2,'lumo':3,'spectral_overlap':4,'delta_homo':5,'delta_lumo':6,'delta_optical_lumo':7,'homo_extrapolated':8,'lumo_extrapolated':9,'gap_extrapolated':10,'optical_lumo_extrapolated':11}

class Molecule():
    EDGE_TYPE = [
        Chem.BondType.SINGLE,
        Chem.BondType.DOUBLE,
        Chem.BondType.TRIPLE,
        Chem.BondType.AROMATIC
    ]
    NODE_TYPE = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17}
    PROPID = {'optical_lumo': 0, 'gap': 1, 'homo': 2, 'lumo': 3, 'spectral_overlap': 4}

    def __init__(self, coords, atoms, edges, smi, props, distance=None, loc=False, glob=True):

        self.coordinates = torch.tensor(coords)
        self.atoms = atoms      # types corresponds to coordiantes
        self.node_num = len(atoms)
        self.edges = edges
        self.smi = smi


        # self.optical_lumo,self.gap,self.homo,self.lumo,self.spectral_overlap,self.delta_homo,self.delta_lumo,self.delta_optical_lumo,self.homo_extrapolated,self.lumo_extrapolated,self.gap_extrapolated,self.optical_lumo_extrapolated = props
        self.props = props      #dict
        # self.nx_g, self.nidx, self.loc_g, self.ful_g =  nx.Graph(), 0, dgl.DGLGraph(), dgl.DGLGraph() #init


        self._build_nidx()
        if loc:
            self._build_loc()
        if glob:
            self._build_ful(distance)
            # pass

    def _build_nx_g(self):
        # first build mol
        self.mol = Chem.RWMol()
        for i in range(len(self.atoms)):
            self.mol.AddAtom(Chem.rdchem.Atom(self.atoms[i]))
        for i in range(len(self.edges)):
            self.mol.AddBond(int(self.edges[i][0]), int(self.edges[i][1]), Molecule.EDGE_TYPE[self.edges[i][2]])
        self.nx_g = mol2nx(self.mol)
        return self

    # build a dgl graph only contains edges existing considering edge relation,now no self edge
    def _build_loc(self):
        self.loc_g = dgl.DGLGraph()
        self.loc_g.add_nodes(self.node_num)
        self.loc_g.add_edges(self.edges[:,0], self.edges[:,1], data={'edge_type':torch.tensor(self.edges[:,2]).long()})
        self.loc_g = dgl.to_bidirected(self.loc_g)
        self.loc_g.ndata['pos'] = self.coordinates
        self.loc_g.ndata['nodes'] = self.nidx

    # build a dgl graph for long interaction(full graph)
    def _build_ful(self,distance_matrix=None):
        self.ful_g = dgl.DGLGraph()
        self.ful_g.add_nodes(self.node_num)
        self.ful_g.add_edges([i for i in range(self.node_num) for j in range(self.node_num)],
                             [j for i in range(self.node_num) for j in range(self.node_num)])
        # self.ful_g.add_edges(self.ful_g.nodes(), self.ful_g.nodes())    #add self edge
        self.ful_g.ndata['pos'] = self.coordinates
        self.ful_g.ndata['nodes'] = self.nidx

        if distance_matrix is None:
            distance_matrix = torch.zeros(self.node_num*self.node_num)
            for i in range(self.node_num):
                distance_matrix[i*self.node_num:(i+1)*self.node_num] = torch.norm(self.coordinates[i]-self.coordinates,p=2,dim=1)
        else:
            distance_matrix = torch.Tensor(distance_matrix)
        self.ful_g.edata['distance'] = distance_matrix
        return


    def _build_nidx(self):
        self.nidx = torch.zeros(self.node_num).long()
        for i in range(self.node_num):
            self.nidx[i] = Molecule.NODE_TYPE[self.atoms[i]]
        return





def chunks(l, n):
    l_chunk = []
    chunk_size = int(len(l)/n)
    for i in range(0, n-1):
        l_chunk.append(l[i*chunk_size:(i+1)*chunk_size])
    l_chunk.append(l[(n-1)*chunk_size:])
    return l_chunk

class MoleDataset(Dataset):
    def __init__(self,path=None, mols=None,datas=None, prop_name='homo', loc=False, glob=True):
        self.path = path
        self.loc_g = loc
        self.glob_g = glob
        self.prop_name = prop_name
        self.datas = datas
        self.mols = mols
        if self.path:
            # data element ( pos, atoms, edges, smi, props, dists)
            self.datas = pickle.load(open(path,'rb'))
            # self.mols = [[] for i in range(self.__len__())]
            # self.prop = torch.Tensor(np.stack([data[4] for data in self.datas],axis=0)[:,Molecule.PROPID[prop_name]])
        self.__get_props()



    def __len__(self):
        if self.datas:
            return len(self.datas)
        else:
            return len(self.mols)


    def __getitem__(self, item):
        return self.mols[item], self.prop[item]


    def __get_props(self):
        if self.mols:
            self.prop = torch.Tensor([mol.props[self.prop_name] for mol in self.mols])
            self.mean = torch.mean(self.prop)
            self.std = torch.std(self.prop)
        elif self.datas:
            self.prop_keys = self.datas[0][4].keys()
            self.props = {}
            for key in self.prop_keys:
                self.props[key] = torch.Tensor([data[4][key] for data in self.datas])
            # mean for only properties to predict
            self.prop = self.props[self.prop_name]
            self.mean = torch.mean(self.props[self.prop_name])
            self.std = torch.std(self.props[self.prop_name])
        else:
            assert 'Not initialized dataset'

    # Now no parallel, load all to dataset
    def build(self):
        # mp = torch.multiprocessing.get_context('fork')
        # cnt = 0
        # pool = mp.Pool(processes=5)
        # for _ in pool.imap(get_mol, self.datas):
        #     self.mols[cnt] = _
        #     cnt+=1
        #     print(cnt)
        self.mols = [[] for i in range(self.__len__())]
        for i in range(self.__len__()):
            pos, atoms, edges, smi, prop, dists = self.datas[i]
            self.mols[i] = Molecule(pos,atoms,edges,smi, prop, distance=dists,loc=self.loc_g,glob=self.glob_g)
            print(i)
        return


    def load_mol(self,paths):
        self.mols = []
        if type(paths) is list:
            for path in paths:
                self.mols.extend(pickle.load(open(path,'rb')))
                print('loading...')
        else:
            self.mols = pickle.load(open(paths,'rb'))
        self.__get_props()

    def save_mol(self,paths):
        if type(paths) is list:
            path_num = len(paths)
            mols_chunks = chunks(self.mols,path_num)
            for i in range(path_num):
                pickle.dump(mols_chunks[i],open(paths[i],'wb'))
        else:
            pickle.dump(self.mols,open(paths,'wb'))
        return

# deal with large molecules, load mols from seperate .pkl
class FMolDataSet(Dataset):
    def __init__(self,dir,prop_name='homo',data_ids=None):
        self.dir = dir
        self.prop_name = prop_name
        self.data_ids = np.sort(data_ids) if data_ids is not None else np.arange(len(os.listdir(dir))-1)
        self.data_num = len(self.data_ids)
        self.__get_props()


    def __len__(self):
        return self.data_num



    def __getitem__(self, item):
        return pickle.load(open(self.dir+'/mol_{0:06d}'.format(self.data_ids[item])+'.pkl','rb')),self.prop[item]

    # data id must be increasing list
    def __get_props(self):
        props = pickle.load(open(self.dir+'/props.pkl','rb'))
        props = [props[i] for i in self.data_ids]
        self.prop = torch.Tensor([prop[self.prop_name] for prop in props])
        self.mean = torch.mean(self.prop)
        self.std = torch.std(self.prop)



def get_dataset_from_files(paths, prop_name):
    if type(paths) is list:
        props = []
        mols = []
        for path in paths:
            mols_, props_ = pickle.load(open(path,'rb'))
            mols.extend(mols_), props.append(props_)
        props = np.concatenate(props,axis=0)
        return MoleDataset(mols, torch.Tensor(props), prop_name)
    else:
        mols, props = pickle.load(open(paths,'rb'))
        return MoleDataset(mols, torch.Tensor(props), prop_name)

def load_dataset(path,prop_name,loc_g=False,glob_g=True):
    datas = pickle.load(open(path,'rb'))
    print('load success, preprocessing...')
    mols = []
    props = []
    i = 0
    for data in datas:
        pos, atoms, edges, smi, prop, dists = data
        mols.append(Molecule(pos,atoms,edges,smi,prop,distance=dists,loc=False,glob=True))
        props.append(prop)
        print(i)
        i += 1
    props = np.stack(props,axis=0)
    return MoleDataset(mols, torch.Tensor(props), prop_name)


# a list of tuple to tuple of list
def batcher(input):
    mols, props = zip(*input)
    props = torch.stack(props,dim=0)
    return mols, props

# dataparallel support set_mean and std for model
class RefDataParallel(DataParallel):
    def set_mean_std(self,mean,std):
        return getattr(self.module, 'set_mean_std')(mean,std)

def mol2nx(mol):
    G = nx.Graph()

    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(),
                   atomic_num=atom.GetAtomicNum(),
                   formal_charge=atom.GetFormalCharge(),
                   chiral_tag=atom.GetChiralTag(),
                   hybridization=atom.GetHybridization(),
                   num_explicit_hs=atom.GetNumExplicitHs(),
                   is_aromatic=atom.GetIsAromatic())
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx(),
                   bond_type=bond.GetBondType())
    return G


class Cifar(data.Dataset):
    def __init__(self,data, label ):
        '''获取并划分数据'''
        self.data = data
        self.label = label
        self.num = self.data.shape[0]

    def __getitem__(self, index):
        data = torch.Tensor(self.data[index])
        label = torch.Tensor(self.label[index])
        return data,label

    def __len__(self):
        return self.num



def nx2mol(G):
    mol = Chem.RWMol()
    atomic_nums = nx.get_node_attributes(G, 'atomic_num')
    chiral_tags = nx.get_node_attributes(G, 'chiral_tag')
    formal_charges = nx.get_node_attributes(G, 'formal_charge')
    node_is_aromatics = nx.get_node_attributes(G, 'is_aromatic')
    node_hybridizations = nx.get_node_attributes(G, 'hybridization')
    num_explicit_hss = nx.get_node_attributes(G, 'num_explicit_hs')
    node_to_idx = {}
    for node in G.nodes():
        a=Chem.Atom(atomic_nums[node])
        a.SetChiralTag(chiral_tags[node])
        a.SetFormalCharge(formal_charges[node])
        a.SetIsAromatic(node_is_aromatics[node])
        a.SetHybridization(node_hybridizations[node])
        a.SetNumExplicitHs(num_explicit_hss[node])
        idx = mol.AddAtom(a)
        node_to_idx[node] = idx

    bond_types = nx.get_edge_attributes(G, 'bond_type')
    for edge in G.edges():
        first, second = edge
        ifirst = node_to_idx[first]
        isecond = node_to_idx[second]
        bond_type = bond_types[first, second]
        mol.AddBond(ifirst, isecond, bond_type)

    # Chem.SanitizeMol(mol)
    return mol






# class MoleDataset(Dataset):
#     def __init__(self, mols, props,prop_name):
#         self.mols = mols
#         self.prop = props[:,Molecule.PROPID[prop_name]]
#         self.mean = torch.mean(self.prop)
#         self.std = torch.std(self.prop)
#
#
#     def __len__(self):
#         return len(self.mols)
#
#
#     def __getitem__(self, item):
#         return self.mols[item], self.prop[item]