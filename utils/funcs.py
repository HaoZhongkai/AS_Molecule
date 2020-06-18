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
import random
import time
from sklearn.cluster import KMeans
# import torch.multiprocessing as mp
# torch.multiprocessing.set_sharing_strategy('file_system')


def get_mol(data):
    return Molecule(*data)


# PROPID = {'optical_lumo':0,'gap':1,'homo':2,'lumo':3,'spectral_overlap':4,'delta_homo':5,'delta_lumo':6,'delta_optical_lumo':7,'homo_extrapolated':8,'lumo_extrapolated':9,'gap_extrapolated':10,'optical_lumo_extrapolated':11}


def get_atom_ref(prop_name=None):
    atom_ref_ids = [1, 6, 7, 8, 9]  # H,C,N,O,F
    atom_refs = {
        'zpve':
        torch.Tensor([0, 0, 0, 0, 0]),
        'U0':
        torch.Tensor(
            [-0.500273, -37.846772, -54.583861, -75.064579, -99.718730]),
        'U':
        torch.Tensor(
            [-0.498857, -37.845355, -54.582445, -75.063163, -99.717314]),
        'H':
        torch.Tensor(
            [-0.497912, -37.844411, -54.581501, -75.062219, -99.716370]),
        'G':
        torch.Tensor(
            [-0.510927, -37.861317, -54.598897, -75.079532, -99.733544]),
        'Cv':
        torch.Tensor([2.981, 2.981, 2.981, 2.981, 2.981])
    }
    atom_ref = torch.zeros([100])
    if prop_name in ['zpve', 'U0', 'U', 'H', 'G', 'Cv']:
        atom_ref[atom_ref_ids] = atom_refs[prop_name]
        atom_ref = atom_ref.unsqueeze(1)

    else:
        atom_ref = None
    return atom_ref


class Molecule():
    EDGE_TYPE = [
        Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE,
        Chem.BondType.AROMATIC
    ]
    NODE_TYPE = {
        'H': 1,
        'C': 6,
        'N': 7,
        'O': 8,
        'F': 9,
        'Si': 14,
        'P': 15,
        'S': 16,
        'Cl': 17
    }
    PROPID = {
        'optical_lumo': 0,
        'gap': 1,
        'homo': 2,
        'lumo': 3,
        'spectral_overlap': 4
    }

    def __init__(self,
                 coords,
                 atoms,
                 edges,
                 smi,
                 props,
                 distance=None,
                 loc=False,
                 glob=True):

        self.coordinates = torch.tensor(coords)
        self.atoms = atoms  # types corresponds to coordiantes
        self.node_num = len(atoms)
        self.edges = edges
        self.smi = smi

        # self.optical_lumo,self.gap,self.homo,self.lumo,self.spectral_overlap,self.delta_homo,self.delta_lumo,self.delta_optical_lumo,self.homo_extrapolated,self.lumo_extrapolated,self.gap_extrapolated,self.optical_lumo_extrapolated = props
        self.props = props  #dict
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
            self.mol.AddBond(int(self.edges[i][0]), int(self.edges[i][1]),
                             Molecule.EDGE_TYPE[self.edges[i][2]])
        self.nx_g = mol2nx(self.mol)
        return self

    # build a dgl graph only contains edges existing considering edge relation,now no self edge
    def _build_loc(self):
        self.loc_g = dgl.DGLGraph()
        self.loc_g.add_nodes(self.node_num)
        self.loc_g.add_edges(
            self.edges[:, 0],
            self.edges[:, 1],
            data={'edge_type': torch.tensor(self.edges[:, 2]).long()})
        self.loc_g = dgl.to_bidirected(self.loc_g)
        self.loc_g.ndata['pos'] = self.coordinates
        self.loc_g.ndata['nodes'] = self.nidx

    # build a dgl graph for long interaction(full graph)
    def _build_ful(self, distance_matrix=None):
        self.ful_g = dgl.DGLGraph()
        self.ful_g.add_nodes(self.node_num)
        self.ful_g.add_edges(
            [i for i in range(self.node_num) for j in range(self.node_num)],
            [j for i in range(self.node_num) for j in range(self.node_num)])
        # self.ful_g.add_edges(self.ful_g.nodes(), self.ful_g.nodes())    #add self edge
        self.ful_g.ndata['pos'] = self.coordinates
        self.ful_g.ndata['nodes'] = self.nidx

        if distance_matrix is None:
            distance_matrix = torch.zeros(self.node_num * self.node_num)
            for i in range(self.node_num):
                distance_matrix[i * self.node_num:(i + 1) *
                                self.node_num] = torch.norm(
                                    self.coordinates[i] - self.coordinates,
                                    p=2,
                                    dim=1)
        else:
            distance_matrix = torch.Tensor(distance_matrix)
        self.ful_g.edata['distance'] = distance_matrix
        return

    def _build_nidx(self):
        self.nidx = torch.zeros(self.node_num).long()
        for i in range(self.node_num):
            self.nidx[i] = Molecule.NODE_TYPE[self.atoms[i]]
        return


class Molecule_MD(object):
    NODE_TYPE = {
        'H': 1,
        'C': 6,
        'N': 7,
        'O': 8,
        'F': 9,
        'Si': 14,
        'P': 15,
        'S': 16,
        'Cl': 17
    }

    def __init__(self, coords, atoms, forces, props, distance):

        self.coordinates = torch.tensor(coords)
        self.atoms = atoms  # types corresponds to coordiantes
        self.forces = torch.tensor(forces)
        self.node_num = len(atoms)

        self.props = props

        self._build_nidx()
        self._build_ful()

    def _build_ful(self):
        self.ful_g = dgl.DGLGraph()
        self.ful_g.add_nodes(self.node_num)
        self.ful_g.add_edges(
            [i for i in range(self.node_num) for j in range(self.node_num)],
            [j for i in range(self.node_num) for j in range(self.node_num)])
        # self.ful_g.add_edges(self.ful_g.nodes(), self.ful_g.nodes())    #add self edge
        self.ful_g.ndata['pos'] = self.coordinates
        self.ful_g.ndata['pos'] = self.forces
        self.ful_g.ndata['nodes'] = self.nidx

    def _build_nidx(self):
        self.nidx = torch.zeros(self.node_num).long()
        for i in range(self.node_num):
            self.nidx[i] = Molecule.NODE_TYPE[self.atoms[i]]
        return


def chunks(l, n):
    l_chunk = []
    chunk_size = int(len(l) / n)
    for i in range(0, n - 1):
        l_chunk.append(l[i * chunk_size:(i + 1) * chunk_size])
    l_chunk.append(l[(n - 1) * chunk_size:])
    return l_chunk


class MoleDataset(Dataset):
    def __init__(self,
                 path=None,
                 mols=None,
                 datas=None,
                 prop_name='homo',
                 loc=False,
                 glob=True):
        self.path = path
        self.loc_g = loc
        self.glob_g = glob
        self.prop_name = prop_name
        self.datas = datas
        self.mols = mols
        if self.path:
            # data element ( pos, atoms, edges, smi, props, dists)
            self.datas = pickle.load(open(path, 'rb'))
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
            self.prop = torch.Tensor(
                [mol.props[self.prop_name] for mol in self.mols])
            self.mean = torch.mean(self.prop)
            self.std = torch.std(self.prop)
        elif self.datas:
            self.prop_keys = self.datas[0][4].keys()
            self.props = {}
            for key in self.prop_keys:
                self.props[key] = torch.Tensor(
                    [data[4][key] for data in self.datas])
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
            self.mols[i] = Molecule(pos,
                                    atoms,
                                    edges,
                                    smi,
                                    prop,
                                    distance=dists,
                                    loc=self.loc_g,
                                    glob=self.glob_g)
            print(i)
        return

    def get_props(self):
        return self.prop

    def load_mol(self, paths):
        self.mols = []
        if type(paths) is list:
            for path in paths:
                self.mols.extend(pickle.load(open(path, 'rb')))
                print('loading...')
        else:
            self.mols = pickle.load(open(paths, 'rb'))
        self.__get_props()

    def save_mol(self, paths):
        if type(paths) is list:
            path_num = len(paths)
            mols_chunks = chunks(self.mols, path_num)
            for i in range(path_num):
                pickle.dump(mols_chunks[i], open(paths[i], 'wb'))
        else:
            pickle.dump(self.mols, open(paths, 'wb'))
        return


class SelfMolDataSet(Dataset):
    def __init__(self, mols=None, level='n', prop_name='homo'):
        super(Dataset, self).__init__()
        self.level = level
        self.mols = mols
        self.data_num = len(mols)
        self.n_ids = [mol.ful_g.ndata['nodes'] for mol in self.mols]
        self.prop = torch.Tensor([mol.props[prop_name] for mol in self.mols])
        self.mean = torch.mean(self.prop)
        self.std = torch.std(self.prop)

        if level not in ['n', 'g', 'w']:
            raise ValueError

    def get_level(self):
        return self.level

    def __len__(self):
        return self.data_num

    def __getitem__(self, item):
        if (self.level == 'n'):
            return self.mols[item], self.n_ids[item]
        elif (self.level == 'g') or (self.level == 'w'):
            return self.mols[item], self.n_ids[item], item

        else:
            raise ValueError


# deal with large molecules, load mols from seperate .pkl
class FMolDataSet(Dataset):
    def __init__(self, dir, prop_name='homo', data_ids=None):
        self.dir = dir
        self.prop_name = prop_name
        self.data_ids = np.sort(
            data_ids) if data_ids is not None else np.arange(
                len(os.listdir(dir)) - 1)
        self.data_num = len(self.data_ids)
        self.__get_props()

    def __len__(self):
        return self.data_num

    def __getitem__(self, item):
        return pickle.load(
            open(
                self.dir + '/mol_{0:06d}'.format(self.data_ids[item]) + '.pkl',
                'rb')), self.prop[item]

    # data id must be increasing list
    def __get_props(self):
        props = pickle.load(open(self.dir + '/props.pkl', 'rb'))
        props = [props[i] for i in self.data_ids]
        self.prop = torch.Tensor([prop[self.prop_name] for prop in props])
        self.mean = torch.mean(self.prop)
        self.std = torch.std(self.prop)


def get_dataset_from_files(paths, prop_name):
    if type(paths) is list:
        props = []
        mols = []
        for path in paths:
            mols_, props_ = pickle.load(open(path, 'rb'))
            mols.extend(mols_), props.append(props_)
        props = np.concatenate(props, axis=0)
        return MoleDataset(mols, torch.Tensor(props), prop_name)
    else:
        mols, props = pickle.load(open(paths, 'rb'))
        return MoleDataset(mols, torch.Tensor(props), prop_name)


def load_dataset(path, prop_name, loc_g=False, glob_g=True):
    datas = pickle.load(open(path, 'rb'))
    print('load success, preprocessing...')
    mols = []
    props = []
    i = 0
    for data in datas:
        pos, atoms, edges, smi, prop, dists = data
        mols.append(
            Molecule(pos,
                     atoms,
                     edges,
                     smi,
                     prop,
                     distance=dists,
                     loc=False,
                     glob=True))
        props.append(prop)
        print(i)
        i += 1
    props = np.stack(props, axis=0)
    return MoleDataset(mols, torch.Tensor(props), prop_name)


# a list of tuple to tuple of list
def batcher(input):
    mols, props = zip(*input)
    props = torch.stack(props, dim=0)
    return mols, props


def batcher_n(input):
    mols, nodes = zip(*input)
    nodes = torch.cat(nodes, dim=0)
    return mols, nodes


def batcher_g(input):
    mols, nodes, ids = zip(*input)
    nodes = torch.cat(nodes, dim=0)
    return mols, nodes, ids


# dataparallel support set_mean and std for model
class RefDataParallel(DataParallel):
    def set_mean_std(self, mean, std):
        return getattr(self.module, 'set_mean_std')(mean, std)


# c_ij = \sum_k (a_ik - b_jk)^2,torch Tensor
def pairwise_L2(A, B):
    A_norm2 = torch.sum(A**2,
                        dim=1).unsqueeze(1).expand([A.shape[0], B.shape[0]])
    B_norm2 = torch.sum(B**2,
                        dim=1).unsqueeze(0).expand([A.shape[0], B.shape[0]])
    C = A_norm2 + B_norm2 - 2 * A @ B.t()
    return C


# torch tensor data_num*embed_dim, calculate by L2 distance
# show stats: calculate mean MSE (mean on L2 and clusters)
def k_medoid(embeddings,
             cluster_num,
             iters,
             center_ids=None,
             show_stats=False):
    # init
    if center_ids is None:
        center_ids = random.sample(range(embeddings.shape[0]), cluster_num)
    cluster_centers = embeddings[center_ids]
    for iteration in range(iters):

        distances = pairwise_L2(embeddings, cluster_centers)
        data_tags = torch.argmin(distances, dim=1)
        n_count = torch.ones_like(embeddings).float()
        # see: https://stackoverflow.com/questions/58007127/pytorch-differentiable-conditional-index-based-sum
        n_centers = torch.zeros([data_tags.max() + 1, embeddings.shape[1]])
        avg_centers = torch.zeros([data_tags.max() + 1, embeddings.shape[1]])
        n_centers.index_add_(0, data_tags, n_count)
        avg_centers.index_add_(0, data_tags, embeddings)
        avg_centers = avg_centers / n_centers
        cls_ids = [[] for _ in range(cluster_num)
                   ]  # record the data ids of each cluster

        [
            cls_ids[int(data_tags[i])].append(i)
            for i in range(embeddings.shape[0])
        ]
        for i in range(cluster_num):
            if len(cls_ids[i]):
                center_ids[i] = cls_ids[i][int(
                    torch.argmin(
                        torch.sum((embeddings[cls_ids[i]] - avg_centers[i])**2,
                                  dim=1)))]
        # update cluster centers
        cluster_centers = embeddings[center_ids]

        if show_stats:
            E = 0
            for i in range(cluster_num):
                E += torch.sum(
                    torch.mean((embeddings[cls_ids[i]] - avg_centers[i])**2,
                               dim=1))
            E = E / cluster_num
            print('Iteration {} Mean MSE {}'.format(iteration + 1, E))
    return center_ids


def k_center(embeddings, cluster_num):
    time0 = time.time()
    center_ids = []
    center_ids.append(random.choice(range(embeddings.shape[0])))
    cluster_center = embeddings[center_ids[-1]].unsqueeze(0)
    # distances = 1e9 * torch.ones(embeddings.shape[0])
    min_dist = 1e9 * torch.ones(embeddings.shape[0])
    for k in range(cluster_num - 1):
        # anchor_ids = random.sample(range(embeddings.shape[0]),embeddings.shape[0]//10)
        distances = torch.sum((embeddings - cluster_center)**2, dim=1)
        min_dist = torch.min(torch.stack([min_dist, distances], dim=0),
                             dim=0)[0]
        center_ids.append(int(torch.argmax(min_dist)))
        cluster_center = embeddings[center_ids[-1]]
    print('k_center finished {}'.format(time.time() - time0))
    return center_ids


# see: https://stackoverflow.com/questions/58007127/pytorch-differentiable-conditional-index-based-sum
#clear up clusters
def get_centers(embeddings, data_tags, cluster_num):
    val_cls_num = data_tags.max() + 1
    cluster_centers = torch.zeros([cluster_num, embeddings.shape[1]])
    n_count = torch.ones_like(embeddings).float()
    n_centers = torch.zeros([data_tags.max() + 1,
                             embeddings.shape[1]]).to(embeddings.device)
    avg_centers = torch.zeros([data_tags.max() + 1,
                               embeddings.shape[1]]).to(embeddings.device)
    n_centers.index_add_(0, data_tags, n_count)
    avg_centers.index_add_(0, data_tags, embeddings)
    avg_centers = avg_centers / n_centers
    cluster_centers[:val_cls_num] = avg_centers
    cls_ids = [[] for _ in range(cluster_num)
               ]  # record the data ids of each cluster
    [cls_ids[int(data_tags[i])].append(i) for i in range(data_tags.shape[0])]

    # reassign empty clusters
    ept_cls_num = 0
    for i in range(cluster_num):
        if len(cls_ids[i]) == 0:
            cls_ids[i] = random.choice(range(embeddings.shape[0]))
            cluster_centers[i] = embeddings[cls_ids[i]]
            ept_cls_num += 1

    if ept_cls_num > 0:
        print('{} empty cluster detected'.format(ept_cls_num))
    return cluster_centers, cls_ids


# return the id of each data
def k_means(embeddings, cluster_num, iters, inits='random', show_stats=True):
    if inits is 'random':
        center_ids = random.sample(range(embeddings.shape[0]), cluster_num)
        cluster_centers = embeddings[center_ids]
        data_tags = 0

    elif inits is 'k_center':
        center_ids = k_center(embeddings, cluster_num)
        cluster_centers = embeddings[center_ids]
        data_tags = 0
        print('seed initialization finished')

    else:
        data_tags = inits
        cluster_centers, cls_ids = get_centers(embeddings, data_tags,
                                               cluster_num)

    for iteration in range(iters):

        distances = pairwise_L2(embeddings, cluster_centers)
        data_tags = torch.argmin(distances, dim=1)
        cluster_centers, cls_ids = get_centers(embeddings, data_tags,
                                               cluster_num)

        if show_stats:
            E = 0
            for i in range(cluster_num):
                try:
                    E += torch.sum(
                        torch.mean((embeddings[cls_ids[i]].view(
                            -1, embeddings.size(1)) - cluster_centers[i])**2,
                                   dim=1))
                except Exception:
                    pass
            E = E / cluster_num
            print('Iteration {} Mean MSE {}'.format(iteration + 1, E))

    return data_tags


def k_medoids_pp(embeddings, cluster_num, iters, show_stats=True):
    # seed initialization
    time0 = time.time()

    center_ids = k_center(embeddings, cluster_num)
    if show_stats:
        print(time.time() - time0)
        print('seed initialization finished')
    cluster_centers = embeddings[center_ids]
    for iteration in range(iters):
        distances = pairwise_L2(embeddings, cluster_centers)
        data_tags = torch.argmin(distances, dim=1)
        n_count = torch.ones_like(embeddings).float()
        # see: https://stackoverflow.com/questions/58007127/pytorch-differentiable-conditional-index-based-sum
        n_centers = torch.zeros([data_tags.max() + 1, embeddings.shape[1]])
        avg_centers = torch.zeros([data_tags.max() + 1, embeddings.shape[1]])
        n_centers.index_add_(0, data_tags, n_count)
        avg_centers.index_add_(0, data_tags, embeddings)
        avg_centers = avg_centers / n_centers
        cls_ids = [[] for _ in range(cluster_num)
                   ]  # record the data ids of each cluster

        [
            cls_ids[int(data_tags[i])].append(i)
            for i in range(embeddings.shape[0])
        ]
        for i in range(cluster_num):
            if len(cls_ids[i]):
                center_ids[i] = cls_ids[i][int(
                    torch.argmin(
                        torch.sum((embeddings[cls_ids[i]] - avg_centers[i])**2,
                                  dim=1)))]
        # update cluster centers
        cluster_centers = embeddings[center_ids]

        if show_stats:
            E = 0
            for i in range(cluster_num):
                E += torch.sum(
                    torch.mean((embeddings[cls_ids[i]] - avg_centers[i])**2,
                               dim=1))
            E = E / cluster_num
            print('Iteration {} Mean MSE {}'.format(iteration + 1, E))
    return center_ids


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
    def __init__(self, data, label):
        '''获取并划分数据'''
        self.data = data
        self.label = label
        self.num = self.data.shape[0]

    def __getitem__(self, index):
        data = torch.Tensor(self.data[index])
        label = torch.Tensor(self.label[index])
        return data, label

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
        a = Chem.Atom(atomic_nums[node])
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


class AccMeter():
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.correct_num = 0.0
        self.total_num = 0
        self.iters = 0

    def add(self, preds, targets):
        self.correct_num += torch.sum(preds.eq(targets))
        self.total_num += preds.shape[0]
        self.iters += 1

    def reset(self):
        self.correct_num = 0.0
        self.total_num = 0.0
        self.iters = 0

    def value(self):
        return float(self.correct_num) / (self.total_num + 1e-10)

# avg_centers = get_avg_centers(data_tags, embeddings)
# val_cls_num = data_tags.max() + 1
# cls_ids = [[] for _ in range(cluster_num)]  # record the data ids of each cluster
# [cls_ids[int(data_tags[i])].append(i) for i in range(embeddings.shape[0])]
# cluster_centers = avg_centers
#
# # avoid empty clusters
# ept_cls_num = 0
# for i in range(cluster_num):
#     if len(cls_ids[i]) == 0:
#         cls_ids[i] = random.choice(range(embeddings.shape[0]))
#         try:
#             cluster_centers[i] = embeddings[cls_ids[i]]
#         except Exception:
#             print("errrrrrrrrror")
#         ept_cls_num += 1
#
# if ept_cls_num > 0:
#     print('{} empty cluster detected'.format(ept_cls_num))


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
