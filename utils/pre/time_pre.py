import dgl
import networkx as nx
import numpy as np
import time
import pickle
# import torch.multiprocessing as _mp
from torch.multiprocessing import Pool, Manager, Process, Queue
import sys
import rdkit.Chem as Chem
import torch
from copy import deepcopy
# from utils.funcs import Molecule
torch.multiprocessing.set_sharing_strategy('file_system')


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

        self.coordinates = torch.Tensor(coords)
        self.atoms = atoms  # types corresponds to coordiantes
        self.node_num = len(atoms)
        self.edges = edges
        self.smi = smi

        # self.optical_lumo,self.gap,self.homo,self.lumo,self.spectral_overlap,self.delta_homo,self.delta_lumo,self.delta_optical_lumo,self.homo_extrapolated,self.lumo_extrapolated,self.gap_extrapolated,self.optical_lumo_extrapolated = props
        self.optical_lumo, self.gap, self.homo, self.lumo, self.spectral_overlap = props
        self.nx_g, self.nidx, self.loc_g, self.ful_g = nx.Graph(
        ), 0, dgl.DGLGraph(), dgl.DGLGraph()  #init

        self._build_nidx()
        if loc:
            self._build_loc()
        if glob:
            self._build_ful(distance)

    def _build_nx_g(self):
        # first build mol
        self.mol = Chem.RWMol()
        for i in range(len(self.atoms)):
            self.mol.AddAtom(Chem.rdchem.Atom(self.atoms[i]))
        for i in range(len(self.edges)):
            self.mol.AddBond(int(self.edges[i][0]), int(self.edges[i][1]),
                             Molecule.EDGE_TYPE[self.edges[i][2]])
        # self.nx_g = mol2nx(self.mol)
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


def get_mol(data):

    pos, atoms, edges, smi, prop, dists = data
    return Molecule(pos,
                    atoms,
                    edges,
                    smi,
                    prop,
                    distance=dists,
                    loc=False,
                    glob=False)
    # return pos, atoms, edges, smi, prop, dists


def get_mols(l, datas):
    l.extend(deepcopy([get_mol(data) for data in datas]))
    # l.put([get_mol(data) for data in datas])

    return


def bd_mol(mol):
    mol._build_ful()


# if __name__ == '__main__':
if __name__ == '__main__':

    num_atoms = 100
    gs = []
    t0 = time.time()
    '''
    for i in range(10000):
        g = dgl.DGLGraph()
        g.from_networkx(nx.complete_graph(num_atoms))
        # g.add_nodes(num_atoms)
        # g.add_edges(
        #     [i for i in range(num_atoms) for j in range(num_atoms)],
        #     [j for i in range(num_atoms) for j in range(num_atoms)])
        print(i)
        gs.append(g)
    '''
    path = '/media/sdc/seg3d/anaconda3/pkgs/libsnn/datasets/OPV/data_elem_train.pkl'
    datas = pickle.load(open(path, 'rb'))[1:]
    print('load success, preprocessing...')
    # mols = [[] for _ in range(len(datas))]
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
                     glob=False))
        props.append(prop)
        print(i)
        i += 1

    # mp = _mp.get_context('forkserver')
    # set_start_method('spawn')
    pool = Pool(processes=4)
    manager = Manager()
    # mols = manager.list([[] for _ in range(len(datas))])
    # mols = manager.list([])

    # pool.map(get_mol,datas,chunksize=100)

    i = 0
    for _ in pool.imap(bd_mol, mols, chunksize=100):
        print(i)
        # mols[i] = _
        i += 1

    # props = np.concatenate(props,axis=0)

    #process based
    # processes = []
    #
    # indexes = [range(i*3000,(i+1)*3000) for i in range(4)]
    # for i in range(4):
    #     p = Process(target=get_mols, args=(mols, [datas[j] for j  in indexes[i]]))
    #     p.start()
    #     processes.append(p)
    #
    # for i in range(4):
    #     processes[i].join()

    t1 = time.time()

    print(t1 - t0)
    print(0)
