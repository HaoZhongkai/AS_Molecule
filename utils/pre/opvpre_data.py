import csv
import numpy as np
from copy import deepcopy
from utils.funcs import mol2nx, Molecule
import pickle
from rdkit import Chem
import os
import multiprocessing as mp
import networkx as nx
import sys
from config import Global_Config as Config
config = Config()


# for idx, row in enumerate(lines):
def process_data(row):
    # record atom
    info = row[0].split('\n')
    atom_num0, edge_num = int(info[3].split()[0]), int(info[3].split()[1])
    atom_num = 0
    # edgenum be 0 then do
    if edge_num is 0:
        atom_num = int(str(atom_num0)[:-3])
        edge_num = int(str(atom_num0)[-3:])
    else:
        atom_num = atom_num0
    pos, edges, atoms = np.zeros([atom_num, 3]), np.zeros([edge_num, 3],
                                                          dtype=int), []
    at_index, ed_index = range(4, 4 + atom_num + 1), range(
        4 + atom_num, 4 + atom_num + edge_num)  # there is a M END at last line
    for j in range(atom_num):
        # try:
        pos[j] = np.asarray(info[at_index[j]].split()[:3], dtype=float)
        atoms.append(info[at_index[j]].split()[3])
        # except Exception:
        #     print('error')
    for j in range(edge_num):
        edge_str = info[ed_index[j]].split()
        if len(edge_str) == 6:
            if len(edge_str[0]) == 5 or len(edge_str[0]) == 6 or len(
                    edge_str[0]) == 4:
                edge_start = int(edge_str[0][:-3]) - 1
                edge_end = int(edge_str[0][-3:]) - 1
                edge_type = int(edge_str[1]) - 1
            else:
                raise ValueError
            edges[j] = np.array([edge_start, edge_end, edge_type], dtype=int)
        elif len(edge_str) == 7:
            edges[j] = np.asarray(info[ed_index[j]].split()[:3],
                                  dtype=int) - 1  # notice -1 ***
        else:
            raise ValueError
    # try:
    props = np.asarray(row[4:4 + 5], dtype=float)
    # except Exception:
    #     props = 0
    #     print('error')
    # props_all.append(props)
    smi = row[-1]

    # mole = Molecule(pos, atoms, edges, smi, props)
    # mols.append(mole)
    # print(' atom_num {} edge_num {}'.format( atom_num, edge_num))
    return (pos, atoms, edges, smi, props)
    # return (mole, props)


def dist(pos):
    node_num = pos.shape[0]
    distance_matrix = np.zeros(node_num * node_num)
    for i in range(node_num):
        distance_matrix[i * node_num:(i + 1) * node_num] = np.linalg.norm(
            pos[i] - pos, ord=2, axis=1)
    return distance_matrix


# save elements [(pos, atoms, edges, smi, props, dist, RWMol, nx_graph)]
def get_elem(row):
    pos, atoms, edges, smi, props = process_data(row)
    dists = dist(pos)
    # try:
    #     mol = Chem.RWMol()
    #     for i in range(len(atoms)):
    #         mol.AddAtom(Chem.rdchem.Atom(atoms[i]))
    #     for i in range(len(edges)):
    #         mol.AddBond(int(edges[i][0]), int(edges[i][1]), EDGE_TYPE[edges[i][2]])
    # except Exception:
    #     nx_g = None
    #     print('error')

    return pos, atoms, edges, smi, props, dists


def preprocess_all(path):
    # data pos: nparr num*3 edge nparr int enum*3 (first second type) list str props nparr num*props
    with open(path) as datafile:
        csvreader = csv.reader(datafile)
        lines = list(csvreader)[1:]
        mol_num = len(lines)
        mols = []
        props_all = []
        datas = []
        distances = []
        pool = mp.Pool(processes=12)
        cnt = 0

        # pool.map(process_data, lines)

        for _ in pool.imap(process_data, lines):
            sys.stdout.write('id:{}\n'.format(cnt))
            cnt += 1
            datas.append(_)

        poses = [datas[i][0] for i in range(mol_num)]
        cnt = 0
        for _ in pool.imap(dist, poses):
            sys.stdout.write('id:{}\n'.format(cnt))
            distances.append(_)
            cnt += 1

        for i in range(mol_num):
            mols.append(Molecule(*datas[i], distances[i]))
            props_all.append(datas[i][-1])
            print(i)

        props_all = np.stack(props_all, axis=0)
        return mols, props_all


def save_elements():
    path = config.PATH + '/datasets/OPV/mol_train.csv'
    # path = config.PATH+'/datasets/OPV/mol_valid.csv'
    # path = config.PATH+'/datasets/OPV/mol_test.csv'
    save_path = config.PATH + '/datasets/OPV/data_elem_train.pkl'
    # save_path = config.PATH+'/datasets/OPV/data_elem_valid.pkl'
    # save_path = config.PATH+'/datasets/OPV/data_elem_test.pkl'
    with open(path) as datafile:
        csvreader = csv.reader(datafile)
        lines = list(csvreader)[1:]
        mol_num = len(lines)
        pool = mp.Pool(processes=15)
        cnt = 0

        data_l = []

        for _ in pool.imap(get_elem, lines):
            sys.stdout.write('id:{}\n'.format(cnt))
            data_l.append(_)
            cnt += 1

        pickle.dump(data_l, open(save_path, 'wb'))
        print('okokok')


def save_all():
    train_path = config.PATH + '/datasets/OPV/mol_train.csv'
    test_path = config.PATH + '/datasets/OPV/mol_test.csv'
    valid_path = config.PATH + '/datasets/OPV/mol_valid.csv'
    save_path_train = config.PATH + '/datasets/OPV/' + 'OPV_dataset_train_3.pkl'
    save_path_test = config.PATH + '/datasets/OPV/' + 'OPV_dataset_test.pkl'
    save_path_valid = config.PATH + '/datasets/OPV/' + 'OPV_dataset_valid.pkl'
    # path = 'D:/0Lab/ML/repos/AL/dataset/mol_valid.csv'
    # train_mols, train_props = preprocess_all(train_path)
    test_mols, test_props = preprocess_all(test_path)
    # valid_mols, valid_props = preprocess_all(valid_path)

    # pickle.dump((train_mols,train_props),open(save_path_train,'wb'))
    pickle.dump((test_mols, test_props), open(save_path_test, 'wb'))
    # pickle.dump((valid_mols,valid_props),open(save_path_valid,'wb'))
    print('okokokokok')


if __name__ == '__main__':

    save_elements()
