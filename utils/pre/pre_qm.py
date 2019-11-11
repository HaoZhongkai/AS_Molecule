# from utils.pre.sparse_molecular_dataset import SparseMolecularDataset
from config import Global_Config as Config
from utils.funcs import MoleDataset

config = Config()


if __name__ == '__main__':
    paths = {'train': config.DATASET_PATH['qm9'] + '/data_elem_train.pkl',
            'test': config.DATASET_PATH['qm9'] + '/data_elem_test.pkl',
            'valid': config.DATASET_PATH['qm9'] + '/data_elem_valid.pkl'}
    path_save = {'train':config.DATASET_PATH['qm9']+'/qm9_mol_train.pkl',
                # 'train':config.train_pkl['qm9'],
                 'test': config.DATASET_PATH['qm9'] + '/qm9_mol_test.pkl',
    }

    dataset_train = MoleDataset(paths['train'])
    dataset_test = MoleDataset(paths['test'])
    dataset_train.build()
    dataset_test.build()
    dataset_test.save_mol(path_save['train'])
    dataset_test.save_mol(path_save['test'])
    print('ok')