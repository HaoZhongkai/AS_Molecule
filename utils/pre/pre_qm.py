# from utils.pre.sparse_molecular_dataset import SparseMolecularDataset
from config import Global_Config as Config
from utils.funcs import MoleDataset

config = Config()


if __name__ == '__main__':
    paths = {'train': config.DATASET_PATH['qm9'] + '/data_elem_train.pkl',
            'test': config.DATASET_PATH['qm9'] + '/data_elem_test.pkl',
            'valid': config.DATASET_PATH['qm9'] + '/data_elem_valid.pkl'}
    path_save = {'train':config.DATASET_PATH['qm9']+'/qm9_mol_train.pkl',
                  'valid': config.DATASET_PATH['qm9'] + '/qm9_mol_valid.pkl',
                 'test': config.DATASET_PATH['qm9'] + '/qm9_mol_test.pkl',
    }

    dataset_train = MoleDataset(paths['train'])
    dataset_train.build()
    dataset_train.save_mol(path_save['train'])
    print('ok')

    dataset_test = MoleDataset(paths['test'])
    dataset_test.build()
    dataset_test.save_mol(path_save['test'])
    print('ok')

    dataset_val = MoleDataset(paths['valid'])
    dataset_val.build()
    dataset_val.save_mol(path_save['valid'])
    print('ok')

