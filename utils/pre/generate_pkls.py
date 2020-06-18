import os
import pickle
from config import Global_Config

if __name__ == "__main__":
    config = Global_Config()
    path = config.train_pkl['qm9']
    print('start', path)
    mols = pickle.load(open(path, 'rb'))
    save_path = config.mols_dir['qm9']
    print(save_path)
    props = []
    for i in range(len(mols)):

        pickle.dump(
            mols[i],
            open(save_path + '/' + 'mol_{0:06d}'.format(i) + '.pkl', 'wb'))
        props.append(mols[i].props)
        print('{0:06d}'.format(i))

    pickle.dump(props, open(save_path + '/props.pkl', 'wb'))
    print('ok')
