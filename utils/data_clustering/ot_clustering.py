#!/usr/bin/env python
#-*- coding:utf-8 _*-

import ot
import numpy as np
import pickle
import matplotlib.pyplot as plt
import random
import time

from config import Global_Config, make_args
from pre_training.sch_embeddings import SchEmbedding
config = Global_Config()
args = make_args()

N = 1
d = 2
measures_locations = []
measures_weights = []

# for i in range(N):
#
#     n_i = np.random.randint(low=1, high=20)  # nb samples
#
#     mu_i = np.random.normal(0., 4., (d,))  # Gaussian mean
#
#     A_i = np.random.rand(d, d)
#     cov_i = np.dot(A_i, A_i.transpose())  # Gaussian covariance matrix
#
#     x_i = ot.datasets.make_2D_samples_gauss(n_i, mu_i, cov_i)  # Dirac locations    (nd array!)
#     b_i = np.random.uniform(0., 1., (n_i,))
#     b_i = b_i / np.sum(b_i)  # Dirac weights
#
#     measures_locations.append(x_i)
#     measures_weights.append(b_i)

data_path = config.DATASET_PATH['qm9'] + '/sch_ebd.pkl'
datas_pca = pickle.load(open(data_path, 'rb'))[:10000]  # O(n^2)
weights = np.ones(datas_pca.shape[0]) / datas_pca.shape[0]

measures_locations.append(datas_pca)
measures_weights.append(weights)

k = 4000  # number of Diracs of the barycenter
# X_init = np.random.normal(0., 1., (k, d))  # initial Dirac locations
init_ids = random.sample(range(datas_pca.shape[0]), k)
X_init = datas_pca[init_ids]
b = np.ones(
    (k, )
) / k  # weights of the barycenter (it will not be optimized, only the locations are optimized)

time0 = time.time()
print('start calculate barycenter')
X = ot.lp.free_support_barycenter(measures_locations, measures_weights, X_init,
                                  b)
print('time consume {}'.format(time.time() - time0))

plt.figure(1)
for (x_i, b_i) in zip(measures_locations, measures_weights):
    color = np.random.randint(low=1, high=10 * N)
    plt.scatter(x_i[:, 0], x_i[:, 1], marker='.', label='input measure')
plt.scatter(X[:, 0],
            X[:, 1],
            c='black',
            marker='.',
            label='2-Wasserstein barycenter')
plt.title('Data measures and their barycenter')
plt.legend(loc=0)
plt.savefig('ot_cls.png')
