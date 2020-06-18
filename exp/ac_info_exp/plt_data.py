#!/usr/bin/env python
#-*- coding:utf-8 _*-

import matplotlib.pyplot as plt
import numpy as np

init_data_num = 5000
total_data_num = 113800

file_lists = [
    'qm9_bayes_by_valid_1114_22_31.txt',
    'qm9_k_center_by_valid_1114_22_12.txt',
    'qm9_random_by_valid_1115_13_02.txt',
    'qm9_k_center_by_valid_1115_09_07.txt',

    # 'qm9_random_fixed_epochs_1119_16_35.txt',
    # 'qm9_k_center_fixed_epochs_1119_17_10.txt',
    # 'qm9_msg_mask_fixed_epochs_1118_11_07.txt',

    # 'qm9_bayes_fixed_epochs_1118_11_39.txt',
    # 'qm9_msg_mask_by_valid_1119_15_41.txt',
    # 'qm9_random_fixed_epochs_1121_22_35.txt'
]

datas = []
for file in file_lists:
    with open(file, 'r') as fp:
        lines = fp.readlines()
        test_mae = []
        for line in lines:
            test_mae.append(float(line.split('\t')[4]))
        label_nums = np.linspace(init_data_num, total_data_num, len(test_mae))
        datas.append((label_nums, np.array(test_mae)))

plt.figure()
for i in range(len(file_lists)):
    color = np.random.rand(3)
    plt.plot(datas[i][0], datas[i][1], color=color, label=file_lists[i])
plt.legend()
plt.ylim([0, 0.01])
plt.show()
