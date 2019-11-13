import torch
import numpy as np
import time

class Bayes_sampler(object):
    def __init__(self,args,total_data_num,batch_data_num,init_ids):
        self.args = args
        self.total_data_num = total_data_num
        self.batch_data_num = batch_data_num
        self.data_ids = np.delete(np.arange(self.total_data_num,dtype=int),init_ids)  #data unselected


    # query by TODO: the calculation of variance need to be checked
    def query(self,preds):
        time0 = time.time()
        variance = np.std(preds, axis=0).squeeze()
        vars_ids = np.stack([variance, np.arange(0, len(variance))], axis=0)
        queries = vars_ids[:, vars_ids[0].argsort()]
        query_ids = queries[1].astype(int)[-self.batch_data_num:]  # query id according to new dataset
        # query_data_ids = queries[2].astype(int)[-self.batchnum:]       #query id according to origin dataset(whole)
        query_data_ids = self.data_ids[query_ids]
        self.data_ids = np.delete(self.data_ids, query_ids)  # del from unlabeled
        print('query new data  {}'.format(time.time() - time0))
        return query_data_ids

