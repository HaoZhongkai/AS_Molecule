import numpy as np
import torch
import multiprocessing as mp
import time
from utils.funcs import *
import random
import math
class K_center(object):
    def __init__(self,args,total_data_num,batch_data_num,init_ids):
        self.args = args
        self.total_data_num = total_data_num
        self.batch_data_num = batch_data_num
        self.data_ids = np.delete(np.arange(self.total_data_num,dtype=int),init_ids)  #data unselected
        self.core_ids = init_ids

    # metric: function calculates distance of embeddings
    #embeddings: np n*ebd
    def query(self,embeddings,process_num=10):
        time0 = time.time()


        new_batch = []
        # pool = mp.Pool(process_num)
        for id in range(self.batch_data_num):
            un_embeddings = embeddings[self.data_ids]
            core_embeddings = embeddings[self.core_ids]
            minimal_cover_dist = torch.zeros(len(self.data_ids)).to(un_embeddings.device)
            chunk_ids = chunks(range(un_embeddings.size(0)), int(math.sqrt(un_embeddings.size(0))))
            un_ebd_a = torch.sum(un_embeddings**2,dim=1)
            c_ebd_b = torch.sum(core_embeddings**2,dim=1)
            for i in range(len(chunk_ids)):
                # minimal_cover_dist[i] = torch.min(torch.norm(un_embeddings[i]-core_embeddings,p=2,dim=1,keepdim=False))
                minimal_cover_dist[chunk_ids[i]] = torch.min(c_ebd_b-2*un_embeddings[chunk_ids[i]]@core_embeddings.t(),dim=1)[0]

            core_point_id = torch.argmax(minimal_cover_dist+un_ebd_a).cpu().numpy()    #id in data_ids
            new_batch.append(self.data_ids[core_point_id])
            self.data_ids = np.delete(self.data_ids,core_point_id)
            # print(id)
        self.core_ids = np.sort(np.concatenate([self.core_ids,new_batch]))
        print('query new data {}'.format(time.time()-time0))
        return new_batch


    def random_query(self):
        new_batch_ids = np.sort(random.sample(self.data_ids,self.batch_data_num))
        self.data_ids = np.delete(self.data_ids,new_batch_ids)
        return new_batch_ids












# @numba.jit(nopython=True)
# def add(l:np.ndarray,m:np.ndarray):
#     for i in range(l.shape[0]):
#         q = np.zeros(m.shape[0])
#         for j in range(m.shape[0]):
#             q[j] = np.sqrt(np.dot(l[i]-m[j],l[i]-m[j]))
#         # np.min(np.sqrt(np.einsum('ij,ij->i',l[i] - m,l[i]-m)))
#     return
#
# @numba.jit()
# def add_t(l:torch.tensor,m:torch.tensor):
#     for i in range(l.shape[0]):
#         torch.min(torch.norm(l[i] - m, p=2, dim=1, keepdim=False))
#     return


# def mini_dist(u_embeddings,core_embeddings):
#     return np.min(np.linalg.norm(u_embeddings - core_embeddings,ord=2,axis=1,keepdims=False))
# test code
if __name__ == "__main__":
    sampler = K_center(None,110000,20,np.arange(10000,dtype=int))
    l = torch.Tensor(np.random.randn(110000,64)).cuda(0)
    # un_embeddings = torch.Tensor(np.random.randn(50000,64)).cuda(0)
    # core_embeddings = torch.Tensor(np.random.randn(50000,64)).cuda(0)

    # l = torch.Tensor(np.random.randn(100000,64))
    # m = torch.Tensor(np.random.randn(1000,64))

    # l = np.random.randn(100000,64)
    # m = np.random.randn(1000,64)
    # d = torch.zeros(100000).cuda(0)
    # q = np.zeros(100000)

    time0 = time.time()

    # chunk_ids = chunks(range(un_embeddings.size(0)),int(math.sqrt(un_embeddings.size(0))))
    #
    # for i in range(len(chunk_ids)):
    #     d[chunk_ids[i]] = torch.max(un_embeddings[chunk_ids[i]]@core_embeddings.t(),dim=1)[0]





    # for i in range(l.shape[0]):
    #     torch.min(torch.norm(l[i] - m, p=2, dim=1, keepdim=False))
        # np.min(np.linalg.norm(l[i]-m,ord=2,axis=1))

    # [torch.min(torch.norm(l[i]-m,p=2,dim=1,keepdim=False)) for i in range(l.shape[0])]

    # add(l,m)
    # add_t(l,m)




    #time 100000+1000 *64
    #                           GPU(torch)         CPU(numpy)       CPU(torch)
    # for  python               7.38                23.15              17.63
    # list compression
    # for  numba


    sampler.query(l)

    print(time.time() - time0)
