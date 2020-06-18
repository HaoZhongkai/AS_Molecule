import numpy as np
import time
from multiprocessing import Pool
import torch
from geo_al.k_center import K_center

if __name__ == "__main__":
    l = torch.randn(50000, 320)
    g = torch.randn(50000, 320)
    l = l.cuda(0)
    g = g.cuda(0)
    m = torch.zeros(l.size(0))
    time0 = time.time()
    for i in range(l.size(0)):
        m[i] = torch.min(torch.norm(l[i] - g, p=2, dim=1))

    print('cost', time.time() - time0)
