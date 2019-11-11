from torch.multiprocessing import Manager,Queue,Process
import numpy as np
def tar(share,i):
    share[i][0] = i

if __name__ == "__main__":
    manager = Manager()
    share = manager.list([np.zeros(4),np.zeros(4),np.zeros(4)])
    ps = []
    for i in range(3):
        p = Process(target=tar,args=(share,i))
        p.start()
        ps.append(p)

    for p in ps:
        p.join()

    print(share)
