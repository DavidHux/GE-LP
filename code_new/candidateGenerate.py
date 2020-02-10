
import random
import numpy as np

from multiprocessing import Pool
from u import *
from sklearn.metrics.pairwise import cosine_similarity


class canGen():
    pass


class canGen_knn(canGen):
    def __init__(self, seedEdgeNum, poolnum, knn):
        self.seedEdgeNum = seedEdgeNum
        self.poolnum = poolnum
        self.knn = knn

    def cans(self, adj, embds):
        '''generate candidates with KNN method
        params:
            adj: the adj matrix in current iteration
        
        returns:
            edgelist: for example, [(1, 2), (2, 4)]

        '''
        
        edges_existed = edge2list.list(adj)
        p=Pool(self.poolnum)
        res_simedges = []
        for i in range(self.seedEdgeNum):
            ran = random.randint(0, len(edges_existed)-1)
            r = p.apply_async(self.getSimEdgesCOS, args=(ran, embds, edges_existed))
            res_simedges.append(r)

        p.close()
        p.join()

        candiset = set()
        for x in res_simedges:
            retedges = x.get()
            for e in retedges:
                if e[0] > e[1]:
                    e = (e[1], e[0])
                candiset.add(e)

        candidates = list(candiset)
        print('candidates len{}'.format(len(candiset)))
        return candidates
    

    def getSimEdgesCOS(self, edgenum, embds, edges_existed):
        e = edges_existed[edgenum]

        res1 = embds[e[0]].dot(embds.T)
        res2 = embds[e[1]].dot(embds.T)

        norm = np.linalg.norm(embds, axis=1)
        res1 = [res1[i]/(norm[e[0]] * norm[i]) for i in range(len(res1)) ]
        res2 = [res2[i]/(norm[e[1]] * norm[i]) for i in range(len(res2)) ]

        index1 = [(e[0], i) for i in range(len(embds))]
        index2 = [(e[1], i) for i in range(len(embds))]

        t1 = list(zip(res1, index1))
        t2 = list(zip(res2, index2))
        # t1.extend(t2)

        r1 = sorted(t1, key=lambda x:x[0], reverse=True)
        r2 = sorted(t2, key=lambda x:x[0], reverse=True)

        # print('get sim edges lp res:{}'.format(r[:10]))
        r = (r1, r2)
        edgeex = set(edges_existed)
        ret = []
        for rrr in r:
            count = 0
            for x in rrr:
                if x[1][0]==x[1][1] or x[1] in edgeex or (x[1][1], x[1][0]) in edgeex:
                    continue
                ret.append(x[1])
                count += 1
                if count == self.knn:
                    break

        return 

if __name__ == "__main__":
         
    def test():
        a = [[0.1, 0.1, 0.1, 0.1, 0.6],
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [0.3, 0.3, 0.2, 0.1, 0.1],
            [0.6, 0.1, 0.1, 0.1, 0.1],
            [0.9, 0.025, 0.025, 0.025, 0.025],
            [0.1, 0.1, 0.1, 0.1, 0.6]]
        
        embds = np.array(a)
        e = (0, 4)

        res1 = embds[e[0]].dot(embds.T)
        res2 = embds[e[1]].dot(embds.T)

        norm = np.linalg.norm(embds, axis=1)
        res1 = [res1[i]/(norm[e[0]] * norm[i]) for i in range(len(res1)) ]
        res2 = [res2[i]/(norm[e[1]] * norm[i]) for i in range(len(res2)) ]

        index1 = [(e[0], i) for i in range(len(embds))]
        index2 = [(e[1], i) for i in range(len(embds))]

        t1 = list(zip(res1, index1))
        t2 = list(zip(res2, index2))
        # t1.extend(t2)

        r1 = sorted(t1, key=lambda x:x[0], reverse=True)
        r2 = sorted(t2, key=lambda x:x[0], reverse=True)

        print(r1, r2)
    
    test()
