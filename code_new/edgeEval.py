import collections
import utils_nete as utils
from u import heap, Sim
from multiprocessing import Pool
from gemodels import gemodel_GCN, Modeltest_GCN
import multiprocessing
import random
import collections
import copy
import numpy as np

class edgeEval():
    pass


class edgeEval_max(edgeEval):
    def __init__(self, initadj, features, labels, split_t, poolnum, knn, edgeevaltimes, seed=-1, dropout=0):
        self.initadj = initadj
        self.features = features
        self.labels = labels
        self.poolnum = poolnum
        self.seed = seed
        self.dropout = dropout
        
        self.knn = knn
        self.evalEdgeNum = 20
        self.edgeevaltimes = edgeevaltimes

        self.split_train, self.split_val, self.split_unlabeled = split_t
        self.split_t = split_t

    def eval(self, candidates, adj):
        ''' eval edges' performanve in candidates set
        params:
            candidates: can edges set

        returns:
            p:
        '''

        p = Pool(self.poolnum)
        res_eval = []
        # psize = int(len(candidators) / poolnum) + 1
        for i in range(self.poolnum):
            r = p.apply_async(self.edgeEvaNew, args=(candidates, adj))
            res_eval.append(r)
        
        p.close()
        p.join()

        final_res = []

        evalres = collections.defaultdict(list)
        s = 0
        ret = []
        try:
            for x in res_eval:
                d = x.get()
                for y, v in d.items():
                    evalres[y].extend(v)

            '''max score as eval result'''
            for k, v in evalres.items():
                s += len(v)
                ret.append((k, max(v)))
        except BaseException as err:
            print('raised exception evalres: {}'.format(err))

        print('items: {}, totol items: {}'.format(len(evalres), s))

        # return sorted(ret, key=lambda x: x[1], reverse=True)
        return ret

    def edgeEvaNew(self, candidates, adj):
        res = collections.defaultdict(list)

        try:
            lencan = len(candidates)
            modelnum = int(2 * lencan* self.edgeevaltimes / self.poolnum / self.evalEdgeNum) + 1
            # print('eval model num: {}'.format(modelnum))
            for i in range(modelnum):
                addededge = []
                tempadj = copy.deepcopy(adj)
                for j in range(self.evalEdgeNum):
                    a, b = candidates[random.randint(0, lencan-1)]
                    tempadj[a, b] = 1
                    tempadj[b, a] = 1
                    addededge.append((a,b))

                g = gemodel_GCN(tempadj, self.features, self.labels, split_t=self.split_t, seed=self.seed, dropout=self.dropout)
                g.train()
                p1 = g.acu()
                
                for e in addededge:
                    res[e].append(p1)
        except BaseException as err:
            print('raised exception edgeEvaNew: {}'.format(err))

        return res