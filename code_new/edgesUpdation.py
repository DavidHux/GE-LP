import collections
import copy
import random
from multiprocessing import Pool
from gemodels import Modeltest_GCN, gemodel_GCN

class edgesUpdate():

    def update(self, initadj, scoreset, prevupdation, k=0.1, minnum=20):
        ''' edge reconstruction interface
            args:
                prevAdj: adjacent matrix, N*N 
                embeddings: embedding matrix trained by ge models, N*D size
            
            return:
                newAdj: reconstructed adjacent matrix
                updation: updated edges set
        '''
        print('not completed edges reconstruction method')
        pass


class edgesUpdate_easy(edgesUpdate):
    def __init__(self, initadj, features, labels, split_t, edges2addNum, poolnum=1, subsetevalnum=20, seed=-1, dropout=0):
        self.seed = seed
        self.initadj = initadj
        self.features = features
        self.labels = labels
        self.split_train, self.split_val, self.split_unlabeled = split_t
        self.split_t = split_t
        self.dropout = dropout
        self.prevperformance = None
        self.edgeaddnum = edges2addNum
        self.currentAddNum = self.edgeaddnum
        # self.outf = outf
        self.scoreset = collections.defaultdict(float)
        self.it = 0
        self.topfactor = 3
        self.poolnum = poolnum
        self.subSetEvalNum = subsetevalnum

        _, self.prevperformance = self.test(self.initadj)
        self.initperformance = self.bestperformance = self.prevperformance


    def test(self, adj):
        embds, per = Modeltest_GCN.subprocess_GCN(adj, self.features, self.labels, split_t=(self.split_train, self.split_val, self.split_unlabeled), seed=self.seed, dropout=self.dropout)
        return embds, per

    def update(self, newscoreset):
        '''update adj matrix, add some good edges to form a new adj matrix, which will be used in new training round
        args:
            newscoreset: new learned edges: [((1, 2), 0.98), ((2,3), 0.97), ...]
        returns:
            es_adj, new adj matrix
            res_p, new performance
            currentAddNum, added edge num
        '''       
        self.it += 1
        
        '''union the learned result of edges in current around with all'''
        for k, v in newscoreset:
            self.scoreset[k] = max(self.scoreset[k], v)
        
        lllist = []
        for k, v in self.scoreset.items():
            lllist.append((k, v))
        
        lllist_sorted = sorted(lllist, key=lambda x: x[1], reverse=True)

        A_temp = copy.deepcopy(self.initadj)
        for a, b in lllist_sorted[:self.currentAddNum]:
            A_temp[a[0], a[1]] = 1
            A_temp[a[1], a[0]] = 1
        _, p = self.test(A_temp)

        # self.outf('it: {} update edges, from top {}, performance: {}, prev: {}'.format(self.it, self.currentAddNum, p, self.prevperformance))
        print('it: {} update edges, from top {}, performance: {}, prev: {}'.format(self.it, self.currentAddNum, p, self.prevperformance))

        enset, per_n = self.evalSetRandom(lllist_sorted[:self.currentAddNum * self.topfactor])
        # assert(len(enset) == self.currentAddNum)
        print('added set len: {}, current add num {}'.format(len(enset), self.currentAddNum))
        # self.outf('it: {} update edges, subeval Performance: {}, topset performance: {}, best p: {}'.format(self.it, per_n, p, self.bestperformance))
        print('it:{} updedges, subeval Perf: {}, topset perf: {}, best p: {}'.format(self.it, per_n, p, self.bestperformance))

        if per_n > p:
            res_p = per_n
            res_adj = copy.deepcopy(self.initadj)
            res_num = len(enset)
            for a, b in enset:
                res_adj[a, b] = 1
                res_adj[b, a] = 1
        else:
            res_p = p
            res_adj = A_temp
            res_num = self.currentAddNum
        
        if res_p > self.bestperformance:
            self.bestperformance = res_p
            self.currentAddNum += self.edgeaddnum

        return res_adj, res_p, res_num

    
    def evalSetRandom(self, topscoreset):
        '''eval random edge set from top score edges'''
        
        # sslist_ = sorted(listss, key=lambda x: x[+1], reverse=True)
        sslist = []
        for x in topscoreset:
            sslist.append(x[0])

        print('topscoreset edge[0]: {}, p[0]: {}, ss[0] {}'.format(topscoreset[0][0], topscoreset[0][1], sslist[0]))

        p=Pool(self.poolnum)
        res_subset = []
        for i in range(self.subSetEvalNum):
            r = p.apply_async(self.subseteval, args=(sslist,))
            res_subset.append(r)

        p.close()
        p.join()

        edgesets_eval = []
        for x in res_subset:
            a = x.get()
            edgesets_eval.append(a)
        
        eee = sorted(edgesets_eval, key=lambda x: x[1], reverse=True)
        print(eee[0][1], eee[1][1], eee[2][1])

        return eee[0]

    def subseteval(self, topset):
        '''eval edge set performance, randomly and some edges from top set
        '''
        # print('topset len: {}'.format(len(topset)))
        tempadj = copy.deepcopy(self.initadj)
        eset = set()
        for i in range(self.currentAddNum):
            ran = random.randint(0, len(topset)-1)
            eset.add(topset[ran])
            a, b = topset[ran]
            tempadj[a, b] = 1
            tempadj[b, a] = 1

        g = gemodel_GCN(tempadj, self.features, self.labels, split_t=self.split_t, seed=self.seed, dropout=self.dropout)
        g.train()
        return (eset, g.acu())