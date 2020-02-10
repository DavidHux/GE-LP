#coding=utf-8

import copy
import time

from addEdges import *
from candidateGenerate import *
from edgeEval import *
from edgesUpdation import *
from gemodels import *
from u import Preprocess

''' 迭代式边重构方法
'''


class IALGE():
    '''
    '''
    ''' need to be set
    gemodel = None
    edge_strategy = None
    '''

    def __init__(self, adj, features, labels, tao, n, s, gemodel='GCN', cangen='knn', edgeEval='max', edgeUpdate='easy', early_stop=20, seed=-1, dropout=0.5, deleted_edges=None, initadj=None, params=None, dataset=('cora', 1), testindex=1, split_share=(0.1, 0.1)):
        '''
        args:
            adj: init adj matrix, N*N
            feature: N*D
            tao: iter times
            n: candidate patch size
            s: one patch size
            params: (edgenumPit2add, cannumPit, knn, subsetnum) e2a, cand, knn, se
        '''
        self.adj = adj
        self.features = features
        self.tao = tao
        self.n = n
        self.s = s
        self.labels = labels
        self.early_stop = early_stop
        self.seed = seed
        self.deleted_edges = deleted_edges
        self.dropout = dropout
        self.split_share = split_share

        if params == None:
            self.params = (20, 20, 20, 20, 5)
        else:
            self.params = params
        
        self.edgenumPit2add, self.seedEdgeNum, self.knn, self.subsetnum, self.evalPerEdge = self.params
        self.poolnum = 20
        
        print('iterAddlinks: params:{} start'.format(self.params))
        timenow = time.asctime(time.localtime(time.time()))

        self.taskname = 'ial_res_{}_{}_{}_{}_{}'.format(dataset, edgeEval, self.params, testindex, timenow)

        self.outfile = open('{}.txt'.format(self.taskname), 'w')

        _N = self.adj.shape[0]
        self.split_train, self.split_val, self.split_unlabeled = Preprocess.splitdata(_N, self.labels, share=self.split_share)
        self.split_t = (self.split_train, self.split_val, self.split_unlabeled)

        if initadj !=  None:
            e, p = self.test(initadj)
            self.output('complete adj performance: {}'.format(p), f=True)

        # if gemodel == None:
        #     self.gemodel = model_i()
        # elif gemodel == 'GCN':
        #     # self.gemodel = gemodel_GCN(self.adj, self.features, self.labels, seed=self.seed, dropout=0)
        #     self.gemodel = None
        # else:
        #     print('ERR: wrong graph embedding class')
        #     exit(-1)

        if cangen == 'knn':
            self.cangen = canGen_knn(self.seedEdgeNum, self.poolnum, self.knn)
        else:
            self.output('cangen params err')
            exit(0)
        
        if edgeEval == 'max':
            self.edgeEval = edgeEval_max(self.adj, self.features, self.labels, self.split_t, self.poolnum, self.knn, self.evalPerEdge, seed=self.seed, dropout=self.dropout)
        else:
            self.output('edgeeval params err')
            exit(0)

        if edgeUpdate == 'easy':
            self.edgeUpdate = edgesUpdate_easy(self.adj, self.features, self.labels, self.split_t, self.edgenumPit2add, self.poolnum, self.subsetnum, self.seed, self.dropout)
        else:
            self.output('edgeUpdation params err')
            exit(0)

        
    # def setgemodel(self, model):
    #     if not isinstance(model, gemodel):
    #         print('model(type: {}) to set is not instance of {}'.format(
    #             type(model), type(gemodel)))
    #         exit(0)

    #     self.gemodel = model
    #     self.gemodel.setfeature(self.features)
    #     self.gemodel.setAdj(self.adj)

    
    def output(self, s, p=True, f=False):
        if p:
            print(s)
        if f:
            self.outfile.write(s+'\n')
            self.outfile.flush()

    
    def run(self):
        A_ = self.adj
        best_adj = copy.deepcopy(A_)
        adde = 0
        Preprocess.savedata('{}_initadj.npz'.format(self.taskname), self.adj, self.features, self.labels)

        embds1, per1 = self.test(self.adj)
        init_performance = best_performance = per1
        self.output('initial performace: {}, initial edges: {}'.format(init_performance, A_.nnz), f=True)

        early_it = 0
        for i in range(self.tao):
            embd_, _ = self.test(A_)
            cans = self.cangen.cans(A_, embd_)
            edges_p = self.edgeEval.eval(cans, A_)
            new_adj, new_perf, addededgenum = self.edgeUpdate.update(edges_p)

            self.output('time: {}, it: {}, performance: {}, init:{}, best:{}, added {} edges'.format(time.asctime(time.localtime(time.time())), i, new_perf, init_performance, best_performance, addededgenum), f=True)

            if new_perf < best_performance:
                early_it += 1
                if early_it >= self.early_stop:
                    self.output('\nearly stop at it: {}, performance: {}, init: {}\n'.format(i, best_performance, init_performance), f=True)
                    Preprocess.savedata('{}_finaladj.npz'.format(self.taskname), best_adj, self.features, self.labels)
                    break
            else:
                best_performance = new_perf
                best_adj = new_adj
                adde = addededgenum
                early_it = 0
                A_ = new_adj

        self.output('final performace: {}, added {} edges'.format(best_performance, adde), f=True)
        return best_adj, best_performance
        
    
    def test(self, adj):
        embds, per = Modeltest_GCN.subprocess_GCN(adj, self.features, self.labels, split_t=(self.split_train, self.split_val, self.split_unlabeled), seed=self.seed, dropout=self.dropout)
        return embds, per
