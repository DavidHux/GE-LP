from u import Preprocess, spa
from gemodels import gemodel_GCN

import warnings
import os
import copy
import time

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

dataset = 'cora'
percent = 0.5

def run_time(func, *args, **kw):
    start_time = time.time()
    func(*args, **kw)
    end_time = time.time()
    print('run time: {}s'.format(end_time-start_time))

if __name__ == "__main__":
    _A_obs, feas, labels = Preprocess.loaddata(
        'data/{}.npz'.format(dataset), llc=False)
    _A_prev = _A_obs
    adj, remained, deleted = spa.delete_edges(_A_obs, k=percent)

    _N = _A_prev.shape[0]
    # split_train, split_val, split_unlabeled = Preprocess.splitdata(_N, labels) #seed share as default
    # split_t = (split_train, split_val, split_unlabeled)
    # gcn = gemodel_GCN(_A_prev, feas, labels, split_t=split_t, seed=1, dropout=0)
    # run_time(gcn.train)
    # print('performance: {}, acu: {}'.format(gcn.performance(), gcn.acu()))

    # gcn = gemodel_GCN(adj, feas, labels, split_t=split_t, seed=1, dropout=0)
    # gcn.train()
    # print('performance: {}, acu: {}'.format(gcn.performance(), gcn.acu()))

    split_train, split_val, split_unlabeled = Preprocess.splitdata(_N, labels, seed=12, share=(0.052, 0.3693)) #seed share as default
    split_t = (split_train, split_val, split_unlabeled)
    gcn = gemodel_GCN(_A_prev, feas, labels, split_t=split_t, seed=1, dropout=0)
    run_time(gcn.train)
    print('performance: {}, acu: {}'.format(gcn.performance(), gcn.acu()))

    gcn = gemodel_GCN(_A_prev, feas, labels, split_t=split_t, seed=1, dropout=0, sGCN=True)
    run_time(gcn.train)
    print('performance: {}, acu: {}'.format(gcn.performance(), gcn.acu()))

    gcn = gemodel_GCN(adj, feas, labels, split_t=split_t, seed=1, dropout=0)
    gcn.train()
    print('performance: {}, acu: {}'.format(gcn.performance(), gcn.acu()))

    
    # print('preprocess, delete some edges, remaind edges num(bi): {}'.format(adj.nnz))

    # savefile = 'data/coradele.npz'
    # Preprocess.savedata(savefile, adj, feas, labels)
    # a, f, l = Preprocess.loaddata(savefile, llc=False)
    # print('reloadfile, edges num:{}, shape: {}, feas: {}'.format(a.nnz, a.shape, f.nnz))