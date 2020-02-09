from itl import IALGE
from u import Preprocess, spa

import warnings
import os
import copy

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

dataset = 'cora'
percent = 0.5
share_cora = (0.052, 0.3693)

ps = [0.1, 0.25, 0.5, 0.75, 1]

edgenumPit2adds = [10, 20, 50, 100]
cannumpits = [20, 50, 100, 200, 500]
knns = [20, 50, 100]
subsetevalnum = [20, 50, 100, 200, 300]
etimeperedge = [5, 10, 20, 30]

hps = [edgenumPit2adds, cannumpits, knns, subsetevalnum, etimeperedge]


if __name__ == "__main__":
    _A_obs, feas, labels = Preprocess.loaddata(
        'data/{}.npz'.format(dataset), llc=False)
    _A_prev = _A_obs
    adj, remained, deleted = spa.delete_edges(_A_obs, k=percent)
    print('preprocess, delete some edges, remaind edges num(bi): {}'.format(adj.nnz))

    # t = IALGE(adj, feas, labels, 100, 10, 10, edge_Rec='rand')
    # t = IALGE(adj, feas, labels, 100, 10, 10, seed=1, dropout=0)
    # t = IALGE(adj, feas, labels, 100, 10, 10, seed=1, dropout=0, edge_Rec='rand_test', deleted_edges=deleted, gemodel=None)
    # t = IALGE(adj, feas, labels, 100, 10, 10, seed=1, dropout=0, edge_Rec='MLE', deleted_edges=deleted, gemodel=None)
    # t = IALGE(adj, feas, labels, 100, 10, 10)

    def testhp(index=1, testtimes=3):
        hp = [20, 20, 50, 50, 10]
        ds = (dataset, percent)
        for x in hps[index]:
            hptemp = copy.deepcopy(hp)
            hptemp[index] = x
            for y in range(testtimes):
                t = IALGE(adj, feas, labels, 100, 10, 10, seed=1, dropout=0, edge_Rec='KNN',
                          deleted_edges=deleted, gemodel=None, initadj=_A_prev, params=hptemp, dataset=ds, testindex=y)
                # print('''\nstart...\n''')
                t.run()

    # for i in range(len(hps)):
    #     testhp(index=i)

    # (edgenumPit2add, cannumPit, knn, subsetnum, etimeperedge)
    # params = [(20, 20, 20, 20, 5), (20, 100, 50, 200, 10), (20, 20, 50, 200, 10), (20, 200, 50, 200, 10), (20, 100, 100, 200, 10)]
    # for pp in params:
    #     t = IALGE(adj, feas, labels, 100, 10, 10, seed=1, dropout=0, edge_Rec='KNN', deleted_edges=deleted, gemodel=None, initadj=_A_prev, params=pp)
    #     print('''\nstart...\n''')
    #     # t.test(adj)
    #     t.run()

    pp = (20, 20, 20, 20, 5)
    ds = (dataset, percent)
    t = IALGE(adj, feas, labels, 100, 10, 10, seed=1, dropout=0,
              deleted_edges=deleted, initadj=_A_prev, params=pp, dataset=ds, split_share=share_cora)
    # __init__(adj, features, labels, tao, n, s, gemodel='GCN', cangen='knn', edgeEval='max',
    # edgeUpdate='easy', early_stop=10, seed=-1, dropout=0.5,
    # deleted_edges=None, initadj=None, params=None, dataset=('cora', 1), testindex=1):

    print('''\nstart...\n''')
    t.run()
