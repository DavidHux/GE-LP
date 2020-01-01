from iterAddLinks import IALGE
from u import Preprocess, spa

import warnings
import os

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == "__main__":
    _A_obs, feas, labels = Preprocess.loaddata(
        'data/citeseer.npz', llc=True)
    _A_prev = _A_obs
    adj, remained, deleted = spa.delete_edges(_A_obs)
    print('preprocess, delete some edges, remaind edges num(bi): {}'.format(adj.nnz))

    # t = IALGE(adj, feas, labels, 100, 10, 10, edge_Rec='rand')
    # t = IALGE(adj, feas, labels, 100, 10, 10, seed=1, dropout=0)
    # t = IALGE(adj, feas, labels, 100, 10, 10, seed=1, dropout=0, edge_Rec='rand_test', deleted_edges=deleted, gemodel=None)
    t = IALGE(adj, feas, labels, 100, 10, 10, seed=1, dropout=0, edge_Rec='MLE', deleted_edges=deleted, gemodel=None)
    # t = IALGE(adj, feas, labels, 100, 10, 10)


    print('''
    start...
    ''')
    # t.test(adj)
    t.run()
