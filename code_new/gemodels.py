
import numpy as np
import utils_nete as utils

class model_i():

    def __init__(self):
        print('basic gemodel, not completed __init__')
        

    def train(self):
        print('basic gemodel, not completed train method')
        

    def getembeddings(self):
        print('basic gemodel, not completed get embedding method')
        return None

    def setAdj(self, newadj):
        print('basic gemodel, not completed set adj method')
        # self.adj = newadj

    def setFeature(self, feature):
        self.features = feature

    def setLabels(self, labels):
        self.labels = labels

    def performance(self):
        print('basic gemodel, not completed performace method')
        return 0


from models.GCN_n import GCN_n
from models.GCN_s import GCN_s
from u import Preprocess, Eval

class gemodel_GCN(model_i):
    def __init__(self, Adj, features, labels, layersize=16, split_t=None, seed=-1, dropout=0.5, sGCN=False):
        # print('GCN model init')
        self.Adj = Adj
        self.features = features
        self.labels = labels

        _N = Adj.shape[0]
        _K = labels.max()+1
        self._Z_obs = np.eye(_K)[labels]
        self.sizes = [layersize, _K]
        self.seed = seed
        self.dropout = dropout

        if sGCN:
            self.GCN = GCN_s
        else:
            self.GCN = GCN_n

        if split_t == None:
            self.split_train, self.split_val, self.split_unlabeled = Preprocess.splitdata(_N, self.labels)
        else:
            assert type(split_t) == tuple and len(split_t) == 3
            self.split_train, self.split_val, self.split_unlabeled = split_t

        adj = utils.preprocess_graph(self.Adj)
        self.model = self.GCN(self.sizes, adj, self.features, "gcn_orig", gpu_id=None, seed=self.seed, params_dict={'dropout': self.dropout})

    def setAdj(self, newadj):
        self.Adj = newadj
        adj_processed = utils.preprocess_graph(self.Adj)
        self.model = self.GCN(self.sizes, adj_processed, self.features, "gcn_orig", gpu_id=None, seed=self.seed, params_dict={'dropout': self.dropout})

    def train(self):
        self.model.train(self.split_train, self.split_val, self._Z_obs, print_info=False)

    def getembeddings(self):
        predictions = self.model.predictions.eval(session=self.model.session, feed_dict={self.model.node_ids: range(self.model.N)})

        return predictions

    def performance(self, standard='acu', prt=False):
        if standard == 'acu':
            per = Eval.acu(self.getembeddings(), self.labels)
            if prt:
                print('acu: {}'.format(per))
            return per

        print('err standard for performace')

    def acu(self):
        test_pred = self.model.predictions.eval(session=self.model.session, feed_dict={self.model.node_ids: self.split_unlabeled})
        test_real = self.labels[self.split_unlabeled]
        return Eval.acu(test_pred, test_real)


from multiprocessing import Process, Pool
class Modeltest_GCN():
    '''
    run a GCN model in a sub process, for parallel computing
    '''

    def f(adj, fea, labels, split_t=None, seed=-1, dropout=0.5):
        gcn = gemodel_GCN(adj, fea, labels, split_t=split_t, seed=seed, dropout=dropout)
        gcn.train()
        return (gcn.getembeddings(), gcn.acu())

    def subprocess_GCN(adj, fea, labels, split_t=None, seed=-1, dropout=0.5):
        if split_t == None:
            print('error params in utils.u.model.subprocess_GCN')
        p = Pool()
        r = p.apply_async(Modeltest_GCN.f, args=(adj, fea, labels, split_t, seed, dropout))
        p.close()
        p.join()
        res = r.get()
        return res

if __name__ == "__main__":
    gemodel_GCN(1,2,3)