import heapq
import random
import numpy as np
import matplotlib.pyplot as plt
import utils_nete as utils
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
import scipy.sparse as sp
# from gemodels import gemodel_GCN


class Sim():
    def min_sq(vec):
        ''' 最小二乘法
        args:
            vec: N*D array, features

        returns:
            res: ordered pairs, sorted by sims
        '''
        res = []
        for i in range(vec.shape[0]):
            for j in range(i+1, vec.shape[0]):
                sim = sum(pow(vec[i]-vec[j], 2))
                res.append([sim, i, j])
            # print(i, sim)

        res = sorted(res)
        return res
    
    def min_sq(vec1, vec2):
        assert(len(vec1) == len(vec2))
        res = sum(pow(vec1-vec2, 2))
        return res


class heap():
    def __init__(self, size=20, element=-10000):
        self.size = size
        self.element = element
        self.h = [element]*self.size
    
    def push(self, a):
        if a < self.h[0]:
            return

        heapq.heapreplace(self.h, a)
    
    def sortedlist(self):
        return sorted(self.h, reverse=True)



class Preprocess():
    def loaddata(filename, llc=False):
        ''' load data using nettack api
        args:
            filename: dataset name

        returns:
            _A_obs
            _X_obs
            _z_obs
        '''
        _A_obs, _X_obs, _z_obs = utils.load_npz(filename)
        _A_obs = _A_obs + _A_obs.T
        _A_obs[_A_obs > 1] = 1
        _X_obs = _X_obs.astype('float32')
        if not llc:
            return _A_obs, _X_obs, _z_obs
        lcc = utils.largest_connected_components(_A_obs)

        _A_obs = _A_obs[lcc][:, lcc]
        _A_obs = _A_obs + _A_obs.T
        _A_obs[_A_obs > 1] = 1

        _X_obs = _X_obs[lcc].astype('float32')
        _z_obs = _z_obs[lcc]
        return _A_obs, _X_obs, _z_obs

    def splitdata(_N, _z_obs_1):
        seed = 15
        unlabeled_share = 0.8
        val_share = 0.1
        train_share = 1 - unlabeled_share - val_share
        np.random.seed(seed)

        split_train, split_val, split_unlabeled = utils.train_val_test_split_tabular(np.arange(_N),
                                                                                     train_size=train_share,
                                                                                     val_size=val_share,
                                                                                     test_size=unlabeled_share,
                                                                                     stratify=_z_obs_1)

        return split_train, split_val, split_unlabeled


class spa():
    def delete_edges(adj, k = 0.5, strat='random'):
        ''' delete edges from a given graph
            args:
                adj: adjacent matrix, sparse
                k: delete size
            
            returns:
                new_adj: sparse
        '''
        if strat == 'random':
            t = adj.nonzero()
            data = adj.data
            rows = t[0]
            cols = t[1]

            dd = []
            for i in range(len(data)):
                if rows[i] > cols[i]:
                    continue
                dd.append((data[i], rows[i], cols[i]))
            random.shuffle(dd)
            deleted_d = dd[int(len(dd)*k):]
            remained_d = dd[:int(len(dd)*k)]
            rrr = []
            ccc = []
            d = []
            for a, b, c in remained_d:
                if b < c:
                    ccc.append(c)
                    rrr.append(b)
                    d.append(a)
                ccc.append(b)
                rrr.append(c)
                d.append(a)
        return sp.csr_matrix((d, (rrr, ccc)), shape=adj.shape), remained_d, deleted_d


class Eval():
    def acu(predictions, labels):
        preds = np.argmax(predictions, axis=1)
        # precision, recall, fscore, support = score(labels, preds)
        # print('precisions for each class: {}'.format(precision))
        acu = accuracy_score(labels, preds)
        return acu


# x=np.arange(20,350)

class Matplot():
    def line(data, labels=['type1', 'type2', 'type3']):
        ''' plot line
        args:
            data: [[x1, y1], ...]
        '''
        color = ['r--', 'g--', 'b--']
        for i in range(len(data)):
            plt.plot(data[i][0], data[i][1], color[i], label=labels[i])
        # plt.plot(x1,y1,'ro-',x2,y2,'g+-',x3,y3,'b^-')
        # plt.title('The Lasers in Three Conditions')
        plt.xlabel('row')
        plt.ylabel('column')
        plt.legend()
        plt.show()
        
          

# class Model():
#     def subprocess_GCN(adj, fea, labels, sizes, split_t=None):
#         if split_t == None:
#             print('error params in utils.u.model.subprocess_GCN')
#             exit(-1)
#         sp_train, sp_val, sp_test = split_t
        
#     def new_gcn(n_An, _X_obs, _Z_obs, sizes, split_train, split_val, gpu_id=None):
#         gcn = GCN.GCN(sizes, n_An, _X_obs, "gcn_orig", gpu_id=gpu_id)
#         return gcn

#     def new_gcn_train(n_An, _X_obs, _Z_obs, sizes, split_train, split_val, gpu_id=None):
#         gcn = GCN.GCN(sizes, n_An, _X_obs, "gcn_orig", gpu_id=gpu_id)
#         gcn.train(split_train, split_val, _Z_obs)

#         return gcn
    
#     def preds(g_model):
#         logits = g_model.logits.eval(session=g_model.session)
#         predictions = g_model.predictions.eval(session=g_model.session, feed_dict={g_model.node_ids: range(g_model.N)})

#         return predictions

#     def logits(g_model):
#         return g_model.logits.eval(session=g_model.session)


if __name__ == "__main__":
    x1 = [20, 33, 51, 79, 101, 121, 132, 145, 162, 182,
          203, 219, 232, 243, 256, 270, 287, 310, 325]
    y1 = [49, 48, 48, 48, 48, 87, 106, 123, 155, 191,
          233, 261, 278, 284, 297, 307, 341, 319, 341]
    x2 = [31, 52, 73, 92, 101, 112, 126, 140, 153,
          175, 186, 196, 215, 230, 240, 270, 288, 300]
    y2 = [48, 48, 48, 48, 49, 89, 162, 237, 302,
          378, 443, 472, 522, 597, 628, 661, 690, 702]
    x3 = [30, 50, 70, 90, 105, 114, 128, 137, 147, 159, 170,
          180, 190, 200, 210, 230, 243, 259, 284, 297, 311]
    y3 = [48, 48, 48, 48, 66, 173, 351, 472, 586, 712, 804, 899,
          994, 1094, 1198, 1360, 1458, 1578, 1734, 1797, 1892]
    Matplot.line([[x1, y1]])
