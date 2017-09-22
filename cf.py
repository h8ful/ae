# from __future__ import print_function
#
# import numpy as np
# from scipy.spatial import distance
#
# from scipy.io import mmread
# from scipy.sparse import csr_matrix
# from sklearn.metrics.pairwise import pairwise_distances
# import os

# from __future__ import print_function

import numpy as np
# from scipy.spatial import distance
#
# from scipy.io import mmread
# from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import pairwise_distances
# import os


# del IBCF
class IBCF():
    def __init__(self, sim, knn, topk):
        self.sim = sim
        self.knn = knn
        self.topk = topk

    def asymcos(self, u, v, alpha=0.2):
        result = u.dot(v).astype(np.float32)
        result = result / ((np.power(u.dot(u), alpha)).dot(np.power((v.dot(v), 1 - alpha))))
        return result

    def psim(self, X, metric='cosine'):
        sim = np.zeros((X.shape[0], X.shape[0]))
        if metric == 'dot':
            sim = X.dot(X.T)
        elif metric == 'asymcos':
            for i in xrange(X.shape[0]):
                for j in xrange(X.shape[0]):
                    u = X[i, :]
                    v = X[j, :]
                    sim[i, j] = self.asymcos(u, v, alpha=0.2)
        else:
            sim = 1 - pairwise_distances(X, metric=self.sim, n_jobs=10)
        return sim

    def fit(self, X):
        self.X = X
        self.similarities_ = self.psim(X, metric=self.sim)
        #         self.similarities_ = pairwise_distances(X,metric=self.sim,n_jobs=10) * 0.3
        self.item_neighbors_ = dict()
        #         item_num = X.shape[1]
        self.item_num = X.shape[0]
        for i in xrange(self.item_num):
            #             self.item_neighbors_[i] = np.argpartition(self.similarities_[i, :], -1 * self.knn)
            self.similarities_[i, i] = 0
            #             import pdb; pdb.set_trace()

            self.item_neighbors_[i] = np.argpartition(self.similarities_[i, :], self.item_num - self.knn)[-self.knn:]

        # import pdb; pdb.set_trace()
        return self

    def predict(self, targets):
        self.prediction_ = np.zeros_like(self.X)
        self.recommendation_ = dict()
        self.targets = targets
        for user in targets:
            # self.purchased_items = self.X[:, user].nonzero()[0]
            #             purchased_items = self.X[:, user].nonzero()[0]
            for i in xrange(self.item_num):
                neighbors = self.item_neighbors_[i]
                self.prediction_[i, user] = self.X[neighbors, user].dot(self.similarities_[neighbors, i])
            purchased_items = self.X[:, user].nonzero()[0]
            self.prediction_[purchased_items, user] = 0
            s = self.prediction_[:, user].shape
            #             self.prediction_[:,user] = np.random.random(size=s)

            self.recommendation_[user] = np.argpartition(self.prediction_[:, user], self.item_num - self.topk)[
                                         -1 * self.topk:]
        # import pdb; pdb.set_trace()

        #             print(user, self.recommendation_[user])
        return self

    def evaluate(self, test):
        self.user_perf = dict()
        for u in self.targets:
            y_true = test[u].nonzero()[0]
            y_pred = self.recommendation_[u]
            right_rec = len(set(y_true).intersection(set(y_pred))) * 1.0
            precision = right_rec / self.topk
            recall = right_rec / len(y_true)
            self.user_perf[u] = (precision, recall)
        self.precision_ = np.average([i[0] for i in self.user_perf.values()])
        self.recall_ = np.average([i[1] for i in self.user_perf.values()])
        return self.precision_, self.recall_


def main():
    from scipy.io import mmread
    import os
    DATA_DIR = './ml-1m-5cv'
    perf = dict()
    for cv in xrange(5):
        train = mmread(os.path.join(DATA_DIR, 'train.%s' % cv)).A
        test = mmread(os.path.join(DATA_DIR, 'test.%s' % cv)).A
        targets = np.unique(test.nonzero()[0])
        cf = IBCF(sim='cosine', knn=133, topk=5)
        cf.fit(train)
        cf.predict(targets)
        precision, recall = cf.evaluate(test)
        print(cv,':',precision,recall)





if __name__ == '__main__':
    main()

