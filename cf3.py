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
    def __init__(self, sim, knn, topk, s):
        self.sim = sim
        self.knn = knn
        self.topk = topk
        self.s = s

    def asymcos(self, u, v, alpha=0.2):
        # print(type(u))
        # print(type(v))
        # print((((np.power(np.dot(v,v.T), (1-alpha)) ))))
        # # print()
        # self.u = u
        # self.v = v

        # import pdb; pdb.set_trace()
        result = np.dot(u,v.T)
        denominator = max(np.power(np.dot(u, u.T), alpha) * np.power(np.dot(v,v.T),(1-alpha)), 1e-10)
        result  = result / denominator
        return result
        # return 88

    def psim(self, X, metric='cosine'):
        sim = np.zeros((X.shape[0], X.shape[0]))
        # print(type(X))
        if metric == 'dot':
            sim = X.dot(X.T)
        elif metric == 'asymcos':
            # # ids = []
            # for i in xrange(X.shape[0]):
            #     for j in xrange(X.shape[0]):
            #         sim[i,j] = self.asymcos(X[i],X[j])
            #         print(i,j)

        # --------------------------------
            #         ids.append((i, j))
            # # from multiprocessing import Pool
            # # p = Pool(10)
            # # def func(idp):
            # #     return self.asymcos(self.X[idp[0]], self.X[idp[1]], alpha=0.2)
            # # sim_trips =  p.map(func, ids)
            # # for i,j,v in sim_trips:
            # #     sim[i,j] = v
            # # for i in xrange(X.shape[0]):
            # #     for j in xrange(X.shape[0]):
            # self.asymcos(X[4,:],X[6,:])
            numerator = np.dot(X,X.T)
            self.alpha = 0.2
            A = np.power(numerator.copy(),self.alpha)
            B = np.power(numerator.copy(),1-self.alpha)
            sim = np.divide(numerator,np.multiply(A,B))

        else:
            sim = 1 - pairwise_distances(X, metric=self.sim, n_jobs=10)

        count = X.dot(X.T)
        count = (count > self.s).astype(int)
        sim = np.multiply(sim, count)
        return sim

    def fit(self, train, X):
        '''


        :param train: ratings, item*user
        :param X: representations, item*dim
        :return: self
        '''
        self.train = train
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
            self.num_neighbor = self.knn if self.knn < self.similarities_.shape[0] else self.similarities_.shape[0]
            self.item_neighbors_[i] = np.argpartition(self.similarities_[i, :], self.item_num - self.num_neighbor)[-self.num_neighbor:]

        # import pdb; pdb.set_trace()
        return self

    def predict(self, targets):
        self.prediction_ = np.zeros_like(self.train)
        self.recommendation_ = dict()
        self.targets = targets
        for user in targets:
            # self.purchased_items = self.X[:, user].nonzero()[0]
            #             purchased_items = self.X[:, user].nonzero()[0]
            for i in xrange(self.item_num):
                neighbors = self.item_neighbors_[i]
                self.prediction_[i, user] = self.train[neighbors, user].dot(self.similarities_[i,neighbors])
            purchased_items = self.train[:, user].nonzero()[0]
            self.prediction_[purchased_items, user] = 0

#             s = self.prediction_[:, user].shape
            #             self.prediction_[:,user] = np.random.random(size=s)
            tmp_k = self.topk if self.topk < self.num_neighbor else self.num_neighbor
            self.recommendation_[user] = np.argpartition(self.prediction_[:, user], self.item_num - tmp_k)[
                                         -1 * tmp_k:]
#         import pdb; pdb.set_trace()

        #             print(user, self.recommendation_[user])
        return self

    def evaluate(self, test):
        '''


        :param test: user*item
        :return:
        '''
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


# def main():
#     from scipy.io import mmread
#     import os
#     DATA_DIR = './ml-1m-5cv'
#     perf = dict()
#     for cv in xrange(5):
#         train = mmread(os.path.join(DATA_DIR, 'train.%s' % cv)).A
#         test = mmread(os.path.join(DATA_DIR, 'test.%s' % cv)).A
#         targets = np.unique(test.nonzero()[0])
#         cf = IBCF(sim='cosine', knn=133, topk=5)
#         cf.fit(train, train)
#         cf.predict(targets)
#         precision, recall = cf.evaluate(test)
#         print(cv,':',precision,recall)


#
#
#
# if __name__ == '__main__':
#     main()

