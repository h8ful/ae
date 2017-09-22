from __future__ import print_function

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import rec

class IBCF():
    def __init__(self, sim, knn, topk, s):
        self.sim = sim
        self.knn = knn

        self.topk = topk
        self.s = s
        self.alpha=0.2

    def asymcos(self, X, alpha=0.2):
        sim = np.zeros((X.shape[0], X.shape[0]))
        numerator = np.dot(X, X.T)
        A = np.power(numerator.copy(), self.alpha)
        B = np.power(numerator.copy(), 1 - self.alpha)
        for i in xrange(self.X.shape[0]):
            for j in xrange(self.X.shape[0]):
                sim[i, j] = np.divide(numerator[i, j], max(1e-10,np.multiply(A[i, j], B[i, j])))
        return  sim

    def psim(self, X, metric='cosine'):
        sim = np.zeros((X.shape[0], X.shape[0]))
        if metric == 'dot':
            sim = X.dot(X.T)
        elif metric == 'asymcos':
            sim = self.asymcos(X, self.alpha)
        else:
            sim = 1 - pairwise_distances(X, metric=self.sim, n_jobs=10)

        #filtering out by count
        self.count = X.dot(X.T)
        self.count = (self.count > self.s).astype(int)
        sim = np.multiply(sim, self.count)
        return sim

    def fit(self, train, X):
        '''


        :param train: ratings, user*item
        :param X: representations, item*dim
        :return: self
        '''
        # compute similarities
        self.train = train
        self.X = X
        assert (self.train.shape[1] == self.X.shape[0])
        self.similarities_ = self.psim(X, metric=self.sim)
        # compute neighborhood
        self.item_neighbors_ = dict()
        self.item_num = X.shape[0]
        self.num_neighbor = self.knn if self.knn < self.similarities_.shape[0] else self.similarities_.shape[0]
        self.rec_length = self.topk if self.topk < self.num_neighbor else self.num_neighbor

        for i in xrange(self.item_num):
            self.similarities_[i, i] = 0
            self.item_neighbors_[i] = np.argpartition(self.similarities_[i, :],
                                                      self.item_num - self.num_neighbor)[-self.num_neighbor:]

        return self

    def predict(self, targets, input_data):
        '''


        :param targets: list of target users
        :param input_data: rating of target users, user*item

        :return: self
        '''

        assert (all(np.unique(input_data.nonzero()[0]) == targets)  )
        # import pdb; pdb.set_trace();

        self.input_data = input_data
        self.prediction_ = np.zeros_like(self.train)
        self.recommendation_ = dict()
        self.targets = targets
        for user in targets:
            # self.purchased_items = self.X[:, user].nonzero()[0]
            for i in xrange(self.item_num):
                neighbors = self.item_neighbors_[i]
                self.prediction_[user, i] = self.input_data[user, neighbors].dot(self.similarities_[i,neighbors])
            # import pdb; pdb.set_trace();

            purchased_items = self.train[user,:].nonzero()[0]
            self.prediction_[user, purchased_items] = 0
        self.rec_ = rec.Rec(rec_len=self.topk)
        self.rec_.set_prediction_matrix(self.prediction_)
        
        return self

    def evaluate(self, test):
        '''

        :param test: user*item
        :return: self
        '''
        self.user_perf = dict()
        self.test = test
        for u in self.targets:
            y_true = test[u].nonzero()[0]
            # if u == 8: print ('\n',u,y_true )
            # if len(y_true) == 0: continue

            y_pred = self.recommendation_[u]
            right_rec = len(set(y_true).intersection(set(y_pred))) * 1.0
            precision = right_rec / self.rec_length
            # if user has no rating in test set, ignor him
            recall = right_rec / len(y_true)
            self.user_perf[u] = (precision, recall)
        self.precision_ = np.average([i[0] for i in self.user_perf.values()])
        self.recall_ = np.average([i[1] for i in self.user_perf.values()])
        return self.precision_, self.recall_



