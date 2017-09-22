import numpy as np
class Rec():
    def __init__(self):
        self.default_rec_len = 50
    def set_prediction_matrix(self, prediction_matrix):
        '''

        :param prediction_matrix: user*item
        :return:
        '''
        self.prediction_matrix = prediction_matrix
        self.item_num = self.prediction_matrix.shape[1]

        return self
    def produce_rec_list(self, targets):
        self.targets = targets
        self.recommendation_ = dict()
        for user in targets:
            # find the disordered recommendation list
            disorder_idx = np.argpartition(self.prediction_matrix[user, :], self.item_num - self.default_rec_len)[
                       -1 * self.default_rec_len:]
            # order the recomendataion list, descending order
            # import pdb; pdb.set_trace();
            tmp = np.argsort(-1 * self.prediction_matrix[user,disorder_idx])
            self.recommendation_[user] = list(( disorder_idx[tmp] ))
        return self
    def evaluate(self, test, rec_len):
        '''

                :param test: user*item
                :return: self
                '''
        if rec_len > self.default_rec_len:
            raise AttributeError("rec_len > default_rec_len." )

        self.rec_len = rec_len

        self.user_perf = dict()
        self.test = test
        for u in self.targets:
            y_true = test[u].nonzero()[0]
            # if u == 8: print ('\n',u,y_true )
            # if len(y_true) == 0: continue

            y_pred = self.recommendation_[u][:self.rec_len]
            right_rec = len(set(y_true).intersection(set(y_pred))) * 1.0
            precision = right_rec / self.default_rec_len
            # if user has no rating in test set, ignor him
            recall = right_rec / len(y_true)
            self.user_perf[u] = (precision, recall)
        self.precision_ = np.average([i[0] for i in self.user_perf.values()])
        self.recall_ = np.average([i[1] for i in self.user_perf.values()])
        return self.precision_, self.recall_

from sklearn.metrics.pairwise import pairwise_distances

class IBCF():
    def __init__(self, sim, topN):
        self.sim = sim
        self.topN = topN
    def compute_similarity(self, profile):
        '''

        :param profile: each row is a profile vector, r*c size
        :return: similarity matrix, r*r size
        '''
        self.profile = profile
        self.item_num_ = profile.shape[0]
        self.similarities_ = np.zeros(self.item_num_)
        if self.sim == 'dot':
            self.similarities_ = self.profile.dot(self.profile.T)
        else:
            self.similarities_ = pairwise_distances(self.profile, metric=self.sim, n_jobs= 10)
        return self
    def find_neighbors(self):
        self.item_neighbors_ = dict()
        for item in xrange(self.item_num_):
            self.item_neighbors_[item] = np.argpartition(-1 * self.similarities_, self.topN)[:self.topN]
        return  self

    def score_pair_ib(self, user_id, item_id):
        neighborhood = self.item_neighbors_[item_id]
        score = self.train_ratings[user_id, neighborhood].dot(self.similarities_[user_id,neighborhood])
        return  score

    def fit(self,train_ratings, profile):
        '''
        :param train_ratings: user*item
        :param profile: item based, item*dim
        :return: self
        '''
        self.train_ratings = train_ratings
        self.compute_similarity(profile)
        self.find_neighbors()
        return self

    def compute_score(self, targets):
        self.predicted_score_ = np.zeros(self.train_ratings.shape)
        for u in targets:
            consumed = self.train_ratings[u, :].nonzero()
            candinates = [i for i in self.item_neighbors_[k] for k in consumed]
            for i in candinates:
                self.predicted_score_[u, i] == self.score_pair_ib(user_id=u, item_id=i)
        return self

    def produce_reclist(self,targets):
        self.compute_score(targets)
        self.rec_ = Rec()
        self.rec_.set_prediction_matrix(self.predicted_score_)
        self.rec_.produce_rec_list(targets)
        self.recommendations_ = self.rec_.recommendation_
        return self

    def evaluate(self, test, rec_len):
        self.test = test
        return self.rec_.evaluate(test=test, rec_len=rec_len)
