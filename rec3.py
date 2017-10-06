import numpy as np

class Rec():
    def __init__(self):
        self.default_rec_len = 50
    def set_prediction_matrix(self,known_ratings, prediction_matrix):
        '''

        :param prediction_matrix: user*item
        :return:
        '''
        # self.prediction_matrix = prediction_matrix

        self.known_ratings = known_ratings
        self.prediction_matrix = np.multiply(prediction_matrix, 1-known_ratings)
        self.item_num = self.prediction_matrix.shape[1]


        return self
    def produce_rec_list(self, train_ratings, targets):
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
        self.rec_list_ = dict()
        self.test = test
        for u in self.targets:
            y_true = test[u].nonzero()[0]
            # if u == 8: print ('\n',u,y_true )
            # if len(y_true) == 0: continue

            y_pred = self.recommendation_[u][:self.rec_len]
            self.rec_list_[u] = y_pred
            right_rec = len(set(y_true).intersection(set(y_pred))) * 1.0
            precision = right_rec / self.rec_len
            # if user has no rating in test set, ignor him
            recall = right_rec / len(y_true)
            self.user_perf[u] = (precision, recall, right_rec)
        self.precision_ = np.average([i[0] for i in self.user_perf.values()])
        self.recall_ = np.average([i[1] for i in self.user_perf.values()])
        return self.precision_, self.recall_

from sklearn.metrics.pairwise import pairwise_distances
# from numba import jit
# @jit
def _asymcos(x,alpha=0.2):
    numerator = x.dot(x.T)
    dominator_a = np.power(numerator, alpha)
    dominator_b = np.power(numerator, (1 - alpha))
    sim = np.zeros_like(numerator).astype(np.float64)
    for i in xrange(numerator.shape[0]):
        for j in xrange(numerator.shape[0]):
            if dominator_a[i, i] ==0 or dominator_b[j, j] == 0:
                sim[i,j] = 0
            else:
                sim[i, j] = numerator[i, j] / dominator_a[i, i] / dominator_b[j, j]
    return sim


class IBCF():
    def __init__(self, sim):
        self.sim = sim


    def asymmetric_cosine(self, x, alpha=0.2):
        '''
        :param x:
        :return:
        '''
        return _asymcos(x,alpha=0.2)

    def compute_similarity(self, profile):
        '''

        :param profile: each row is a profile vector, r*c size
        :return: similarity matrix, r*r size
        '''
        self.item_num_ = profile.shape[0]
        self.similarities_ = np.zeros((self.item_num_, self.item_num_))
        if self.sim == 'dot':
            self.similarities_ = self.profile.dot(self.profile.T)
        elif self.sim == 'asymcos':
            self.similarities_ = self.asymmetric_cosine(self.profile,alpha=0.2)
        else:
            # import pdb; pdb.set_trace()
            self.similarities_ = 1.0 - pairwise_distances(self.profile, metric=self.sim, n_jobs= 4)
        # set similarity to identity to 0
        self.similarities_ = np.multiply(   self.similarities_, (1-np.eye(self.similarities_.shape[0])))
        return self


    def find_neighbors(self):
        self.item_neighbors_ = dict()
        for item in xrange(self.item_num_):
            self.item_neighbors_[item] = np.argpartition(-1 * self.similarities_[item], self.topN)[:self.topN]
        return  self

    def score_pair_ib(self, user_id, item_id):
        neighborhood = self.item_neighbors_[item_id]
        score = self.input_ratings[user_id, neighborhood].dot(self.similarities_[item_id,neighborhood])
        return  score

    def fit(self,train_ratings, profile):
        '''
        :param train_ratings: user*item
        :param profile: item based, item*dim
        :return: self
        '''
        self.train_ratings = train_ratings
        self.profile = profile

        self.compute_similarity(profile)
        return self

    def compute_score(self,input_ratings, topN, targets):
        self.input_ratings = input_ratings
        self.topN = topN
        self.find_neighbors()
        self.known_ratings = self.train_ratings + self.input_ratings
        self.comsumed_num_ = {}
        self.candinates_num_ = {}

        self.predicted_score_ = np.zeros(self.train_ratings.shape)
        for u in targets:
            consumed = set(self.input_ratings[u, :].nonzero()[0])
            # self.comsumed_num_[u] = set()
            # import pdb; pdb.set_trace()
            candinates = list(set([i for k in consumed for i in self.item_neighbors_[k] ]) - (consumed))
            # for i in xrange(self.item_num_):
            for i in candinates:
                self.predicted_score_[u, i] = self.score_pair_ib(user_id=u, item_id=i)
        return self

    def produce_reclist(self,targets):
        # predict score first, ensure self.predicted_score_ exsit
        if True:
        # try:
            self.rec_ = Rec()
            self.rec_.set_prediction_matrix(self.train_ratings+self.input_ratings,self.predicted_score_)
            self.rec_.produce_rec_list(self.known_ratings,targets)
            self.recommendations_ = self.rec_.recommendation_
            return self
        # except Exception:
        #     raise Exception
    def evaluate(self, test, rec_len):
        self.test = test
        return self.rec_.evaluate(test=test, rec_len=rec_len)
