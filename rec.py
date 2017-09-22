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
            tmp = np.argsort(-1 * self.prediction_[user,disorder_idx])
            self.recommendation_[user] = list(reversed( disorder_idx[tmp] ))
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

