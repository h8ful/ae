# def get_id(x):
#     d = {}
#     conter = 0
#     x = set(x)
#     # x = sorted(list(x))
#     for i in x:
#         if i not in d.keys():
#             d[i] = conter
#             conter += 1
#     return d
#
# a = [1,2,4,5,6,6,6,98]
# d = get_id(a)
# print d

shape = (6040,3706)
#
# import  numpy as np
# from scipy.sparse import csr_matrix
# from  scipy.io import  mmwrite
# from math import floor
# df = {}
# for cv in xrange(5):
#     tmp = np.random.permutation(df3.user.unique())
#     k = int(tmp.shape[0]*0.9)
#     train = []
#     for u in tmp[:k]:
#         train.append(df3[df3.user == u].as_matrix())
#     train = np.vstack(train)
#     test = []
#     for u in tmp[k:]:
#         test.append(df3[df3.user == u].as_matrix())
#     test = np.vstack(test)
#     test = np.random.permutation(test)
#     train_test = test[:int(test.shape[0]/2)]
#     test = test[int(test.shape[0]/2):]
#
#     train = np.vstack([train, train_test])
#
#     train = csr_matrix((train[:,2], (train[:,0], train[:,1])),shape=shape,dtype=np.float32)
#     test = csr_matrix((test[:,2], (test[:,0], test[:,1])),shape=shape,dtype=np.float32)
#     mmwrite('train.%s'%cv, train)
#     mmwrite('test.%s'%cv, test)
