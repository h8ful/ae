import dataset5
data6 = dataset5.load_data(1)



import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pprint import pprint


import pandas as pd

import rec3

%matplotlib inline
import matplotlib.pyplot as plt

def weighted_average(input, targets, weights):
    return nn.functional.mse_loss(input.mul(weights), targets.mul(weights),size_average=False)*1.0/weights.sum()
def rec_pred(pred_ratings, train, test, targets):
    pred = rec3.Rec()
    pred.set_prediction_matrix(train, pred_ratings)
    pred.produce_rec_list(train,targets=targets)
    pred.evaluate(test = test, rec_len=5)
    return pred



input_size = 6040
hidden_size = 200
output_size = 6040
learning_rate = 0.001



num_epochs = 70

cv = 1


representations = dict()
loss_history = list()

train_precision = dict()
test_precision = dict()

train_cf_precision = dict()
test_cf_precision = dict()

train_mse = dict()
test_mse = dict()
all_mse = dict()


net = NetNormWeights(input_size, hidden_size, output_size)
# net = nn.DataParallel(net)

certeria = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
print(net)
net.cuda()
for epoch in range(num_epochs):
    for i_batch, sample_batched in enumerate(data6.train_loader):
        net.train()
        id_batch = sample_batched['id']
        sample = Variable(sample_batched['sample']).cuda()
        groud_truth = Variable(torch.from_numpy(data6.train_dataset.known.T[id_batch.numpy()])).cuda()

#         dropout_index =
        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = net(sample)
        loss = certeria(outputs, groud_truth)
#         print(i_batch, loss.data[0])
        loss.backward()
        optimizer.step()

        print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                   % (epoch + 1, num_epochs, i_batch + 1,
                      len(data6.train_loader) ,loss.data[0]))
        loss_history.append(loss.data[0])
    iters = epoch +1
    if (iters) % 1 == 0:
        print ('_'*60)

        print('iteration %s:'%iters)
        name = 'mapusers-%s-epoch_%s_cv_%s'%(str(net).replace('\n',''), epoch+1, cv)
        loss_df= pd.DataFrame(loss_history,columns=['loss'])
        loss_df.plot(title=name)
        plt.show()

        net.eval()
        # reconstruct
        print("reconstructing...")

        input_matrix = Variable(torch.from_numpy(data6.train_dataset.known.T)).cuda()
        reconstructed = net(input_matrix).cpu()
        # copy representations
        representations[iters] = dict()
        for key in net.representations.keys():
            representations[iters][key] = net.representations[key].copy()
        representations[iters]['output'] = reconstructed.data.numpy()

        #         print((reconstructed.size()))
        print('computing mse...')
        test_user_mask = np.zeros_like(data6.train_dataset.known)
        test_user_mask[data6.test_dataset.targets,:] = 1
        test_user_mask = Variable(torch.from_numpy(test_user_mask))
        test_mse[iters] = weighted_average(reconstructed,
                                           Variable(torch.from_numpy(data6.train_dataset.known)),
                                           (test_user_mask)).data.numpy()[0]
        all_mse[iters] = certeria(reconstructed,Variable(torch.from_numpy(data6.train_dataset.known.T))).data.numpy()[0]
#         print((reconstructed.size()))

        print('prediction, computing precision...')
        reconstructed = reconstructed.data.numpy()
        reconstructed = reconstructed.T
#         print((reconstructed.shape))
        predicted_rating =  np.zeros_like(data6.train_dataset.known)
        predicted_rating[data6.test_dataset.targets,:] = reconstructed

        test_rec = rec_pred(reconstructed, train = data6.train_dataset.known, test = data6.test_dataset.test,
                             targets = data6.test_dataset.targets)
        train_precision[iters] = (train_rec.precision_, train_rec.recall_)
        test_precision[iters] = (test_rec.precision_, test_rec.recall_)


    # CF on representations
        print ("CF on representations...")
        test_cf_precision[iters] = dict()
        for key in representations[iters].keys():
#             for sim in ['dot']:
            for sim in ['asymcos','cosine']:
                cf = rec3.IBCF(sim=sim)
                cf.fit(train_ratings=data6.train_dataset.train, profile = representations[iters][key])

#                 for knn in [5]:
                for knn in [5,50,100,200,500]:
                        print ("CF (%s,%s,%s) on representations..."%((key,sim,knn)))

                        cf.compute_score(input_ratings = data6.train_dataset.newusers, topN = knn ,
                                                 targets=data6.test_dataset.targets)
                        cf.produce_reclist(targets=data6.test_dataset.targets)
                        tmp_cf_perf = cf.evaluate(test=data6.test_dataset.test, rec_len = 5)
                        test_cf_precision[iters][key,sim,knn] = ( tmp_cf_perf)
        print('results:')
        print("train_precision:")
        pprint(train_precision[iters])
        print("test_precision" )
        pprint(test_precision[iters])

        print("test_cf_precision:")
        pprint(test_cf_precision[iters])

        print("train_mse:")
        pprint(train_mse[iters])
        print("test_mse" )
        pprint(test_mse[iters])

        print ('-'*60)

print ('*'*60)
print('results:')
print("train_precision:")
pprint(train_precision)
print("test_precision" )
pprint(test_precision)

print("test_cf_precision:")
pprint(test_cf_precision)

print("train_mse:")
pprint(train_mse)
print("test_mse" )
pprint(test_mse)

print ('*'*60)

tmp = []
for i in (test_cf_precision.keys()):
    for k in (test_cf_precision[i].keys()):
        v = test_cf_precision[i][k]
        tmp.append((i,k[0],k[1],k[2],v[0],v[1]))
test_cf_precision_df = pd.DataFrame(tmp)
test_cf_precision_df = test_cf_precision_df.sort_values(by=4,ascending=False)
pprint(test_cf_precision_df.iloc[0])
