input_size = 6040
hidden_size = 2750
output_size = 6040
loss_history = list()
learning_rate = 0.001

knn = 500

pred_perf = dict()
cf_perf = dict()

linearNet = LinearNet(input_size, hidden_size, output_size)

certeria = nn.MSELoss()
optimizer = torch.optim.Adam(linearNet.parameters(), lr=learning_rate)
print(linearNet.train())
num_epochs = 20
for epoch in range(num_epochs):
    for i_batch, sample_batched in enumerate(data.train_loader):
        #         print(i_batch,sample_batched)

        sample = Variable(sample_batched['sample'])
        #         print(sample)
        #         sample = Variable()
        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = linearNet(sample)
        loss = certeria(outputs, sample)
        #         print(i_batch, loss.data[0])
        loss.backward()
        optimizer.step()

        print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
               % (epoch + 1, num_epochs, i_batch + 1,
                  len(data.train_loader), loss.data[0]))
        loss_history.append(loss.data[0])
    if epoch >= 0 and epoch % 5 == 0:
        #         model = copy.deepcopy(net)
        name = 'model-epoch_%s' % epoch
        torch.save(linearNet.state_dict(), name)
        print('recommendataion on prediction... Epoch %s' % epoch)
        loss_df = pd.DataFrame(loss_history, columns=['loss'])
        loss_df.plot(title='cv = %s, epoch = %s' % (cv, epoch))

        predicted = np.zeros_like(data.test_dataset.data)
        # dataset.test_dataset.part = 'all'
        for item_id in xrange(data.test_dataset.data.shape[1]):
            sample = Variable(torch.from_numpy(data.test_dataset.test[:, item_id]))
            output = linearNet(sample)
            #     print(type(test_input['id']))
            predicted[:, item_id] = output.data.numpy()
        pred_perf[epoch] = pred_recommend(train=data.train_dataset.train, test=data.test_dataset.test,
                                          predicted=predicted)

        for key in linearNet.repres.keys():
            print('recommendataion on cf_hidden... Epoch %s, layper %s' % (epoch, key))
            hidden_layer = output.data.numpy()
            cf_result = cf_on_hidden_layer(knn=knn, hidden_layers=hidden_layer, test=data.test_dataset.test)
        cf_perf[(epoch, key)] = (cf_result)


{1: (1, (0.09602649006622517, 0.010572003382280011))}, {8: (8, (0.48576158940397346, 0.052608912386610057))}, {6: (6, (0.47384105960264911, 0.052105718157899633))}, {4: (4, (0.3688741721854305, 0.041122600131997117))}, {10: (10, (0.49933774834437089, 0.054367846769667071))}, {2: (2, (0.18774834437086091, 0.019880892417508515))}, {7: (7, (0.47251655629139067, 0.051778644415428093))}, {3: (3, (0.27682119205298017, 0.031035966045551686))}, {5: (5, (0.45860927152317882, 0.051940949961780232))}, {9: (9, (0.49238410596026488, 0.053800012991470275))},

count = X.dot(X.T)
count = (count > 4).astype(int)
sim = np.multiply(sim, count)
return sim


cf_model3 = IBCF(knn=knn, sim='cosine', topk=5)
cf_model3.fit(train=data.train_dataset.train.T, X = data.train_dataset.train.T)
cf_model3.predict(targets=data.test_dataset.targets)

perf[knn] = (knn, cf_model3.evaluate(test=data.test_dataset.test))
print (perf)



# 6039.000000    3882.

user_num = 6040
item_num = 3883

import os
import  pandas as pd
import  numpy as np
from  scipy.sparse import csr_matrix
from  scipy.io import mmwrite
fname_list = os.listdir(os.curdir)
for f in fname_list:
    cv = int(f[-1])
    tmp = f.split('.')
    tmp = tmp[0]
    tmp = tmp.split('_')
    part = tmp[-1]
    print part, cv
    df = pd.read_csv(f,header=None,sep=' ')
    mtx = csr_matrix((df[2].values,(df[0].values, df[1].values)), shape=(user_num,item_num),dtype=np.float32)
    mmwrite("%s.%s"%(part,cv),mtx)