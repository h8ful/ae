import pandas as pd
import numpy as np

df_list = []
for time_number in xrange(1,8):
	name = 'time.%s.log'%time_number + '.max.lr.log'+'.csv'
	df_list.append(pd.read_csv(name))
	keys = []
	rs = {}
	cols = ['epoch','hidden/loss','knn/MSE','precision','recall']
	for t in enumerate(df_list[0].itertuples(index=False)):
		r = t[1]
		for c in cols:
			keys.append((r.type, r.lr, c))
	for k in keys:
		rs[k] = []
	for df in df_list:
		for k in keys:
			rs[k].append(df[df.type == k[0]][df.lr == k[1]][k[2]].values[0])
	for k in keys:
		try:
			rs[k].append(np.mean(rs[k]))
			rs[k].append(np.std(rs[k]))

		except Exception as e:
			pass

df = pd.DataFrame.from_dict(rs,'index')
df['index'] = df.index

df['type'] = df['index'].apply(lambda x:x[0])
df['lr'] = df['index'].apply(lambda x:x[1])
df['val'] = df['index'].apply(lambda x:x[2])

df = df.drop('index',axis=1)
columns = list(df.columns[-3:]) + list(df.columns[-5:-3]) + list(df.columns[:-5])
df = df[columns]
df.columns = list(df.columns[:3]) + ['mean','std'] + [i+1 for i in list(df.columns[5:])]
df = df.sort_values(by=list(df.columns[:3]))
df.to_csv('all.csv',index=False)



