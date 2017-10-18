
import pandas as pd

def clean_all(time_number):
	name = 'time.%s.log'%time_number
	fh  = open(name,'r')
	# fh  = open('result_simgoid.log','r')
	strs = fh.read()
	strs = strs.split('\n')
	strs = [line.strip() for line in strs]
	state = 0
	prediction = []
	cf = []
	loss = 0
	times = 0
	tmp_state = 0
	epoch  =-1
	loss = -1


	rs = []
	flag = 'init'

	for line_num in xrange(len(strs)):
		line = strs[line_num]

		if line == '************************************************************':
			tmp_state += 1
			if tmp_state == 4:
				times += 1
				tmp_state = 0
			else:
				continue
		if line.startswith('leaerning rate'):
			leaerning_rate = float(line[line.index('te')+2:-1])
			continue
		if line.startswith('training... epoch'):
			epoch = int(line[line.index('epoch')+len('epoch'):line.index(',')])
			loss = float(line[line.index('loss:')+len('loss:'):-1])
			continue
		if line.startswith('reconstructing'):
			mse = float(line[line.index('MSE:')+len('MSE:'):-1])
			continue
		if line.startswith('prediction'):
			flag = 'pred'
			continue
		if flag == 'pred':
			line = line.strip(')(').split(',')
			pred_precision = float(line[0])
			pred_recall = float(line[1])
			rs.append(('prediction',leaerning_rate,epoch,loss,mse,pred_precision,pred_recall))
			flag = 'pred_done'
			continue
		if line.startswith('CF'):
			flag = 'cf'
			continue

		if flag == 'cf':
			if line == '':
				flag = 'cf_done'
			else:
				line = line.strip(')(').split(',')
				rs.append(('cf',leaerning_rate,int(line[0]),str(line[1]),int(line[3]),float(line[4]),float(line[5])))

	df = pd.DataFrame(rs)
	df.columns = ['type','lr','epoch','hidden/loss','knn/MSE','precision','recall']
	df.to_csv(name +'.csv',index=False)

	# prediction	0.1	1	0.093586820249	0.042119737714	0.0546357615894	0.00357663407788
	# cf	0.1	1	 'decode'	5	0.108609271523	0.00587225560059

	g = df.groupby(['type','lr','epoch'])                 
	max_idx = g['precision'].transform(max) == df['precision']
	df_max = df[max_idx]
	df_max.to_csv(name+'.max.log'+'.csv',index=False)


	g_lr = df_max.groupby(['lr','type'])
	lr_max_idx = g_lr['precision'].transform(max) == df_max['precision']
	lr_df_max = df_max[lr_max_idx]
	lr_df_max = lr_df_max.sort_values(by='type')
	lr_df_max.to_csv(name + '.max.lr.log'+'.csv',index=False)


# if __name__ == '__main__':
for i in xrange(1,9):
	print i 
	clean_all(i)