fh  = open('result_simgoid.log','r')
# fh  = open('result_simgoid.log','r')
strs = fh.read()
strs = strs.split('************************************************************\n'*4)
strs = [s.strip() for s in strs if len(s.strip()) != 0]

time = 1
for s in strs:
	name = 'time.%s.log'%time
	fout = open(name,'wb')
	fout.write(s)
	fout.flush()
	fout.close()
	print time,name, len(s)
	time += 1
