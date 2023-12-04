'''
THIS SCRIPT REQUIRES THE DATA HARD DRIVE TO FUNCTION
ANALYSIS OF RESULTS FROM CLUSTER

'''

import glob
from os import rename
from os.path import isfile



full_path_data ="/media/nikos/Elements/ClusterResults/results/"
file_type_1 ="predictons"
file_type_2 ="sample_accuracies"
file_type_3 ="weights_per_epoch"
suffix = "_nikos_"

pred=glob.glob(full_path_data+file_type_1+'*')
acc=glob.glob(full_path_data+file_type_2+'*')
w=glob.glob(full_path_data+file_type_3+'*')


c=0
uniq=[]
total=[]
for i in pred:	
	asd=i.split('/')[-1].split('_')[6:]
	asds='_'.join([s for s in asd])
	if asds not in uniq:
		uniq.append(asds)
		c+=1
		temp=glob.glob(full_path_data+file_type_1+'*'+asds)
		old=[]
		new=[]
		for j in temp:
			old.append(j.split('/')[-1])
			oldglob=j.split('/')[-1].split('_')
			oldglob.insert(2,str(c))
			new.append('_'.join([s for s in oldglob]))
		total.append((old,new))


'''
for i in total:
	if len(i[0])!=5:
		print(i[0])
'''


'''
total[-1]
isfile(full_path_data+total[-1][0][0])
'''

for i in total:
	for j in range(len(i[0])):	
		rename(full_path_data+i[0][j],full_path_data+i[1][j])


