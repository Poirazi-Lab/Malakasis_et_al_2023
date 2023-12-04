'''
THIS SCRIPT REQUIRES THE DATA HARD DRIVE TO FUNCTION
ANALYSIS OF RESULTS FROM CLUSTER

predictons_nikos_2_1_digits_3_7_neurons_50_branches_5_syns_1000_labsyns_2_inhsyns_1_Wmax_0.3_Wmin_0.2_slope_50_midpos_0.4_minlr_0.0001_turnoveriters_40_rewireperc_0.9_turnoverthreshold_0.05_maxfreq_40_labelfreq_50_stopperc_0.9.txt

'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
import seaborn as sns
import glob



full_path_data ="/media/nikos/Elements/ClusterResults/results/"
file_type_1 ="predictons"
file_type_2 ="sample_accuracies"
file_type_3 ="weights_per_epoch"
suffix = "_nikos_"

'''
pred=glob.glob(full_path_data+file_type_1+'*')
acc=glob.glob(full_path_data+file_type_2+'*')
w=glob.glob(full_path_data+file_type_3+'*')
'''









#THIS IS FOR FINAL ACCURACY ANALYSIS
#REQUIRES FILE TYPE = predictons!
accs_list=[]		

for i in range(1,401):
	hypers=glob.glob(full_path_data+file_type_1+suffix+str(i)+'_*')	
	accs=[]
	for j in hypers:
		df=pd.read_table(j,header=None, error_bad_lines=False, sep = " ")	
			
		acc=round(sum(df[2])/len(df[2]), 2)
		accs.append(acc)
	if len(accs)!=10:
		accs.append(np.nan)
	accs_list.append(accs)


acc_table=np.array(accs_list)

mean_accs=np.mean(acc_table,axis = 1)
max_accs=np.max(acc_table,axis = 1)
min_accs=np.min(acc_table,axis = 1)
accs_std=np.std(acc_table,axis = 1)


np.argmin(accs_std)

mean_accs[144]
accs_std[144]
acc_table[0]
max_accs[144]
min_accs[144]

np.mean(accs_std)


np.where(min_accs==max(min_accs))

mean_accs[np.where(accs_std<0.0)]

accs_std[np.where(mean_accs>0.8)]

acc_table[44]

hyperdf

hypers=glob.glob(full_path_data+file_type_2+suffix+str(173)+'_*')
hyperlist=hypers[0].split('/')[-1].split('_')[8:]
hypercols=[hyperlist[i-1] for i in range(1,len(hyperlist),2)]
hypercols.append('set')

hyperdf=pd.DataFrame(columns=hypercols)
pd.options.display.max_columns = 21
	
data=[]
for j in range(1,401):
	hypers=glob.glob(full_path_data+file_type_1+suffix+str(j)+'_*')
	hyperlist=hypers[0].split('/')[-1].split('_')[7:]
	hyperlist[-1]=hyperlist[-1][:-4]
	hyperdict={hyperlist[i-1]:float(hyperlist[i]) for i in range(1,len(hyperlist),2)}
	hyperdict['set']=j
	data.append(hyperdict)
hyperdf=hyperdf.append(data)		
	


hyperdf['mean_accuracy']=mean_accs
hyperdf['accuracy_std']=accs_std
hyperdf['max_accuracy']=max_accs
hyperdf['min_accuracy']=min_accs





minacc=0.7
maxstd=0.07

hyperdf[(hyperdf['mean_accuracy']>minacc) & (hyperdf['accuracy_std']<maxstd)]

for i in hyperdf.columns[:-5]:
	ax = sns.displot(hyperdf[(hyperdf['mean_accuracy']>minacc) & (hyperdf['accuracy_std']<maxstd)][i], kde=False, bins=50)
	plt.ylabel('Counts',fontweight="bold")
	plt.xlabel(i,fontweight="bold")
	plt.xticks(list(set(hyperdf[i])),rotation=50)
	plt.title("Accuracy > "+ str(minacc)+ ", std < "+ str(maxstd),fontweight="bold")
	plt.tight_layout()
	plt.show()
	#plt.savefig("temp/Distribution of "+i,dpi=180)
	plt.close()




#THIS IS FOR INDIVIDUAL PLOTS PER HYPERPARAMETER SET
#REQUIRES FILE TYPE = sample_accuracies !

bestsets=list(hyperdf[(hyperdf['mean_accuracy']>minacc) & (hyperdf['accuracy_std']<maxstd)]['set'])

for i in bestsets:
	hypers=glob.glob(full_path_data+file_type_2+suffix+str(i)+'_*')
	
	
	plt.figure(figsize=(5, 5), dpi=80)
	for j in hypers:
		df=pd.read_table(j,header=None, error_bad_lines=False, sep = " ")
		
		
		y=np.vstack(([[0]],df.values))
		x = list(range(50*0,50*len(y),50))			
		plt.plot(x,y)
		plt.grid(True)
		plt.xlabel('Training Iterations', fontweight = 'bold')
		plt.yticks(np.arange(0,110,10))
		plt.ylabel('% Accuracy', fontweight = 'bold')
		plt.title('Learning Curves, set ' + str(i), fontsize=14, fontweight='bold')
	plt.show()
	#plt.savefig("./temp/learning_curves_hyperparam_set_"+str(i))



plt.close('all')


for i in bestsets:
	print(i,"set score(mean acc/std):",hyperdf.iloc[i-1]['mean_accuracy']/hyperdf.iloc[i-1]['accuracy_std'])


hyperdf.iloc[181]



'''
23/06
dead runs --> 184,164,137,133,131,113,92,85,84,82,81,79,75,57,56,54,46,48,49,45,42,32,23,16,11,8
good --> 189,181(long),173(consistant),168,167(medium),149,122,118,111,95,94,88,80,71,34,29,17
check^^:122,118,88,80,34,17
weird ---> 30



BEST----> PREDICTIONS -> 45, SAMPLE_ACCS -> 88, WEIGHTS -> 176 (SAME SET, STUPID NAMING BY ME)
'''

