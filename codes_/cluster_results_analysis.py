'''
THIS SCRIPT REQUIRES THE DATA HARD DRIVE TO FUNCTION
ANALYSIS OF RESULTS FROM CLUSTER

'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
import seaborn as sns
from matplotlib.patches import Rectangle


full_path_data ="/media/nikos/Elements/ClusterResults/results_301121_AllDigits_20runs_8280interstim_random_turnover_alternating_presentation/"
file_type ="predictons"
suffix = "_nikos_"
bad_seeds=[]

full_path_data_rt ="/media/nikos/Elements/ClusterResults/results_301121_AllDigits_20runs_8280interstim_random_turnover_alternating_presentation/"
full_path_data_ct='/media/nikos/Elements/ClusterResults/results_261121_AllDigits_20runs_8280interstim_constrained_turnover_alternating_strongLTD/'
full_path_data_nt='/media/nikos/Elements/ClusterResults/results_021221_AllDigits_20runs_8280interstim_no_turnover_alternating_presentation_200_iters/'

full_path_data=full_path_data_nt

#turn_accs=[]
accs_list=[]
for d2 in range(10):
	for d1 in range(d2):	
		
		digit1=str(d1)
		digit2=str(d2)


		accs=[]
		for s in range(1,21):
			
			if s in bad_seeds:
				continue
			seed = str(s)
			df=pd.read_table(full_path_data + file_type + suffix + seed + "_digits_" 
		      + digit1 + "_" + digit2 + ".txt",header=None, error_bad_lines=False, sep = " ")	





		
			
			#THIS IS FOR FINAL ACCURACY ANALYSIS
			#REQUIRES FILE TYPE = predictons!
			acc=round(sum(df[2])/len(df[2]), 2)
			accs.append(acc)
		accs_list.append(accs)

acc_table=np.array(accs_list)


mean_accs=np.mean(acc_table,axis = 1)
max_accs=np.max(acc_table,axis = 1)
min_accs=np.min(acc_table,axis = 1)
accs_std=np.std(acc_table,axis = 1)


print("Mean mean accuracy= ",100*round(np.mean(acc_table),3),"%, and mean std= ",100*round(np.mean(accs_std),4),"%")
np.mean(accs_std)


meantable = np.zeros((10,10))
maxtable = np.zeros((10,10))
mintable = np.zeros((10,10))
stdtable = 	np.zeros((10,10))
c=0

for d2 in range(10):
	for d1 in range(d2):	
		meantable[d1,d2] = 100*mean_accs[c]
		maxtable[d1,d2] = 100*max_accs[c]
		mintable[d1,d2] = 100*min_accs[c]
		stdtable[d1,d2] = 100*accs_std[c]
		c += 1

plt.figure(figsize=(10, 8), dpi=80)
ax = sns.heatmap(meantable, linewidth=0.5, cmap="inferno",annot=stdtable, vmin=50 , vmax = 100)
ax.yaxis.tick_right()
ax.xaxis.tick_top()


ax.set_xticklabels([0,1,2,3,4,5,6,7,8,9],fontsize=12,fontweight='bold')
ax.set_yticklabels([0,1,2,3,4,5,6,7,8,9], rotation = 0,fontsize=12,fontweight='bold')
plt.title('Mean Accuracy Heatmap', fontweight= "bold",fontsize=14,loc='center')

plt.suptitle('Mean Accuracy Heatmap', fontweight= "bold", x= 0.435, fontsize=18)
plt.title('(std in boxes)', fontsize=13,fontweight='bold')

plt.savefig('./temp/MAH.eps',bbox_inches='tight',dpi=600)
plt.show()





turn_accs.append(acc_table)


accs_3d_arr_turn_diff_runs=np.array(turn_accs)

mean_accs_2d_arr_turn_diff = np.mean(accs_3d_arr_turn_diff_runs, axis=2)
plt.figure(figsize=(10, 8), dpi=80)
for i in range(len(mean_accs_2d_arr_turn_diff[0])):
	plt.plot(turns, 100*mean_accs_2d_arr_turn_diff[:,i], 'ko-',ms=10, alpha=0.5)
plt.yticks(np.arange(50,110,10),fontsize=10,fontweight="bold")
plt.ylabel('% Accuracy', fontweight = 'bold',fontsize=12)
plt.ylim(-1,110)
plt.xticks(turns,["Random\nTurnover","Constrained\nTurnover","No\nTurnover"],fontweight="bold",fontsize=12)
plt.title("Mean Accuracies across Turnovers", fontsize=14, fontweight='bold')	
plt.savefig("./temp/Mean_Dots_MNIST.png",bbox_inches='tight')
plt.show()


std_accs_2d_arr_turn_diff = np.std(accs_3d_arr_turn_diff_runs, axis=2)
plt.figure(figsize=(10, 8), dpi=80)
for i in range(len(std_accs_2d_arr_turn_diff[0])):
	plt.plot(turns, 100*std_accs_2d_arr_turn_diff[:,i], 'ko-',ms=10, alpha=0.5)
plt.yticks(np.arange(0,20,1),fontsize=10,fontweight="bold")
plt.ylabel('% Accuracy', fontweight = 'bold',fontsize=12)
plt.ylim(-1,20)
plt.xticks(turns,["Random\nTurnover","Constrained\nTurnover","No\nTurnover"],fontweight="bold",fontsize=12)
plt.title("Mean Standard Deviations across Turnovers", fontsize=14, fontweight='bold')	
plt.savefig("./temp/Std_Dots_MNIST.png",bbox_inches='tight')
plt.show()







file_type ="sample_accuracies"
sample_interval=50

'''
for d2 in range(10):
	for d1 in range(d2):	
		digit1=str(d1)
		digit2=str(d2)
		plt.figure(figsize=(10, 10), dpi=80) ##FOR INDIVIDUAL PLOTS PER DIGIT PAIR
		
		accs=[]
		for s in range(1,21):
			seed = str(s)
			df=pd.read_table(full_path_data + file_type + suffix + seed + "_digits_" 
		      + digit1 + "_" + digit2 + ".txt",header=None, error_bad_lines=False, sep = " ")	
			#THIS IS FOR INDIVIDUAL PLOTS PER DIGIT PAIR
			#REQUIRES FILE TYPE = sample_accuracies !
			y=np.vstack(([[0]],df.values))
			x = list(range(sample_interval*0,sample_interval*len(y),sample_interval))				
			plt.plot(x,y)
			plt.grid(True)
			plt.xlabel('Training Iterations', fontweight = 'bold')
			plt.yticks(np.arange(0,110,10))
			plt.ylabel('% Accuracy', fontweight = 'bold')
			plt.title('Learning Curves ' + digit1 + " vs " + digit2, fontsize=14, fontweight='bold')
		plt.show()
		#plt.savefig("./temp/learning_curves_"+digit1+"_"+digit2)

plt.close("all")
'''


rand_turn_accs_mean=[]
cons_turn_accs_mean=[]
no_turn_accs_mean=[]

rand_turn_accs_std=[]
cons_turn_accs_std=[]
no_turn_accs_std=[]


diff=[]

turns=[0,1,2]

for turn in turns:
	for d2 in range(10):
		for d1 in range(d2):	
			
			
			
			if turn==0:				
				full_path_data=full_path_data_rt
			elif turn==1:				
				full_path_data=full_path_data_ct
			else:
				full_path_data=full_path_data_nt
				
			diff.append((d1,d2))
			digit1=str(d1)
			digit2=str(d2)	
			
			accs=[]
			for s in range(1,21):
				seed = str(s)
				df=pd.read_table(full_path_data + file_type + suffix + seed + "_digits_" 
			      + digit1 + "_" + digit2 + ".txt",header=None, error_bad_lines=False, sep = " ")	
				
	
				y=[i for i in df[0]]
				y.insert(0,0)
				
				accs.append(y)
				
			maxlen=max([len(i) for i in accs])
			c=1
			while c!=0:
				c=0
				for i in accs:
					if len(i)!=maxlen:
						c+=1
						i.append(np.nan)
			
						
			accsarr=np.array(accs)
			meanarr=np.nanmean(accsarr,axis=0)
			stdarr=np.nanstd(accsarr,axis=0)
			
			
						
			if turn==0:
				rand_turn_accs_mean.append(meanarr)
				rand_turn_accs_std.append(stdarr)
			elif turn==1:
				cons_turn_accs_mean.append(meanarr)
				cons_turn_accs_std.append(stdarr)	
			else:
				no_turn_accs_mean.append(meanarr)
				no_turn_accs_std.append(stdarr)
				
				
			
			x = list(range(sample_interval*0,sample_interval*len(y),sample_interval))				
					
			plt.figure(figsize=(10, 10), dpi=80)
			plt.plot(x,meanarr,"k-")
			plt.fill_between(x, meanarr + stdarr, meanarr - stdarr,alpha=0.3,facecolor='grey')
			plt.xlabel('Training Iterations', fontweight = 'bold')
			plt.yticks(np.arange(0,110,10))
			plt.ylabel('% Accuracy', fontweight = 'bold')
			plt.title('Learning Curve ' + digit1 + " vs " + digit2, fontsize=14, fontweight='bold')
			plt.show()
			#plt.savefig("./temp/mean_learning_curve_"+digit1+"_"+digit2)
			
			
		
			
		
rt = "r"
ct = "b"	
nt = "g"	

	
for i in range(len(rand_turn_accs_mean)):
		
	plt.figure(figsize=(10, 10), dpi=80)
	xr=list(range(sample_interval*0,sample_interval*len(rand_turn_accs_mean[i]),sample_interval))
	xc=list(range(sample_interval*0,sample_interval*len(cons_turn_accs_mean[i]),sample_interval))
	xn=list(range(sample_interval*0,sample_interval*len(no_turn_accs_mean[i]),sample_interval))
	plt.plot(xr,rand_turn_accs_mean[i],"r")
	plt.plot(xc,cons_turn_accs_mean[i],"b")	
	plt.plot(xn,no_turn_accs_mean[i],"g")
	
	plt.fill_between(xr, rand_turn_accs_mean[i] + rand_turn_accs_std[i], np.clip((rand_turn_accs_mean[i] - rand_turn_accs_std[i]),0,100),alpha=0.3,facecolor='r')
	plt.fill_between(xc, cons_turn_accs_mean[i] + cons_turn_accs_std[i], np.clip((cons_turn_accs_mean[i] - cons_turn_accs_std[i]),0,100),alpha=0.3,facecolor='b')
	plt.fill_between(xn, no_turn_accs_mean[i] + no_turn_accs_std[i], np.clip((no_turn_accs_mean[i] - no_turn_accs_std[i]),0,100),alpha=0.3,facecolor='g')
	
	
	handles = [Rectangle((0,0),1,1,color=c,ec="k") for c in [rt,ct,nt]]
	labels= ["Random Turnover","Constrained Turnover","No Turnover"]
	plt.legend(handles, labels,fontsize=12,prop={'weight':'bold','size':12},loc="lower right")
	plt.xlabel('Training Iterations', fontweight = 'bold')
	plt.yticks(np.arange(0,110,10))
	plt.ylabel('% Accuracy', fontweight = 'bold')
	plt.title('Learning Curves Digit 1: '+ str(diff[i][0]) + " Digit 2: " + str(diff[i][1]), fontsize=14, fontweight='bold')
	plt.savefig("./temp/learning_curve_digit_1_"+ str(diff[i][0]) + "_digit_2_"  + str(diff[i][1]) + ".png",bbox_inches='tight')
	#plt.savefig("./temp/learning_curve_stability_"+ str(diff[i][0]) + "_overlap_"  + str(diff[i][1]) + ".eps",bbox_inches='tight')
	plt.show()
			
		
		
		
		