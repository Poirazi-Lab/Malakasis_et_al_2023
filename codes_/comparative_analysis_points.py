'''
THIS SCRIPT REQUIRES THE DATA HARD DRIVE TO FUNCTION
ANALYSIS OF RESULTS FROM CLUSTER

'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

full_path_data_1 ="/media/nikos/Elements/ClusterResults/results_191121_AllDigits_20runs_8280interstim_constrained_turnover_alternating/"
full_path_data_2 ="/media/nikos/Elements/ClusterResults/results_221121_AllDigits_20runs_8280interstim_constrained_turnover_alternating_noCREB/"
#full_path_data_3 ="/media/nikos/Elements/ClusterResults/results/"
file_type ="predictons"
suffix = "_nikos_"




all_accs=[]

for full_path_data in [full_path_data_1,full_path_data_2]:
	accs_list=[]
	for d2 in range(10):
		for d1 in range(d2):	
			
			digit1=str(d1)
			digit2=str(d2)
			
			accs=[]
			for s in range(1,21):
				
				seed = str(s)
				df=pd.read_table(full_path_data + file_type + suffix + seed + "_digits_" 
			      + digit1 + "_" + digit2 + ".txt",header=None, error_bad_lines=False, sep = " ")	
	
	
	
	
	
	
				
				#THIS IS FOR FINAL ACCURACY ANALYSIS
				#REQUIRES FILE TYPE = predictons!
				acc=round(sum(df[2])/len(df[2]), 2)
				accs.append(acc)
			accs_list.append(accs)
	
	acc_table=np.array(accs_list)
	all_accs.append(acc_table)
	

c=0
 	
for d2 in range(10):
	for d1 in range(d2):
		data=[]
		data.append(all_accs[0][c])	
		data.append(all_accs[1][c])
		
		fig, ax1 = plt.subplots(figsize=(8,6),dpi=80)
		fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)
		for i in range(len(data[0])):	
			plt.plot((0,1), (data[0][i],data[1][i]),marker='D',c=(i/20,0,0,1))
		
	
		ax1.set(axisbelow=True)



		top = 1.15
		bottom = 0
		ax1.set_ylim(bottom, top)
		ax1.set_xticks([0,1])
		ax1.set_xticklabels(['Global','Local'],
		                    rotation=45, fontsize=8,fontweight='bold')
		ax1.set_title('Comparison of Global and Local Protein Synthesis, Digits '+str(d1)+' and '+str(d2),fontweight="bold")
		ax1.set_ylabel('Accuracy',fontweight="bold")
		
		for i in range(len(data)):
			for j in range(i):
				x1, x2 = j, i
				y, h, col = np.max(data)+0.02, (i-j)*0.05, 'k'
				
				pval=stats.ttest_ind(data[j],data[i],equal_var=False)[1]
				if 0.01<pval<=0.05:
					plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
					plt.text((x1+x2)*.5, y+h+0.01, "*", ha='center', va='bottom', color=col,fontweight='bold')
				elif 0.001<pval<=0.01:
					plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
					plt.text((x1+x2)*.5, y+h+0.01, "**", ha='center', va='bottom', color=col,fontweight='bold')
				elif 0.0001<pval<=0.001:
					plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
					plt.text((x1+x2)*.5, y+h+0.01, "***", ha='center', va='bottom', color=col,fontweight='bold')
				elif pval<=0.0001:
					plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
					plt.text((x1+x2)*.5, y+h+0.01, "****", ha='center', va='bottom', color=col,fontweight='bold')



		plt.show()
		#plt.savefig("./temp/datapoints_accuracy_comparison_G_vs_L_digits_"+str(d1)+"_and_"+str(d2))
		
		plt.close('all')
		c += 1

