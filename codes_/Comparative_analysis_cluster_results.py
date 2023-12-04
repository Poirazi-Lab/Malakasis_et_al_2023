'''
THIS SCRIPT REQUIRES THE DATA HARD DRIVE TO FUNCTION
ANALYSIS OF RESULTS FROM CLUSTER

'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
import seaborn as sns
from matplotlib.patches import Polygon
from scipy import stats

full_path_data_1 ="/media/nikos/Elements/ClusterResults/results_261121_AllDigits_20runs_8280interstim_constrained_turnover_alternating_strongLTD/"
full_path_data_2 ="/media/nikos/Elements/ClusterResults/results_261121_AllDigits_20runs_8280interstim_constrained_turnover_alternating_maxCREB/"
#full_path_data_3 ="/media/nikos/Elements/ClusterResults/results_220721_550iters/"
file_type ="predictons"
suffix = "_nikos_"

bad_seeds=[]



all_accs=[]

for full_path_data in [full_path_data_1,full_path_data_2]:
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
	all_accs.append(acc_table)
	

c=0
 	
for d2 in range(10):
	for d1 in range(d2):
		data=[]
		data.append(all_accs[0][c])	
		data.append(all_accs[1][c])
		#data.append(all_accs[2][c])
		
		fig, ax1 = plt.subplots(figsize=(8,6),dpi=80)
		fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)
		
		bp = ax1.boxplot(data, notch=0, sym='+', vert=1, whis=1.5)
		plt.setp(bp['boxes'], color='black')
		plt.setp(bp['whiskers'], color='black')
		plt.setp(bp['fliers'], color='red', marker='+')
		
		# Add a horizontal grid to the plot, but make it very light in color
		# so we can use it for reading data values but not be distracting
		ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
		               alpha=0.5)
		
		ax1.set(axisbelow=True)
		
		# Now fill the boxes with desired colors
		box_colors = ['darkkhaki', 'royalblue']
		#box_colors = ['darkkhaki', 'royalblue','salmon']
		num_boxes = len(data)
		medians = np.empty(num_boxes)
		for i in range(num_boxes):
		    box = bp['boxes'][i]
		    box_x = []
		    box_y = []
		    for j in range(5):
		        box_x.append(box.get_xdata()[j])
		        box_y.append(box.get_ydata()[j])
		    box_coords = np.column_stack([box_x, box_y])
		    # Alternate between Dark Khaki and Royal Blue
		    ax1.add_patch(Polygon(box_coords, facecolor=box_colors[i % 3]))
		    # Now draw the median lines back over what we just filled in
		    med = bp['medians'][i]
		    median_x = []
		    median_y = []
		    for j in range(2):
		        median_x.append(med.get_xdata()[j])
		        median_y.append(med.get_ydata()[j])
		        ax1.plot(median_x, median_y, 'k')
		    medians[i] = median_y[0]
		    # Finally, overplot the sample averages, with horizontal alignment
		    # in the center of each box
		    ax1.plot(np.average(med.get_xdata()), np.average(data[i]),
		             color='w', marker='.', markeredgecolor='k')
		
		# Set the axes ranges and axes labels
		top = 1.15
		bottom = 0.4
		ax1.set_ylim(bottom, top)
		ax1.set_xticklabels(['Control','Overexpressed CREB'],
		                    rotation=0, fontsize=15,fontweight='bold')
		ax1.set_title('Digits '+str(d1)+' and '+str(d2),fontweight="bold",fontsize=18)
		ax1.set_ylabel('Accuracy',fontweight="bold",fontsize=13)
		ax1.set_yticks(np.arange(0.4,1.1,0.1))
		
		for i in range(len(data)):
			for j in range(i):
				
				x1, x2 = j+1, i+1
				y, h, col = np.max(data)+0.02, (i-j)*0.05, 'k'
				
				pval=stats.ttest_ind(data[j],data[i],equal_var=False)[1]
				if 0.01<=pval<0.05:
					plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
					plt.text((x1+x2)*.5, y+h+0.01, "*", ha='center', va='bottom', color=col,fontweight='bold')
				elif 0.001<=pval<0.01:
					plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
					plt.text((x1+x2)*.5, y+h+0.01, "**", ha='center', va='bottom', color=col,fontweight='bold')
				elif 0.0001<=pval<0.001:
					plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
					plt.text((x1+x2)*.5, y+h+0.01, "***", ha='center', va='bottom', color=col,fontweight='bold')
				elif pval<0.0001:
					plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
					plt.text((x1+x2)*.5, y+h+0.01, "****", ha='center', va='bottom', color=col,fontweight='bold')
		
		c += 1
		if pval>=0.05:
			#pass
			continue
		#plt.show()
		plt.savefig("./temp/accuracy_comparison_max_CREB_"+str(d1)+"_and_"+str(d2))
		


plt.close('all')