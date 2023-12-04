import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
import seaborn as sns


####MNIST####
#syns=[875,1750,3500]
#syns=[3500]
syns=[1750]
#############

for syn in syns:
	
	####MNIST####
	
	meantable_turn=np.load('./temp/MNIST/Full/Accuracy_Heatmaps/Mean_Table_For_Heatmap_Turnover_'+str(syn)+'.npy')
	stdtable_turn=np.load('./temp/MNIST/Full/Accuracy_Heatmaps/Std_Table_For_Heatmap_Turnover_'+str(syn)+'.npy')

	meantable_noturn=np.load('./temp/MNIST/Full/Accuracy_Heatmaps/Mean_Table_For_Heatmap_No_Turnover_'+str(syn)+'.npy')
	stdtable_noturn=np.load('./temp/MNIST/Full/Accuracy_Heatmaps/Std_Table_For_Heatmap_No_Turnover_'+str(syn)+'.npy')
	
	v_min=50

	###Constrained (Only 3500 syns)
	#meantable_consturn=np.load('./temp/MNIST/3500_Constrained/Full/Accuracy_Heatmaps/Mean_Table_For_Heatmap_Constrained_Turnover_'+str(syn)+'.npy')
	#stdtable_consturn=np.load('./temp/MNIST/3500_Constrained/Full/Accuracy_Heatmaps/Std_Table_For_Heatmap_Constrained_Turnover_'+str(syn)+'.npy')

	#v_min=50

	###Linear Net (Only 1750 syns)

	#meantable_turn=np.load('./temp/MNIST/Linear_Net/Full/Accuracy_Heatmaps/Mean_Table_For_Heatmap_Turnover_'+str(syn)+'.npy')
	#stdtable_turn=np.load('./temp/MNIST/Linear_Net/Full/Accuracy_Heatmaps/Std_Table_For_Heatmap_Turnover_'+str(syn)+'.npy')

	#meantable_noturn=np.load('./temp/MNIST/Linear_Net/Full/Accuracy_Heatmaps/Mean_Table_For_Heatmap_No_Turnover_'+str(syn)+'.npy')
	#stdtable_noturn=np.load('./temp/MNIST/Linear_Net/Full/Accuracy_Heatmaps/Std_Table_For_Heatmap_No_Turnover_'+str(syn)+'.npy')
	
	#v_min=40
	#############



	plt.figure(figsize=(10, 8), dpi=80)
	ax = sns.heatmap(meantable_turn, linewidth=0.5, cmap="inferno",annot=stdtable_turn, vmin=v_min , vmax = 100)
	ax.yaxis.tick_right()
	ax.xaxis.tick_top()
	
	ax.set_xlabel("Digit 1",fontsize=12,fontweight='bold')
	ax.set_ylabel("Digit 2",fontsize=12,fontweight='bold')
	
	ax.set_xticklabels([0,1,2,3,4,5,6,7,8,9],fontsize=12,fontweight='bold')
	ax.set_yticklabels([0,1,2,3,4,5,6,7,8,9], rotation = 0,fontsize=12,fontweight='bold')
	
	plt.suptitle('Mean Accuracy Heatmap, Turnover, Synapses: ' + str(syn), fontweight= "bold", x= 0.435, fontsize=18)
	plt.title('(std in boxes)', fontsize=13,fontweight='bold')
	
	
	plt.savefig('./temp/MAH_random_turnover_' + str(syn) +'_synapses.svg',bbox_inches='tight',dpi=600)
	plt.savefig('./temp/MAH_random_turnover_' + str(syn) +'_synapses.png',bbox_inches='tight',dpi=600)
	plt.show()






	plt.figure(figsize=(10, 8), dpi=80)
	ax = sns.heatmap(meantable_noturn, linewidth=0.5, cmap="inferno",annot=stdtable_noturn, vmin=v_min , vmax = 100)
	ax.yaxis.tick_right()
	ax.xaxis.tick_top()
	 	
	ax.set_xlabel("Digit 1",fontsize=12,fontweight='bold')
	ax.set_ylabel("Digit 2",fontsize=12,fontweight='bold')
	 	
	ax.set_xticklabels([0,1,2,3,4,5,6,7,8,9],fontsize=12,fontweight='bold')
	ax.set_yticklabels([0,1,2,3,4,5,6,7,8,9], rotation = 0,fontsize=12,fontweight='bold')
	 	
	plt.suptitle('Mean Accuracy Heatmap, No Turnover, Synapses: ' + str(syn), fontweight= "bold", x= 0.435, fontsize=18)
	plt.title('(std in boxes)', fontsize=13,fontweight='bold')
	 	
	
	plt.savefig('./temp/MAH_no_turnover_' + str(syn) +'_synapses.svg',bbox_inches='tight',dpi=600)
	plt.savefig('./temp/MAH_no_turnover_' + str(syn) +'_synapses.png',bbox_inches='tight',dpi=600)
	plt.show() 




	'''
	plt.figure(figsize=(10, 8), dpi=80)
	ax = sns.heatmap(meantable_consturn, linewidth=0.5, cmap="inferno",annot=stdtable_consturn, vmin=50 , vmax = 100)
	ax.yaxis.tick_right()
	ax.xaxis.tick_top()
	
	ax.set_xlabel("Digit 1",fontsize=12,fontweight='bold')
	ax.set_ylabel("Digit 2",fontsize=12,fontweight='bold')
	 	
	ax.set_xticklabels([0,1,2,3,4,5,6,7,8,9],fontsize=12,fontweight='bold')
	ax.set_yticklabels([0,1,2,3,4,5,6,7,8,9], rotation = 0,fontsize=12,fontweight='bold')
	
	plt.suptitle('Mean Accuracy Heatmap, Constrained Turnover, Synapses: ' + str(syn), fontweight= "bold", x= 0.435, fontsize=15)
	plt.title('(std in boxes)', fontsize=13,fontweight='bold')
	
	
	#plt.savefig('./temp/MAH_constrained_turnover_' + str(syn) +'_synapses.svg',bbox_inches='tight',dpi=600)
	#plt.savefig('./temp/MAH_constrained_turnover_' + str(syn) +'_synapses.png',bbox_inches='tight',dpi=600)
	plt.show() 
	'''

