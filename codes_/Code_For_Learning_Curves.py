import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

####MNIST####
#syns=[875,1750,3500]
#syns=[1750]
#############

####SHAPES####
syns=[500,1000,2000]
##############

for syn in syns:
	
	####MNIST####
	##Full
	#rand_turn_accs=np.load('./temp/MNIST/Full/Learning_Curves/Accuracy_Table_For_Learning_Curves_Turnover_'+str(syn)+'.npy',allow_pickle=True)
	#no_turn_accs=np.load('./temp/MNIST/Full/Learning_Curves/Accuracy_Table_For_Learning_Curves_No_Turnover_'+str(syn)+'.npy',allow_pickle=True)
	
	##6_pairs
	#rand_turn_accs=np.load('./temp/MNIST/6_pairs/Learning_Curves/Accuracy_Table_For_Learning_Curves_Turnover_'+str(syn)+'.npy',allow_pickle=True)
	#no_turn_accs=np.load('./temp/MNIST/6_pairs/Learning_Curves/Accuracy_Table_For_Learning_Curves_No_Turnover_'+str(syn)+'.npy',allow_pickle=True)
		
	
	
	###Linear Net(ONLY 1750 SYNS)
	
	##Full
	#rand_turn_accs=np.load('./temp/MNIST/Linear_Net/Full/Learning_Curves/Accuracy_Table_For_Learning_Curves_Turnover_'+str(syn)+'.npy',allow_pickle=True)
	#no_turn_accs=np.load('./temp/MNIST/Linear_Net/Full/Learning_Curves/Accuracy_Table_For_Learning_Curves_No_Turnover_'+str(syn)+'.npy',allow_pickle=True)
	
	##6_pairs
	#rand_turn_accs=np.load('./temp/MNIST/Linear_Net/6_pairs/Learning_Curves/Accuracy_Table_For_Learning_Curves_Turnover_'+str(syn)+'.npy',allow_pickle=True)
	#no_turn_accs=np.load('./temp/MNIST/Linear_Net/6_pairs/Learning_Curves/Accuracy_Table_For_Learning_Curves_No_Turnover_'+str(syn)+'.npy',allow_pickle=True)
			
	
	#############
	
	####SHAPES####
	rand_turn_accs=np.load('./temp/Shapes/3_diffs/Learning_Curves/Accuracy_Table_For_Learning_Curves_Turnover_'+str(syn)+'.npy',allow_pickle=True)
	no_turn_accs=np.load('./temp/Shapes/3_diffs/Learning_Curves/Accuracy_Table_For_Learning_Curves_No_Turnover_'+str(syn)+'.npy',allow_pickle=True)
	##############

	sample_interval=20
	rt = "r"
	nt = "b"	
	for i in range(len(rand_turn_accs)):
		
		meanvec_turn=np.nanmean(rand_turn_accs[i][0],axis=0)
		stdvec_turn=np.nanstd(rand_turn_accs[i][0],axis=0)
		
		meanvec_noturn=np.nanmean(no_turn_accs[i][0],axis=0)
		stdvec_noturn=np.nanstd(no_turn_accs[i][0],axis=0)
			
		
		plt.figure(figsize=(10, 10), dpi=80)
		xr=list(range(sample_interval*0,sample_interval*len(meanvec_turn),sample_interval))
		xn=list(range(sample_interval*0,sample_interval*len(meanvec_noturn),sample_interval))
		plt.plot(xr,meanvec_turn,rt)
		plt.plot(xn,meanvec_noturn,nt)
		
		
		
		plt.fill_between(xr, np.clip((meanvec_turn + stdvec_turn),0,100), np.clip((meanvec_turn - stdvec_turn),0,100),alpha=0.3,facecolor=rt)		
		plt.fill_between(xn, np.clip((meanvec_noturn + stdvec_noturn),0,100), np.clip((meanvec_noturn - stdvec_noturn),0,100),alpha=0.3,facecolor=nt)	
			
		handles = [Rectangle((0,0),1,1,color=c,ec="k") for c in [rt,nt]]
		labels= ["Turnover","No Turnover"]
		plt.legend(handles, labels,fontsize=12,prop={'weight':'bold','size':12},loc="lower right")
		plt.xlabel('Training Iterations', fontweight = 'bold')
		plt.yticks(np.arange(0,110,10))
		plt.ylabel('% Accuracy', fontweight = 'bold')
		
		#############MNIST################
		#plt.title('Learning Curve Digit 1: '+ str(rand_turn_accs[i][1][0]) + " Digit 2: " + str(rand_turn_accs[i][1][1]) + " Synapses: " + str(syn), fontsize=14, fontweight='bold')
		#plt.savefig("./temp/learning_curve_digit_1_"+ str(rand_turn_accs[i][1][0]) + "_digit_2_"  + str(rand_turn_accs[i][1][1])  + "_synapses_" + str(syn) + ".png",bbox_inches='tight')
		#plt.savefig("./temp/learning_curve_digit_1_"+ str(rand_turn_accs[i][1][0]) + "_digit_2_"  + str(rand_turn_accs[i][1][1]) + "_synapses_" + str(syn) + ".svg",bbox_inches='tight')
		##################################
		
		#############SHAPES################
		plt.title('Learning Curve Stability: '+ str(rand_turn_accs[i][1][0]) + " Overlap: " + str(rand_turn_accs[i][1][1]) + " Synapses: " + str(syn), fontsize=14, fontweight='bold')
		plt.savefig("./temp/learning_curve_stability_"+ str(rand_turn_accs[i][1][0]) + "_overlap_"  + str(rand_turn_accs[i][1][1])  + "_synapses_" + str(syn) + ".png",bbox_inches='tight')
		plt.savefig("./temp/learning_curve_stability_"+ str(rand_turn_accs[i][1][0]) + "_overlap_"  + str(rand_turn_accs[i][1][1]) + "_synapses_" + str(syn) + ".svg",bbox_inches='tight')
		###################################
		
		plt.show()
