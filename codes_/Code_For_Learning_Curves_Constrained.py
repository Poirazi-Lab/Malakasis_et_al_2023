import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


syn=3500


####Constrained Turnover(only for 3500 synapses)

##6_pairs
rand_turn_accs=np.load('./temp/MNIST/3500_Constrained/6_pairs/Learning_Curves/Accuracy_Table_For_Learning_Curves_Turnover_'+str(syn)+'.npy',allow_pickle=True)
cons_turn_accs=np.load('./temp/MNIST/3500_Constrained/6_pairs/Learning_Curves/Accuracy_Table_For_Learning_Curves_Constrained_Turnover_'+str(syn)+'.npy',allow_pickle=True)

##Full
#rand_turn_accs=np.load('./temp/MNIST/3500_Constrained/Full/Learning_Curves/Accuracy_Table_For_Learning_Curves_Turnover_'+str(syn)+'.npy',allow_pickle=True)
#cons_turn_accs=np.load('./temp/MNIST/3500_Constrained/Full/Learning_Curves/Accuracy_Table_For_Learning_Curves_Constrained_Turnover_'+str(syn)+'.npy',allow_pickle=True)
	
sample_interval=20

rt = "r"
ct = "y"	

for i in range(len(rand_turn_accs)):
	
	meanvec_turn=np.nanmean(rand_turn_accs[i][0],axis=0)
	stdvec_turn=np.nanstd(rand_turn_accs[i][0],axis=0)
	
	meanvec_consturn=np.nanmean(cons_turn_accs[i][0],axis=0)
	stdvec_consturn=np.nanstd(cons_turn_accs[i][0],axis=0)
		
	
	plt.figure(figsize=(10, 10), dpi=80)
	xr=list(range(sample_interval*0,sample_interval*len(meanvec_turn),sample_interval))
	xc=list(range(sample_interval*0,sample_interval*len(meanvec_consturn),sample_interval))
	plt.plot(xr,meanvec_turn,rt)
	plt.plot(xc,meanvec_consturn,ct)
	
	
	
	plt.fill_between(xr, np.clip((meanvec_turn + stdvec_turn),0,100), np.clip((meanvec_turn - stdvec_turn),0,100),alpha=0.3,facecolor=rt)		
	plt.fill_between(xc, np.clip((meanvec_consturn + stdvec_consturn),0,100), np.clip((meanvec_consturn - stdvec_consturn),0,100),alpha=0.3,facecolor=ct)	
		
	handles = [Rectangle((0,0),1,1,color=c,ec="k") for c in [rt,ct]]
	labels= ["Random Turnover","Constrained Turnover"]
	plt.legend(handles, labels,fontsize=12,prop={'weight':'bold','size':12},loc="lower right")
	plt.xlabel('Training Iterations', fontweight = 'bold')
	plt.yticks(np.arange(0,110,10))
	plt.ylabel('% Accuracy', fontweight = 'bold')
	plt.title('Learning Curve Digit 1: '+ str(rand_turn_accs[i][1][0]) + " Digit 2: " + str(rand_turn_accs[i][1][1]) + " Synapses: " + str(syn), fontsize=14, fontweight='bold')
	plt.savefig("./temp/learning_curve_digit_1_"+ str(rand_turn_accs[i][1][0]) + "_digit_2_"  + str(rand_turn_accs[i][1][1])  + "_synapses_" + str(syn) + ".png",bbox_inches='tight')
	plt.savefig("./temp/learning_curve_digit_1_"+ str(rand_turn_accs[i][1][0]) + "_digit_2_"  + str(rand_turn_accs[i][1][1]) + "_synapses_" + str(syn) + ".svg",bbox_inches='tight')
	plt.show()
