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


full_path_data ="/media/nikos/Elements/ClusterResults/results_250822_shapes_synapses_experiment/"
#full_path_data ="/media/nikos/Elements/ClusterResults/results/"


suffix = "_nikos_shapes_"
bad_seeds=[]

file_type ="predictons"


'''
predictons_nikos_shapes_a_b_c_d.txt

a: STABILITY: [0,1,2]
b: OVERLAP: [0,2,4]
c: RUN: [1...20]
d: TURNOVER: [0,1,2] 0: RANDOM , 1: CONSTRAINED , 2: NO


'''

#difflab=["Stability:0\nOverlap:0","Stability:0\nOverlap:2","Stability:0\nOverlap:4","Stability:1\nOverlap:0","Stability:1\nOverlap:2","Stability:1\nOverlap:4","Stability:2\nOverlap:0","Stability:2\nOverlap:2","Stability:2\nOverlap:4"]
difflab=["Easy","Medium","Hard"]
turnlab=["Turnover","No Turnover"]

turns=[0,2]
syns=[2000]
runs =list(range(1,21))
#diffic=[(0,0),(0,2),(0,4),(1,0),(1,2),(1,4),(2,0),(2,2),(2,4)]

diffic=[(2,0),(1,2),(0,4)]


##ACCURACY HEATMAPS:
turn_accs=[]
for turn in turns:
	
	for syn in syns:
		accs_list=[]
	
		for (stab,over) in diffic:
			
			accs=[]
			
			for run in runs:
				
				df=pd.read_table(full_path_data + file_type + suffix + str(stab) + "_" + str(over) + "_" + str(run) + "_" + str(turn) +"_" +str(syn) + ".txt",header=None, error_bad_lines=False, sep = " ")	
				acc=round(sum(df[2])/len(df[2]), 2)
				accs.append(acc)
			accs_list.append(accs)
		acc_table=np.array(accs_list)
		turn_accs.append((acc_table,syn))



#turn_accs[turnsyns][synstup][diffic]
syndf=2000

turn_accs_500=[ i[0] for i in turn_accs if i[1]==500]
turn_accs_1000=[ i[0] for i in turn_accs if i[1]==1000]
turn_accs_2000=[ i[0] for i in turn_accs if i[1]==2000]


dfdiff=[]
dfturn=[]
dfacc=[]

for i in range(len(diffic)):
	dfdiff+=40*[difflab[i]]
	dfturn+=20*[turnlab[0]] +20*[turnlab[1]]
	dfacc+=(100*turn_accs_2000[0][i]).tolist() + (100*turn_accs_2000[1][i]).tolist()


df_plt={"Difficulty":dfdiff,
		"Turn":dfturn,
		"% Accuracy":dfacc
		}



df_plt=pd.DataFrame(df_plt)


df_plt.to_pickle('./temp/Accuracy_Boxplots_Synapses_'+str(syndf))


ax = sns.catplot(x='Difficulty', y="% Accuracy", hue='Turn', 
            data=df_plt, kind='box', height=8,aspect = 1.5, legend=True,
            legend_out = False,dodge=True,width=0.5)
plt.suptitle('Accuracy per Difficulty across 20 runs, Synapses: ' + str(syndf))
plt.legend(loc=0, prop={'size': 10})
#plt.ylim(45,105)
plt.show()



'''
t=0
s=1
for table in turn_accs:
	
	mean_accs=np.mean(table[0],axis = 1)
	accs_std=np.std(table[0],axis = 1)
	
	
	print("Mean mean accuracy= ",100*round(np.mean(table[0]),3),"%, and mean std= ",100*round(np.mean(accs_std[0]),4),"%")
	
	
	meantable = np.zeros((3,3))
	#maxtable = np.zeros((3,3))
	#mintable = np.zeros((3,3))
	stdtable = 	np.zeros((3,3))
	
	c=0	
	for d2 in range(3):
		for d1 in range(3):	
			meantable[d1,d2] = 100*mean_accs[c]
			#maxtable[d1,d2] = 100*max_accs[c]
			#mintable[d1,d2] = 100*min_accs[c]
			stdtable[d1,d2] = 100*accs_std[c]
			c += 1
	
	plt.figure(figsize=(10, 8), dpi=80)
	ax = sns.heatmap(meantable, linewidth=0.5, cmap="inferno",annot=stdtable, vmin=0 , vmax = 100)
	ax.yaxis.tick_right()
	ax.xaxis.tick_top()
	
	ax.set_xlabel("Stability",fontsize=12,fontweight='bold')
	ax.set_ylabel("Overlap",fontsize=12,fontweight='bold')
	
	ax.set_xticklabels([0,1,2],fontsize=12,fontweight='bold')
	ax.set_yticklabels([0,2,4], rotation = 0,fontsize=12,fontweight='bold')
	#plt.title('Mean Accuracy Heatmap', fontweight= "bold",fontsize=14,loc='center')
	
	plt.suptitle('Mean Accuracy Heatmap Turnover: '+ str(t) +' Synapses: ' + str(table[1]), fontweight= "bold", x= 0.435, fontsize=18)
	plt.title('(std in boxes)', fontsize=13,fontweight='bold')
	if t == 0:
		#plt.savefig('./temp/MAH_random_turnover.eps',bbox_inches='tight',dpi=600)
		#plt.savefig('./temp/MAH_random_turnover_' + str(s*500) +'_synapses.png',bbox_inches='tight',dpi=600)
		plt.show()
	#elif t == 1:
		#plt.savefig('./temp/MAH_constrained_turnover.eps',bbox_inches='tight',dpi=600)
		#plt.savefig('./temp/MAH_constrained_turnover_' + str(s*500) +'_synapses.png',bbox_inches='tight',dpi=600)
		#plt.show()
	elif t == 2:
		#plt.savefig('./temp/MAH_no_turnover.eps',bbox_inches='tight',dpi=600)
		#plt.savefig('./temp/MAH_no_turnover_' + str(s*500) +'_synapses.png',bbox_inches='tight',dpi=600)
		plt.show() 
	s*=2
	if s==8:
		t+=2
		s=1



#ACCURACY BOXPLOTS
turn_accs_1000=[]
turn_accs_2000=[]
turn_accs_500=[]
for i in turn_accs:
	if i[1]==1000:
		turn_accs_1000.append(i[0])
	elif i[1]==2000:
		turn_accs_2000.append(i[0])
	else:
		turn_accs_500.append(i[0])

s=500
turnticks=[1,2]
for turnlist in [turn_accs_500,turn_accs_1000,turn_accs_2000]:
	
	accs_3d_arr_turn_diff_runs=np.array(turnlist)
	mean_accs_2d_arr_turn_diff = np.mean(accs_3d_arr_turn_diff_runs, axis=2)
	plt.figure(figsize=(10, 8), dpi=80)
	for i in range(len(mean_accs_2d_arr_turn_diff[0])):
		plt.plot(turnticks, 100*mean_accs_2d_arr_turn_diff[:,i], 'ko-',ms=10, alpha=1)
	plt.boxplot(100*mean_accs_2d_arr_turn_diff.T)
	#plt.yticks(np.arange(50,110,10),fontsize=10,fontweight="bold")
	plt.ylabel('% Accuracy', fontweight = 'bold',fontsize=12)
	#plt.ylim(-1,110)
	plt.xticks(turnticks,["Random\nTurnover","No\nTurnover"],fontweight="bold",fontsize=12)
	plt.title("Mean Accuracies across Turnovers, "+"Synapses:"+str(s), fontsize=14, fontweight='bold')	
	#plt.savefig("./temp/Mean_Dots_Shapes_Synapses_"+ str(s) +".png",bbox_inches='tight')
	plt.show()
	
	
	std_accs_2d_arr_turn_diff = np.std(accs_3d_arr_turn_diff_runs, axis=2)
	plt.figure(figsize=(10, 8), dpi=80)
	for i in range(len(std_accs_2d_arr_turn_diff[0])):
		plt.plot(turnticks, 100*std_accs_2d_arr_turn_diff[:,i], 'ko-',ms=10, alpha=1)
		plt.boxplot(100*std_accs_2d_arr_turn_diff.T)
	#plt.yticks(np.arange(0,20,1),fontsize=10,fontweight="bold")
	plt.ylabel('% Accuracy', fontweight = 'bold',fontsize=12)
	#plt.ylim(-1,20)
	plt.xticks(turnticks,["Random\nTurnover","No\nTurnover"],fontweight="bold",fontsize=12)
	plt.title("Mean Standard Deviations across Turnovers, " + "Synapses:"+str(s), fontsize=14, fontweight='bold')	
	#plt.savefig("./temp/Std_Dots_Shapes_Synapses_"+ str(s) +".png",bbox_inches='tight')
	plt.show()
	s*=2



'''






#LEARNING CURVES
file_type ="sample_accuracies"
sample_interval=20
syn=2000


rand_turn_accs=[]
#cons_turn_accs=[]
no_turn_accs=[]

rand_turn_accs_mean=[]
#cons_turn_accs_mean=[]
no_turn_accs_mean=[]

rand_turn_accs_std=[]
#cons_turn_accs_std=[]
no_turn_accs_std=[]

#all_convs=[]

for turn in turns:
	#convs_diff=[]
	for (stab,over) in diffic:
		
		#convs=[]
		accs=[]
		for run in runs:
			
			df=pd.read_table(full_path_data + file_type + suffix + str(stab) + "_" + str(over) + "_" + str(run) + "_" + str(turn) +"_" + str(syn) + ".txt",header=None, error_bad_lines=False, sep = " ")	
			#THIS IS FOR INDIVIDUAL PLOTS PER DIGIT PAIR
			#REQUIRES FILE TYPE = sample_accuracies !
		
			y=[i for i in df[0]]
			y.insert(0,0)
			
			
			
			#ycon=gaussian_filter1d(y,0.1)
			#xcon=list(range(sample_interval*0,sample_interval*len(ycon),sample_interval))			
			
			#poli=np.polyfit(xcon, ycon, 6)
			#curve=np.poly1d(poli)(xcon)
			
			#x = Symbol('x')
			
			#f = poli[0]*x**6 + poli[1]*x**5 + poli[2]*x**4 + poli[3]*x**3 + poli[4]*x**2 + poli[5]*x**1 + poli[6]
			#delta_f = f.diff(x)
			
			#delta_f=lambdify(x,delta_f)
			
			#dycon=np.array([(delta_f(i)) for i in xcon])
			#conv=xcon[np.where(dycon[2:]<0.2)[0][0]+2]
			
			
			#plt.figure(figsize=(10, 8), dpi=80)
			#plt.plot(xcon,y)
			#plt.plot(xcon,ycon)
			#plt.scatter(conv,ycon[np.where(dycon[2:]<0.2)[0][0]+2],c='k')

			#convs.append(conv)
			
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
			rand_turn_accs.append((accsarr,(stab,over)))
		#elif turn==1:
		#	cons_turn_accs_mean.append(meanarr)
		#	cons_turn_accs_std.append(stdarr)	
		else:
			no_turn_accs_mean.append(meanarr)
			no_turn_accs_std.append(stdarr)
			no_turn_accs.append((accsarr,(stab,over)))
		
		
		#convs_diff.append(convs)
		
		'''
		x = list(range(sample_interval*0,sample_interval*maxlen,sample_interval))				
		plt.figure(figsize=(10, 10), dpi=80)
		plt.plot(x,meanarr,"r")
		#plt.fill_between(x, np.clip((meanarr + stdarr),0,100), np.clip((meanarr - stdarr),0,100),alpha=0.3,facecolor='r')
		plt.fill_between(x, meanarr + stdarr, np.clip((meanarr - stdarr),0,100),alpha=0.3,facecolor='r')
		plt.xlabel('Training Iterations', fontweight = 'bold')
		plt.yticks(np.arange(0,110,10))
		plt.ylabel('% Accuracy', fontweight = 'bold')
		plt.title('Learning Curve Stability: '+ str(stab) + " Overlap: " + str(over) + " Turnover: " + str(turn) +" Synapses: " +str(syn), fontsize=14, fontweight='bold')
		plt.show()
		#plt.savefig("./temp/learning_curves_"+digit1+"_"+digit2,bbox_inches='tight')
		'''

	#all_convs.append(convs_diff)

plt.close("all")
#x = list(range(sample_interval*0,sample_interval*18,sample_interval))


rand_turn_accs=np.array(rand_turn_accs,dtype=object)
#cons_turn_accs=np.array(cons_turn_accs,dtype=object)
no_turn_accs=np.array(no_turn_accs,dtype=object)


np.save('./temp/Accuracy_Table_For_Learning_Curves_Turnover_'+str(syn),rand_turn_accs,allow_pickle=True)
#np.save('./temp/Accuracy_Table_For_Learning_Curves_Constrained_Turnover_'+str(syn),cons_turn_accs,allow_pickle=True)
np.save('./temp/Accuracy_Table_For_Learning_Curves_No_Turnover_'+str(syn),no_turn_accs,allow_pickle=True)





rt = "r"
#ct = "b"	
nt = "b"	

	
for i in range(len(rand_turn_accs_mean)):
		
	plt.figure(figsize=(10, 10), dpi=80)
	xr=list(range(sample_interval*0,sample_interval*len(rand_turn_accs_mean[i]),sample_interval))
	#xc=list(range(sample_interval*0,sample_interval*len(cons_turn_accs_mean[i]),sample_interval))
	xn=list(range(sample_interval*0,sample_interval*len(no_turn_accs_mean[i]),sample_interval))
	plt.plot(xr,rand_turn_accs_mean[i],rt)
	#plt.plot(xc,cons_turn_accs_mean[i],"b")	
	plt.plot(xn,no_turn_accs_mean[i],nt)
	
	

	'''
	#yn=gaussian_filter1d(no_turn_accs_mean[i],0.1)
	ncurve=np.poly1d(np.polyfit(xn, no_turn_accs_mean[i], 7))(xn)
	
	poli=np.polyfit(xn, no_turn_accs_mean[i], 7)
	curve=np.poly1d(poli)(xn)
	
	x = Symbol('x')
	
	
	f = poli[0]*x**7 + poli[1]*x**6 + poli[2]*x**5 + poli[3]*x**4 + poli[4]*x**3 + poli[5]*x**2 + poli[6]*x**1 + poli[7]
	#f = poli[0]*x**6 + poli[1]*x**5 + poli[2]*x**4 + poli[3]*x**3 + poli[4]*x**2 + poli[5]*x**1 + poli[6]
	delta_f = f.diff(x)
	
	f=lambdify(x,f)
	delta_f=lambdify(x,delta_f)
		
	ynn=[f(i) for i in xn]
	dyn=np.array([(delta_f(i)) for i in xn])
	

	
	#plt.plot(xn,yn)
	plt.plot(xn,ncurve)	
	
	plt.scatter(xn[np.where(dyn[1:]<0.05)[0][0]+1],ncurve[np.where(dyn[1:]<0.05)[0][0]+1],c='k')


	
	#print(xn[np.where(dyn[1:]<0.05)[0][0]])
	#print((dyn<0.05))
	#print(min(abs(dyn)))
	

	#yr=gaussian_filter1d(rand_turn_accs_mean[i],0.5)
	rcurve=np.poly1d(np.polyfit(xr, rand_turn_accs_mean[i], 7))(xr)

	poli=np.polyfit(xr, rand_turn_accs_mean[i], 7)
	curve=np.poly1d(poli)(xn)
	
	x = Symbol('x')
	
	f = poli[0]*x**7 + poli[1]*x**6 + poli[2]*x**5 + poli[3]*x**4 + poli[4]*x**3 + poli[5]*x**2 + poli[6]*x**1 + poli[7]
	#f = poli[0]*x**6 + poli[1]*x**5 + poli[2]*x**4 + poli[3]*x**3 + poli[4]*x**2 + poli[5]*x**1 + poli[6]
	delta_f = f.diff(x)
	
	f=lambdify(x,f)
	delta_f=lambdify(x,delta_f)
		
	yrr=[f(i) for i in xr]
	dyr=np.array([(delta_f(i)) for i in xr])


	#plt.plot(xr,yr)	
	
	plt.plot(xr,rcurve)
	plt.scatter(xr[np.where(dyr[1:]<0.05)[0][0]+1],rcurve[np.where(dyr[1:]<0.05)[0][0]+1],c='k')
	'''


	plt.fill_between(xr, np.clip((rand_turn_accs_mean[i] + rand_turn_accs_std[i]),0,100), np.clip((rand_turn_accs_mean[i] - rand_turn_accs_std[i]),0,100),alpha=0.3,facecolor=rt)
	#plt.fill_between(xc, cons_turn_accs_mean[i] + cons_turn_accs_std[i], np.clip((cons_turn_accs_mean[i] - cons_turn_accs_std[i]),0,100),alpha=0.3,facecolor='b')
	plt.fill_between(xn, np.clip((no_turn_accs_mean[i] + no_turn_accs_std[i]),0,100), np.clip((no_turn_accs_mean[i] - no_turn_accs_std[i]),0,100),alpha=0.3,facecolor=nt)
	
	
	handles = [Rectangle((0,0),1,1,color=c,ec="k") for c in [rt,nt]]
	labels= ["Turnover","No Turnover"]
	plt.legend(handles, labels,fontsize=12,prop={'weight':'bold','size':12},loc="lower right")
	plt.xlabel('Training Iterations', fontweight = 'bold')
	plt.yticks(np.arange(0,110,10))
	plt.ylabel('% Accuracy', fontweight = 'bold')
	plt.title('Learning Curve Stability: '+ str(diffic[i][0]) + " Overlap: " + str(diffic[i][1]) + " Synapses: " + str(syn), fontsize=14, fontweight='bold')
	plt.savefig("./temp/learning_curve_stability_"+ str(diffic[i][0]) + "_overlap_"  + str(diffic[i][1])  + "_synapses_" + str(syn) + ".png",bbox_inches='tight')
	#plt.savefig("./temp/learning_curve_stability_"+ str(diffic[i][0]) + "_overlap_"  + str(diffic[i][1]) + "_synapses_" + str(syn) + ".svg",bbox_inches='tight')
	plt.show()























'''
#CONVERGENCE	
gaussian_filter1d(meanarr,1)


yg=gaussian_filter1d(no_turn_accs_mean[i],1)
xg=xn

poli=np.polyfit(xg, yg, 6)
curve=np.poly1d(poli)(xg)

x = Symbol('x')


f = poli[0]*x**6 + poli[1]*x**5 + poli[2]*x**4 + poli[3]*x**3 + poli[4]*x**2 + poli[5]*x**1 + poli[6]
delta_f = f.diff(x)

f=lambdify(x,f)
y=[f(i) for i in xg]

delta_f=lambdify(x,delta_f)
dy=np.array([(delta_f(i)) for i in xg])




xg[np.argmin(abs(dy))]

min(abs(dy))

plt.plot(xg,y)
plt.plot(xg,curve)
plt.plot(100*[xg[np.argmin(abs(dy))]],range(0,100),'k-')


plt.plot(xg,dy)

'''


















#WEIGHTS PER EPOCH (ONLY LAST ITERATION BOXPLOTS)
file_type_1 ="weights_per_epoch"
syn=2000


#diffic=[(0,4),(1,2),(2,0)]
#diffic=[(0,0),(0,2),(0,4),(1,0),(1,2),(1,4),(2,0),(2,2),(2,4)]


thres=0.3
step=10
eff_weights_turns=[]
for turn in turns:
	eff_weights_diff=[]
	for (stab,over) in diffic:		
		eff_weights_runs=[]
		for run in runs:

			
			# Input
			weight_file_1=full_path_data + file_type_1 + suffix + str(stab) + "_" + str(over) + "_" + str(run) + "_" + str(turn) +"_" +str(syn) + ".txt"
							
			
			# Delimiter
			data_file_delimiter = '\t'
			
			# The max column count a line in the file could have
			largest_column_count = 0
			
			# Loop the data lines
			with open(weight_file_1, 'r') as temp_f:
			    # Read the lines
			    lines = temp_f.readlines()
			
			    for l in lines:
			        # Count the column count for the current line
			        column_count = len(l.split(data_file_delimiter)) + 1
			        
			        # Set the new most column count
			        largest_column_count = column_count if largest_column_count < column_count else largest_column_count
			
			# Generate column names (will be 0, 1, 2, ..., largest_column_count - 1)
			column_names = [i for i in range(0, largest_column_count)]
			
			# Read csv
			df = pd.read_csv(weight_file_1, header=None, delimiter=data_file_delimiter, names=column_names)
			



			
			eff_weights=[]

			#for i in range(0,len(df),step):
			#	x=df.values[i]
			#	eff_weights.append(np.sum(x>thres) / np.sum(np.isnan(x)==False))
			x=df.values[-1]
			eff_weights.append(np.sum(x>thres) / np.sum(np.isnan(x)==False))
			
			
			#x=np.sort(x)
			#calculate CDF values
			#y = 1. * np.arange(len(x)) / (len(x) - 1)	
			#plot CDF
			#plt.plot(x, y)	
				
				
			'''
				plt.figure(figsize=(10, 10), dpi=80)
				plt.hist(x,100,range=(0,1),ec="k",alpha=1,color="mediumslateblue")
				labels= ["Subpopulation 1","Subpopulation 2"]
				plt.ylim(0,100)
				plt.xlim(0,1)
				plt.xlabel('Weight Value',fontweight='bold',fontsize=12)
				plt.ylabel("Counts",fontweight='bold',fontsize=12)
				plt.title('Weight Distribution per Subpopulation ' + str(i) + ' iterations',fontweight='bold',fontsize=15)
				#plt.savefig("./temp/Weights" + str(i),bbox_inches='tight')
				plt.show()
			'''
				
			
			eff_weights_runs.append(eff_weights)
		
		#maxlen=max([len(i) for i in eff_weights_runs])
		#c=1
		#while c!=0:
		#	c=0
		#	for i in eff_weights_runs:
		#		if len(i)!=maxlen:
		#			c+=1
		#			i.append(np.nan)
						
						
						
						
						
		eff_weights_diff.append(eff_weights_runs)
	eff_weights_turns.append(eff_weights_diff)


#L1 diff L2 turn, Acc Weight Perc





eff_weights_dict_no_turn={}
eff_weights_dict_rand_turn={}


for i in range(len(diffic)):
	eff_weights_dict_no_turn[diffic[i]] = (100*np.array(eff_weights_turns[1][i]).reshape(1,20)).tolist()[0]
	eff_weights_dict_rand_turn[diffic[i]] = (100*np.array(eff_weights_turns[0][i]).reshape(1,20)).tolist()[0]
		




#difflab=["Stability:0\nOverlap:4","Stability:1\nOverlap:2","Stability:2\nOverlap:0"]
#difflab=["Hard","Medium","Easy"]
turnlab=["Turnover","No Turnover"]



dfdiff=[]
dfturn=[]
dfwperc=[]
for i in range(len(diffic)):
	dfdiff+=40*[difflab[i]]
	dfturn+=20*[turnlab[0]] +20*[turnlab[1]]
	dfwperc+=eff_weights_dict_rand_turn[diffic[i]] + eff_weights_dict_no_turn[diffic[i]]


df_plt={"Difficulty":dfdiff,
		"Turn":dfturn,
		"Effective Weight %":dfwperc
		}



df_plt=pd.DataFrame(df_plt)


df_plt.to_pickle('./temp/effective_weights_synapses_'+str(syn))



ax = sns.catplot(x='Difficulty', y='Effective Weight %', hue='Turn', 
            data=df_plt, kind='box', height=8,aspect = 1.5, legend=True,
            legend_out = False,dodge=True,width=0.5)
plt.suptitle('Effective Weights, Synapses:' + str(syn))
plt.legend(loc=0, prop={'size': 10})






















'''
rt = "r"
#ct = "b"	
nt = "b"	


for i in diffic:
	eff_mean_rand=np.nanmean(eff_weights_dict_rand_turn[i], axis=0)*100
	eff_std_rand=np.nanstd(eff_weights_dict_rand_turn[i], axis=0)*100
	
	eff_mean_no=np.nanmean(eff_weights_dict_no_turn[i], axis=0)*100
	eff_std_no=np.nanstd(eff_weights_dict_no_turn[i], axis=0)*100
	
	xr=list(range(step*0,step*len(eff_mean_rand),step))
	#xc=list(range(sample_interval*0,sample_interval*len(cons_turn_accs_mean[i]),sample_interval))
	xn=list(range(step*0,step*len(eff_mean_no),step))

	plt.figure(figsize=(10, 10), dpi=80)
	plt.xlabel('Training Iterations',fontweight='bold',fontsize=12)
	plt.ylabel("Effective Weight % (> "+str(thres)+")",fontweight='bold',fontsize=12)
	plt.title('Effective Weights per Training Iteration Stability: ' + str(i[0]) + " Overlap: " + str(i[1]),fontweight='bold',fontsize=15)
	plt.plot(xr, eff_mean_rand,c=rt)
	plt.plot(xn, eff_mean_no,c=nt)

	plt.fill_between(xr, np.clip((eff_mean_rand + eff_std_rand),0,100), np.clip((eff_mean_rand - eff_std_rand),0,100),alpha=0.3,facecolor=rt)
	plt.fill_between(xn, np.clip((eff_mean_no + eff_std_no),0,100), np.clip((eff_mean_no - eff_std_no),0,100),alpha=0.3,facecolor=nt)
	
	
	handles = [Rectangle((0,0),1,1,color=c,ec="k") for c in [rt,nt]]
	labels= ["Random Turnover","No Turnover"]
	plt.legend(handles, labels,fontsize=12,prop={'weight':'bold','size':12},loc="lower right")
	plt.xlabel('Training Iterations', fontweight = 'bold')
	plt.yticks(np.arange(0,110,10))
	#plt.savefig("./temp/Eff_Weights_0.3_Stability_" + str(i[0]) + "_Overlap_" + str(i[1]) +".eps",bbox_inches='tight')
	#plt.savefig("./temp/Eff_Weights_" + str(thres) + "_Stability_" + str(i[0]) + "_Overlap_" + str(i[1]) +".png",bbox_inches='tight')
	plt.show()
'''		
	












'''


#WEIGHTS PER EPOCH PER SUB
file_type_1 ="weights_per_epoch_sub1"
file_type_2 ="weights_per_epoch_sub2"
suffix = "nikos_shapes_"
syn=1000
stab=1
over=4
run=13
turn=0





# Input
weight_file_1=full_path_data + file_type_1 + suffix + str(stab) + "_" + str(over) + "_" + str(run) + "_" + str(turn) +"_" +str(syn) + ".txt"
				

# Delimiter
data_file_delimiter = '\t'

# The max column count a line in the file could have
largest_column_count = 0

# Loop the data lines
with open(weight_file_1, 'r') as temp_f:
    # Read the lines
    lines = temp_f.readlines()

    for l in lines:
        # Count the column count for the current line
        column_count = len(l.split(data_file_delimiter)) + 1
        
        # Set the new most column count
        largest_column_count = column_count if largest_column_count < column_count else largest_column_count

# Generate column names (will be 0, 1, 2, ..., largest_column_count - 1)
column_names = [i for i in range(0, largest_column_count)]

# Read csv
df = pd.read_csv(weight_file_1, header=None, delimiter=data_file_delimiter, names=column_names)



# Input
weight_file_2=full_path_data + file_type_2 + suffix + str(stab) + "_" + str(over) + "_" + str(run) + "_" + str(turn) +"_" +str(syn) + ".txt"
	

# The max column count a line in the file could have
largest_column_count = 0

# Loop the data lines
with open(weight_file_2, 'r') as temp_f:
    # Read the lines
    lines = temp_f.readlines()

    for l in lines:
        # Count the column count for the current line
        column_count = len(l.split(data_file_delimiter)) + 1
        
        # Set the new most column count
        largest_column_count = column_count if largest_column_count < column_count else largest_column_count

# Generate column names (will be 0, 1, 2, ..., largest_column_count - 1)
column_names = [i for i in range(0, largest_column_count)]

# Read csv
df2 = pd.read_csv(weight_file_2, header=None, delimiter=data_file_delimiter, names=column_names)
# print(df)




sub1 = "mediumslateblue"
sub2 = "firebrick"

eff_weights_sub1=[]
eff_weights_sub2=[]
for i in range(0,len(df2),20):
	#x0=df0.values[i]	
	x1=df.values[i]
	x2=df2.values[i]
	
	
	
	eff_weights_sub1.append(np.sum(x1>0.2) / np.sum(np.isnan(x1)==False))
	eff_weights_sub2.append(np.sum(x2>0.2) / np.sum(np.isnan(x2)==False))
	
	plt.figure(figsize=(10, 10), dpi=80)
	plt.grid(True)
	#plt.hist(x0,100,range=(0,1),ec="k")
	plt.hist(x1,100,range=(0,1),ec="k",alpha=1,color="mediumslateblue")
	plt.hist(x2,100,range=(0,1),ec="k",alpha=0.7,color="firebrick")
	handles = [Rectangle((0,0),1,1,color=c,ec="k") for c in [sub1,sub2]]
	labels= ["Subpopulation 1","Subpopulation 2"]
	plt.legend(handles, labels,fontsize=12,prop={'weight':'bold','size':12})
	plt.ylim(0,100)
	plt.xlim(0,1)
	plt.xlabel('Weight Value',fontweight='bold',fontsize=12)
	plt.ylabel("Counts",fontweight='bold',fontsize=12)
	plt.title('Weight Distribution per Subpopulation ' + str(i) + ' iterations',fontweight='bold',fontsize=15)
	#plt.savefig("./temp/Weights" + str(i),bbox_inches='tight')
	plt.show()
	















'''









