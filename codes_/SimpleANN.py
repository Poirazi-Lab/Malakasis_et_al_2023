from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from mnist import MNIST
import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
import seaborn as sns

warnings.simplefilter("ignore", category=ConvergenceWarning)



mndata=MNIST('./MNIST/')
X_train,y_train = mndata.load_training()

X_test,y_test = mndata.load_testing()

y_test=np.array(y_test)


meantable = np.zeros((10,10))
stdtable = np.zeros((10,10))

meanlist=[]
stdlist=[]

for c2 in range(10):
	meantable[c2,c2]=np.nan
	stdtable[c2,c2]=np.nan
	for c1 in range(c2):
		print("Running Pair" ,c1,c2)
		class_1=c1
		class_2=c2

		X_tr=[]
		y_tr=[]
		
		for i in range(len(y_train)):
			if y_train[i]==class_1 or y_train[i]==class_2:
				X_tr.append(X_train[i])
				y_tr.append(y_train[i])
		
		
		X_t=[]
		y_t=[]
		
		for i in range(len(y_test)):
			if y_test[i]==class_1 or y_test[i]==class_2:
				X_t.append(X_test[i])
				y_t.append(y_test[i])
		
		
		
		scores=[]
		hidden=2
		
		
		
		
		
		for i in range(20):
			np.random.seed(i)
			
			
			idx_tr = np.random.permutation(len(X_tr))
			X_tr=np.array(X_tr)[idx_tr]
			y_tr=np.array(y_tr)[idx_tr]
		
			train=np.array(X_tr[:200])
			labels=np.array(y_tr[:200])	
			
#			idx = np.random.permutation(len(train))
#			train=train[idx]
#			labels=labels[idx]
		
			clf = MLPClassifier(hidden_layer_sizes=(hidden),solver="adam",activation='relu', learning_rate_init=0.01, batch_size=1, random_state=i, max_iter=1).fit(train, labels)
			
			#clf.predict_proba(X_test[:10])
			
			clf.predict(X_t)
			print(clf.score(X_t, y_t))
			scores.append(clf.score(X_t, y_t))
		
		meanscore=100*round(np.mean(scores),3)
		stdscore=100*round(np.std(scores),3)
		
		print("Mean Acc:",meanscore,", ","Std:",stdscore)
		
		meanlist.append(meanscore)
		stdlist.append(stdscore)
		meantable[c1,c2]=meanscore
		stdtable[c1,c2]=stdscore
		meantable[c2,c1]=np.nan
		stdtable[c2,c1]=np.nan
		
		
print("Trainable Params:",784*hidden + hidden*2 + 1)

plt.figure(figsize=(10, 8), dpi=80)
ax = sns.heatmap(meantable, linewidth=0.5, cmap="inferno",annot=stdtable, vmin=50 , vmax = 100)
ax.yaxis.tick_right()
ax.xaxis.tick_top()


ax.set_xticklabels([0,1,2,3,4,5,6,7,8,9],fontsize=12,fontweight='bold')
ax.set_yticklabels([0,1,2,3,4,5,6,7,8,9], rotation = 0,fontsize=12,fontweight='bold')
plt.title('Mean Accuracy Heatmap', fontweight= "bold",fontsize=14,loc='center')

plt.suptitle('Mean Accuracy Heatmap', fontweight= "bold", x= 0.435, fontsize=18)
plt.title('(std in boxes)', fontsize=13,fontweight='bold')

plt.show()
		
'''
for i in range(10):
	for j in range(i):
		meanlist.append(meantable[j,i])
		stdlist.append(stdtable[j,i])
'''

print("Mean mean accuracy:",np.mean(meanlist),". Mean Standard Deviation:",np.mean(stdlist))
