import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_table("./results/weights_per_epoch_asdf.txt",header=None, error_bad_lines=False)
#df=df.drop(df.columns[[len(df.values[0])-1]],axis=1)




for i in range(0,len(df.values),10):	
	x=df.values[i]
	plt.figure(figsize=(10, 10), dpi=80)
	plt.grid(True)
	plt.hist(x,100,range=(0,1),ec="k",color='darkorchid')
	plt.ylim(0,200)
	plt.xlabel('Weight Value',fontweight='bold',fontsize=12)
	plt.ylabel("Counts",fontweight='bold',fontsize=12)
	plt.title('Total Weight Distribution ' + str(i) + ' iterations',fontweight='bold',fontsize=15)
	#plt.savefig("./temp/JustWeights" + str(i))
	plt.show()



'''
a=[0,61,68,67,62,56,60,55]
b=[0,50,100,150,200,250,300,350]

plt.figure()
plt.grid(True)
plt.xlabel('Training Iterations')
plt.yticks(range(0,100,10))
plt.ylabel('% Accuracy')
plt.title('Learning Curve')
plt.plot(b,a)



X=[]
Y=[]
for x in range(0,8280,60):	
	y=(x>1200.)*((x-20.*60.)/(30.*60)) * np.exp(1. - (x-20.*60. )/(30.*60.))
	X.append(x)
	Y.append(y)
plt.plot(X,Y)
plt.show()
plt.close('all')



X=[]
Y=[]
for x in range(7200):	
	y=((x)/(15.*60)) * np.exp(1. - (x )/(15.*60.))
	X.append(x)
	Y.append(y)
plt.plot(X,Y)
plt.show()
plt.close('all')

14400/3600

Y[-1]
1200/60

ca=0.3
(1.3/(1.+np.exp(-(ca*10.-3.5)*10.))) - (0.3/(1.+np.exp(-(ca*10.0-2.0)*19.)))

cal=[]
tag=[]

for ca in np.arange(0,1,0.01):
	cal.append(ca)
	tag.append((2/(1.+np.exp(-(ca*10.-3.5)*10.))) - (1/(1.+np.exp(-(ca*10.0-2.0)*19.))))

plt.plot(cal,tag)


stag=10
ll=[10]
for i in range(0,8221,60):
	stag-=stag/(180-178)
	ll.append((stag,i))


8280/3600

7200/3600

138/60


wadapt=1
creb=1
l=[1]
ll=[1]
for i in range(0,4000):
	creb-=60/(3600*8)
	wadapt-=wadapt/1
	ll.append((creb,i))
	l.append((wadapt,i))
	
	
l[:100]
	


'''