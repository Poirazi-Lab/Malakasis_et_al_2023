import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from mnist import MNIST
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score
import csv




pca = PCA(n_components=3)

'''
mndata=MNIST('./MNIST/')
imagelists,labels=mndata.load_training()


imagelists=np.array(imagelists)
labels=np.array(labels)



zeros=imagelists[labels==0]
yz=np.ones(len(zeros))
ones=imagelists[labels==1]
yo=2*np.ones(len(ones))


both=np.vstack((zeros,ones))	
yb=np.hstack((yz,yo))

PCAzeros=pca.fit_transform(both)



plt.scatter(PCAzeros[:,0],PCAzeros[:,1],c=yb)
plt.show()

#3D SCATTER
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")

ax.scatter3D(PCAzeros[:,0],PCAzeros[:,1],PCAzeros[:,2],c=yb)
plt.show()




X,y = make_classification(n_samples=10000, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=1, scale=1, shift=0, flip_y=0, weights=[0.5,0.5], random_state=1,class_sep=5,hypercube=True)

plt.scatter(X[:,0],X[:,1], c=y)
plt.show()



X,y=make_blobs(n_samples=6000,n_features=784,centers=2,center_box=(19,45),cluster_std=150,random_state=11)
X=X.astype(int)

pca = PCA(n_components=2)
XX=pca.fit_transform(X)

plt.scatter(XX[:,0],XX[:,1], c=y)
plt.show()
'''


#Make multivariate gaussian from numpy (with cov=I at first)
#Dims=500, play wih mean difference and variance.


#Create Train set
np.random.seed(12)




varval1=100
#meanval1=1

varval2=100
#meanval2=1

dim=400
train_samples=1000
test_samples=200
tot_samples=train_samples+test_samples
step=1


meanvec=np.arange(10,int(dim)+10,1)/step

idx = np.random.permutation(len(meanvec))
idx2 = np.random.permutation(len(meanvec))


meanvec1=meanvec[idx]
meanvec2=meanvec[idx2]

varvec1=varval1*np.ones(dim)
varvec2=varval2*np.ones(dim)


zeros1=np.random.permutation(np.arange(0,dim,1))[:0]
#zeros2=np.random.permutation(np.arange(0,dim))[:350]



X1=np.random.multivariate_normal(meanvec1, varvec1*np.eye(dim),tot_samples)
y1=np.ones(train_samples).astype(int)
y1t=np.ones(test_samples).astype(int)

X1=X1-np.min(X1)
X1=255*X1/np.max(X1)
X1=X1.astype(int)

for i in X1:
	i[zeros1]=0


X1t=X1[:test_samples]
X1=X1[test_samples:]


X2=np.random.multivariate_normal(meanvec2, varvec2*np.eye(dim),tot_samples)
y2=2*np.ones(train_samples).astype(int)
y2t=2*np.ones(test_samples).astype(int)

X2=X2-np.min(X2)
X2=255*X2/np.max(X2)
X2=X2.astype(int)

for i in X2:
	i[zeros1]=0


X2t=X2[:test_samples]
X2=X2[test_samples:]

y=np.hstack((y1,y2))
X=np.vstack((X1,X2))


Xa=[]
ya=[]
for i in range(train_samples):
	Xa.append(X1[i])
	Xa.append(X2[i])
	ya.append(y1[i])
	ya.append(y2[i])
	
	
	
X=np.array(Xa)
y=ya


print(np.min(X),np.max(X))



plt.scatter(X[:,0],X[:,1],c=y)
plt.xlabel("Dimension 1", fontweight='bold')
plt.ylabel("Dimension 2", fontweight='bold')
plt.show()


XP=pca.fit_transform(X)

pca.explained_variance_ratio_
plt.figure()
plt.scatter(XP[:,0],XP[:,1],c=y)
plt.title("PCA", fontweight='bold', fontsize=15)
plt.xlabel("PC 1", fontweight='bold')
plt.ylabel("PC 2", fontweight='bold')
plt.xlim((-1400,1400))
plt.ylim((-100,100))
plt.show()
#plt.savefig("./temp/PCA2D")


#3D SCATTER
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
ax.set_xlabel('PC 1', fontweight ='bold')
ax.set_ylabel('PC 2', fontweight ='bold')
ax.set_zlabel('PC 3', fontweight ='bold')
ax.set_xlim((-1500,1500))
ax.set_ylim((-100,100))
ax.set_zlim((-100,100))
ax.scatter3D(XP[:,0],XP[:,1],XP[:,2],c=y)
plt.show()
#plt.savefig("./temp/PCA3D")



asd=np.reshape(X[0],(int(np.sqrt(dim)),int(np.sqrt(dim))))
plt.figure()
plt.imshow(asd,cmap=plt.cm.binary_r)
plt.show()
#plt.savefig("./temp/Class_1")


asd=np.reshape(X[1],(int(np.sqrt(dim)),int(np.sqrt(dim))))
plt.figure()
plt.imshow(asd,cmap=plt.cm.binary_r)
plt.show()
#plt.savefig("./temp/Class_2")


with open('Blobs_train_labels', 'w+') as f:
    # create the csv writer
    writer = csv.writer(f)

    # write a row to the csv file
    writer.writerow(y)

with open('Blobs_train', 'w+') as ff:
    # create the csv writer
	writer = csv.writer(ff)

    # write a row to the csv file
	for i in range(len(X)):
		writer.writerow(X[i])

##DBI
print("DBI for train:",davies_bouldin_score(X,y))



yt=np.hstack((y1t,y2t))

Xt=np.vstack((X1t,X2t))


print(np.min(Xt),np.max(Xt))

idx_tr = np.random.permutation(len(Xt))

Xt=Xt[idx_tr]
yt=yt[idx_tr]

plt.figure()
plt.scatter(Xt[:,0],Xt[:,1],c=yt)
plt.xlabel("Dimension 1", fontweight='bold')
plt.ylabel("Dimension 2", fontweight='bold')
plt.show()


with open('Blobs_test_labels', 'w+') as f:
    # create the csv writer
    writer = csv.writer(f)

    # write a row to the csv file
    writer.writerow(yt)

with open('Blobs_test', 'w+') as ff:
    # create the csv writer
	writer = csv.writer(ff)

    # write a row to the csv file
	for i in range(len(Xt)):
		writer.writerow(Xt[i])







XP=pca.fit_transform(Xt)

pca.explained_variance_ratio_
plt.figure()
plt.scatter(XP[:,0],XP[:,1],c=yt)
plt.title("PCA", fontweight='bold', fontsize=15)
plt.xlabel("PC 1", fontweight='bold')
plt.ylabel("PC 2", fontweight='bold')
plt.xlim((-1400,1400))
plt.ylim((-100,100))
plt.show()



#3D SCATTER
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")

ax.scatter3D(XP[:,0],XP[:,1],XP[:,2],c=yt)
ax.set_xlabel('PC 1', fontweight ='bold')
ax.set_ylabel('PC 2', fontweight ='bold')
ax.set_zlabel('PC 3', fontweight ='bold')
ax.set_xlim((-1500,1500))
ax.set_ylim((-100,100))
ax.set_zlim((-100,100))
plt.show()




#np.mean(zeros[0])
asd=np.reshape(Xt[0],(int(np.sqrt(dim)),int(np.sqrt(dim))))
plt.figure()
plt.imshow(asd,cmap=plt.cm.binary_r)
plt.show()

asd=np.reshape(Xt[2],(int(np.sqrt(dim)),int(np.sqrt(dim))))
plt.figure()
plt.imshow(asd,cmap=plt.cm.binary_r)
plt.show()



print("DBI for test:",davies_bouldin_score(Xt,yt))

'''

#Playing Around with idea about dataset

test=np.zeros((20,20))

n1=np.random.randint(3,16) #n in 3,16
n2=np.random.randint(3,16) #n in 3,16

m1=np.random.randint(3,16) #m in 3,16
m2=np.random.randint(3,16) #n in 3,16

test[n1-3,m1]=2
test[n1-2,m1-1:m1+2]=3
test[n1-1,m1-2:m1+3]=4
test[n1,m1-3:m1+4]=5
test[n1+1,m1-2:m1+3]=4
test[n1+2,m1-1:m1+2]=3
test[n1+3,m1]=2


plt.figure()
plt.imshow(test,cmap=plt.cm.binary_r)
plt.show()

test=np.zeros((20,20))

test[n2-2,m2-2:m2+3]=3
test[n2-1,m2-2:m2+3]=4
test[n2,m2-2:m2+3]=5
test[n2+1,m2-2:m2+3]=4
test[n2+2,m2-2:m2+3]=3


plt.figure()
plt.imshow(test,cmap=plt.cm.binary_r)
plt.show()



Xtot=np.vstack((X,Xt))
ytot=np.hstack((y,yt))


plt.figure()
plt.scatter(Xtot[:,0],Xtot[:,1],c=ytot)
plt.xlabel("Dimension 1", fontweight='bold')
plt.ylabel("Dimension 2", fontweight='bold')
plt.show()



XP=pca.fit_transform(Xtot)

pca.explained_variance_ratio_
plt.figure()
plt.scatter(XP[:,0],XP[:,1],c=ytot)
plt.title("PCA", fontweight='bold', fontsize=15)
plt.xlabel("PC 1", fontweight='bold')
plt.ylabel("PC 2", fontweight='bold')
plt.xlim((-1400,1400))
plt.ylim((-100,100))
plt.show()


#3D SCATTER
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")

ax.scatter3D(XP[:,0],XP[:,1],XP[:,2],c=ytot)
ax.set_xlabel('PC 1', fontweight ='bold')
ax.set_ylabel('PC 2', fontweight ='bold')
ax.set_zlabel('PC 3', fontweight ='bold')
ax.set_xlim((-1500,1500))
ax.set_ylim((-100,100))
ax.set_zlim((-100,100))
plt.show()
'''