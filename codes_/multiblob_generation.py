import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score
import csv



pca = PCA(n_components=3)


#Create Train set
np.random.seed(12)

varval1=100
#meanval1=1

varval2=1000
#meanval2=1

dim=400

train_samples1=900
train_samples2=100
train_samples=train_samples1+train_samples2

test_samples1=20
test_samples2=180
test_samples=test_samples1+test_samples2

tot_samples1=train_samples1+test_samples1
tot_samples2=train_samples2+test_samples2
tot_samples=tot_samples1+tot_samples2



step=1
#step2=1


meanvec=np.arange(10,int(dim)+10,1)/step
#meanvecnew=np.arange(10,int(dim)+10,1)/step2


idx = np.random.permutation(len(meanvec))
idx2 = np.random.permutation(len(meanvec))


meanvec1=meanvec[idx]
#meanvec12=meanvecnew[idx]


meanvec2=meanvec[idx2]
#meanvec22=meanvecnew[idx2]

meanvecmid=(meanvec1+meanvec2)/2

meanvec12=(meanvec1 + meanvecmid)/2
meanvec22=(meanvecmid + meanvec2)/2


meanvec12=(meanvec1 + meanvec12)/2
meanvec22=(meanvec22 + meanvec2)/2


varvec1=varval1*np.ones(dim)
varvec12=varval2*np.ones(dim)

varvec2=varval1*np.ones(dim)
varvec22=varval2*np.ones(dim)


zeros1=np.random.permutation(np.arange(0,dim,1))[:0]


#X1=np.random.multivariate_normal(meanval1*np.ones(dim), varval1*np.eye(dim),train_samples)
#y1=np.ones(train_samples).astype(int)

X11=np.random.multivariate_normal(meanvec1, varvec1*np.eye(dim),tot_samples1)
X12=np.random.multivariate_normal(meanvec12, varvec12*np.eye(dim),tot_samples2)

y1=np.ones(train_samples).astype(int)
y1t=np.ones(test_samples).astype(int)

for i in X11:
	i[zeros1]=0
	
for i in X12:
	i[zeros1]=0
	
#X2=np.random.multivariate_normal(meanval2*np.ones(dim), varval2*np.eye(dim),train_samples)
#y2=2*np.ones(train_samples).astype(int)

X21=np.random.multivariate_normal(meanvec2, varvec2*np.eye(dim),tot_samples1)
X22=np.random.multivariate_normal(meanvec22, varvec22*np.eye(dim),tot_samples2)

y2=2*np.ones(train_samples).astype(int)
y2t=2*np.ones(test_samples).astype(int)


for i in X21:
	i[zeros1]=0

for i in X22:
	i[zeros1]=0

X1=np.vstack((X11,X12))
X2=np.vstack((X21,X22))

X1=X1-np.min(X1)
X1=255*X1/np.max(X1)
X1=X1.astype(int)


X2=X2-np.min(X2)
X2=255*X2/np.max(X2)
X2=X2.astype(int)

shuf1 = np.random.permutation(len(X1))
shuf2 = np.random.permutation(len(X2))

X1=X1[shuf1]
X2=X2[shuf2]

X1t=X1[:test_samples]
X1=X1[test_samples:]

X2t=X2[:test_samples]
X2=X2[test_samples:]




y=np.hstack((y1,y2))
X=np.vstack((X1,X2))



Xa=[]
ya=[]
for i in range(int(train_samples)):
	Xa.append(X1[i])
	Xa.append(X2[i])
	ya.append(y1[i])
	ya.append(y2[i])
	
	
X=np.array(Xa)
y=ya


yt=np.hstack((y1t,y2t))
Xt=np.vstack((X1t,X2t))


idx_tr = np.random.permutation(len(Xt))

Xt=Xt[idx_tr]
yt=yt[idx_tr]

print("DBI for train:",davies_bouldin_score(X,y))
print("DBI for test:",davies_bouldin_score(Xt,yt))




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





plt.scatter(X[:,0],X[:,1],c=y)
plt.xlabel("Dimension 1", fontweight='bold')
plt.ylabel("Dimension 2", fontweight='bold')
plt.show()

plt.scatter(X[:,0],X[:,2],c=y)
plt.xlabel("Dimension 1", fontweight='bold')
plt.ylabel("Dimension 3", fontweight='bold')
plt.show()

plt.scatter(X[:,1],X[:,2],c=y)
plt.xlabel("Dimension 2", fontweight='bold')
plt.ylabel("Dimension 3", fontweight='bold')
plt.show()

plt.scatter(X[:,1],X[:,3],c=y)
plt.xlabel("Dimension 1", fontweight='bold')
plt.ylabel("Dimension 4", fontweight='bold')
plt.show()


XP=pca.fit_transform(X)

pca.explained_variance_ratio_
plt.figure()
plt.scatter(XP[:,0],XP[:,1],c=y)
plt.title("PCA", fontweight='bold', fontsize=15)
plt.xlabel("PC 1", fontweight='bold')
plt.ylabel("PC 2", fontweight='bold')
#plt.xlim((-1400,1400))
#plt.ylim((-100,100))
plt.show()
#plt.savefig("./temp/PCA2D")


#3D SCATTER
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
ax.set_xlabel('PC 1', fontweight ='bold')
ax.set_ylabel('PC 2', fontweight ='bold')
ax.set_zlabel('PC 3', fontweight ='bold')
#ax.set_xlim((-1500,1500))
#ax.set_ylim((-100,100))
#ax.set_zlim((-100,100))
ax.scatter3D(XP[:,0],XP[:,1],XP[:,2],c=y)
plt.show()
#plt.savefig("./temp/PCA3D")







#test


plt.scatter(Xt[:,0],Xt[:,1],c=yt)
plt.xlabel("Dimension 1", fontweight='bold')
plt.ylabel("Dimension 2", fontweight='bold')
plt.show()

plt.scatter(Xt[:,0],Xt[:,2],c=yt)
plt.xlabel("Dimension 1", fontweight='bold')
plt.ylabel("Dimension 3", fontweight='bold')
plt.show()

plt.scatter(Xt[:,1],Xt[:,2],c=yt)
plt.xlabel("Dimension 2", fontweight='bold')
plt.ylabel("Dimension 3", fontweight='bold')
plt.show()

plt.scatter(Xt[:,1],Xt[:,3],c=yt)
plt.xlabel("Dimension 1", fontweight='bold')
plt.ylabel("Dimension 4", fontweight='bold')
plt.show()


XP=pca.fit_transform(Xt)

pca.explained_variance_ratio_
plt.figure()
plt.scatter(XP[:,0],XP[:,1],c=yt)
plt.title("PCA", fontweight='bold', fontsize=15)
plt.xlabel("PC 1", fontweight='bold')
plt.ylabel("PC 2", fontweight='bold')
#plt.xlim((-1400,1400))
#plt.ylim((-100,100))
plt.show()
#plt.savefig("./temp/PCA2D")


#3D SCATTER
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
ax.set_xlabel('PC 1', fontweight ='bold')
ax.set_ylabel('PC 2', fontweight ='bold')
ax.set_zlabel('PC 3', fontweight ='bold')
#ax.set_xlim((-1500,1500))
#ax.set_ylim((-100,100))
#ax.set_zlim((-100,100))
ax.scatter3D(XP[:,0],XP[:,1],XP[:,2],c=yt)
plt.show()
#plt.savefig("./temp/PCA3D")


asd=np.reshape(X[0],(int(np.sqrt(dim)),int(np.sqrt(dim))))
plt.figure()
plt.imshow(asd,cmap=plt.cm.binary_r)
plt.show()

asd=np.reshape(X[1],(int(np.sqrt(dim)),int(np.sqrt(dim))))
plt.figure()
plt.imshow(asd,cmap=plt.cm.binary_r)
plt.show()