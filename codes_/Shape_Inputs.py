import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score
import csv
from skimage import draw,morphology


'''
img=np.zeros((20,20))

n1=np.random.randint(3,10) #n in 3,16
n2=np.random.randint(10,17) #n in 3,16

m1=np.random.randint(3,10) #m in 3,16
m2=np.random.randint(10,17) #m in 3,16

img[n1-3,m1]=2
img[n1-2,m1-1:m1+2]=3
img[n1-1,m1-2:m1+3]=4
img[n1,m1-3:m1+4]=5
img[n1+1,m1-2:m1+3]=4
img[n1+2,m1-1:m1+2]=3
img[n1+3,m1]=2


plt.figure()
plt.imshow(img,cmap=plt.cm.binary_r)
plt.show()

img2=np.zeros((20,20))

img2=np.zeros((20,20))

img2[n2-2,m2-2:m2+3]=3
img2[n2-1,m2-2:m2+3]=4
img2[n2,m2-2:m2+3]=5
img2[n2+1,m2-2:m2+3]=4
img2[n2+2,m2-2:m2+3]=3



plt.figure()
plt.imshow(img2,cmap=plt.cm.binary_r)
plt.show()

'''

np.random.seed(12)

pca = PCA(n_components=3)

cls_1=[]
cls_2=[]

var=0#opposite of var

#thresh in [3, 17]

thresh_1_down_y=6
thresh_1_up_y=12

thresh_1_down_x=6
thresh_1_up_x=12



thresh_2_down_y=8
thresh_2_up_y=14

thresh_2_down_x=8
thresh_2_up_x=14





for i in range(1200):
	img=np.zeros((20,20))
	img2=np.zeros((20,20))
	
	n1=np.random.randint(thresh_1_down_y+var,thresh_1_up_y-var)
	m1=np.random.randint(thresh_1_down_x+var,thresh_1_up_x-var)
	
	n2=np.random.randint(thresh_2_down_y+var,thresh_2_up_y-var)
	m2=np.random.randint(thresh_2_down_x+var,thresh_2_up_x-var)
	
	img[n1-3,m1]=2
	img[n1-2,m1-1:m1+2]=3
	img[n1-1,m1-2:m1+3]=4
	img[n1,m1-3:m1+4]=5
	img[n1+1,m1-2:m1+3]=4
	img[n1+2,m1-1:m1+2]=3
	img[n1+3,m1]=2
	
	img2[n2-3,m2]=2
	img2[n2-2,m2-1:m2+2]=3
	img2[n2-1,m2-2:m2+3]=4
	img2[n2,m2-3:m2+4]=5
	img2[n2+1,m2-2:m2+3]=4
	img2[n2+2,m2-1:m2+2]=3
	img2[n2+3,m2]=2
	
	
	
	cls_1.append(img.flatten())
	cls_2.append(img2.flatten())
	
	'''
	plt.figure(figsize=(10,12))
	plt.imshow(img,cmap=plt.cm.binary_r)
	plt.title("Class 1")
	plt.show()
 	
	plt.figure(figsize=(10,12))
	plt.imshow(img2,cmap=plt.cm.binary_r)
	plt.title("Class 2")
	plt.show()
	'''
	
cls_1=np.array(255*(cls_1/np.max(cls_1)))
cls_2=np.array(255*(cls_2/np.max(cls_2)))

cls_1=cls_1.astype(int)
cls_2=cls_2.astype(int)


cls_1_tr=cls_1[200:]
cls_1_te=cls_1[:200]

cls_2_tr=cls_2[200:]
cls_2_te=cls_2[:200]


mean1_tr=np.mean(cls_1_tr,axis=0).reshape(20,20)
mean1_te=np.mean(cls_1_te,axis=0).reshape(20,20)
mean2_tr=np.mean(cls_2_tr,axis=0).reshape(20,20)
mean2_te=np.mean(cls_2_te,axis=0).reshape(20,20)

y1=np.ones(1000).astype(int)
y2=2*np.ones(1000).astype(int)

Xa=[]
ya=[]

for i in range(1000):
	Xa.append(cls_1_tr[i])
	Xa.append(cls_2_tr[i])
	ya.append(y1[i])
	ya.append(y2[i])
	
	
X=np.array(Xa)
y=ya



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
#x.set_ylim((-100,100))
#ax.set_zlim((-100,100))
ax.scatter3D(XP[:,0],XP[:,1],XP[:,2],c=y)
plt.show()
#plt.savefig("./temp/PCA3D")




with open('Shapes_train_labels', 'w+') as f:
    # create the csv writer
    writer = csv.writer(f)

    # write a row to the csv file
    writer.writerow(y)

with open('Shapes_train', 'w+') as ff:
    # create the csv writer
	writer = csv.writer(ff)

    # write a row to the csv file
	for i in range(len(X)):
		writer.writerow(X[i])
		

		
Xt=np.vstack((cls_1_te,cls_2_te))

y1t=np.ones(200).astype(int)
y2t=2*np.ones(200).astype(int)
yt=np.hstack((y1t,y2t))

idx_tr = np.random.permutation(len(Xt))

Xt=Xt[idx_tr]
yt=yt[idx_tr]


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



#3D SCATTER
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")

ax.scatter3D(XP[:,0],XP[:,1],XP[:,2],c=yt)
ax.set_xlabel('PC 1', fontweight ='bold')
ax.set_ylabel('PC 2', fontweight ='bold')
ax.set_zlabel('PC 3', fontweight ='bold')
#ax.set_xlim((-1500,1500))
#ax.set_ylim((-100,100))
#ax.set_zlim((-100,100))
plt.show()




with open('Shapes_test_labels', 'w+') as f:
    # create the csv writer
    writer = csv.writer(f)

    # write a row to the csv file
    writer.writerow(yt)

with open('Shapes_test', 'w+') as ff:
    # create the csv writer
	writer = csv.writer(ff)

    # write a row to the csv file
	for i in range(len(Xt)):
		writer.writerow(Xt[i])





plt.figure(figsize=(10,12))
plt.imshow(mean1_tr,cmap=plt.cm.binary_r)
plt.title("Class 1 mean train",fontsize=15,fontweight="bold")
plt.show()

plt.figure(figsize=(10,12))
plt.imshow(mean1_te,cmap=plt.cm.binary_r)
plt.title("Class 1 mean test",fontsize=15,fontweight="bold")
plt.show()
	
plt.figure(figsize=(10,12))
plt.imshow(mean2_tr,cmap=plt.cm.binary_r)
plt.title("Class 2 mean train",fontsize=15,fontweight="bold")
plt.show()

plt.figure(figsize=(10,12))
plt.imshow(mean2_te,cmap=plt.cm.binary_r)
plt.title("Class 2 mean test",fontsize=15,fontweight="bold")
plt.show()

plt.figure(figsize=(10,12))
plt.imshow(mean1_tr+mean2_tr,cmap=plt.cm.binary_r)
plt.title("Class 1 mean train + Class 2 mean test",fontsize=15,fontweight="bold")
plt.show()


print("DBI for train:",davies_bouldin_score(X,y))
print("DBI for test:",davies_bouldin_score(Xt,yt))



arr = np.zeros((20, 20))
rad=3

strt_x=np.random.randint(0,10-2*rad+1)
strt_y=np.random.randint(0,10-2*rad+1)


#start = (strt_x,strt_y)
#extent = (5,5)
#rr, cc = draw.rectangle(start, extent=extent, shape=img.shape)

rr, cc = draw.disk((strt_x+rad,strt_y+rad), radius=rad, shape=arr.shape)
#arr[rr, cc] = 1
arr[rr, cc] = np.random.rand(arr[rr, cc].shape[0])
rr, cc = draw.circle_perimeter(strt_x+rad, strt_y+rad, rad)
arr[rr, cc] = 1


plt.figure(figsize=(10,10))
plt.imshow(arr,cmap=plt.cm.binary_r)
plt.xticks(range(20))

plt.show()



#arr = np.zeros((20, 20))

strt_x=np.random.randint(10+2*rad-1,20)
strt_y=np.random.randint(10+2*rad-1,20)



rr, cc = draw.disk((strt_x-rad,strt_y-rad), radius=rad, shape=arr.shape)
#arr[rr, cc] = 1
arr[rr, cc] = np.random.rand(arr[rr, cc].shape[0])
rr, cc = draw.circle_perimeter(strt_x-rad, strt_y-rad, rad,shape=arr.shape)
arr[rr, cc] = 1


plt.figure(figsize=(10,10))
plt.imshow(arr,cmap=plt.cm.binary_r)
plt.xticks(range(20))
plt.grid(True)
plt.show()



#start = (strt_x,strt_y)
#extent = (5,5)
#rr, cc = draw.rectangle(start, extent=extent, shape=img.shape)
