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

run=2
runseed=int('2020'+str(run))
np.random.seed(runseed)

pca = PCA(n_components=3)

cls_1=[]
cls_2=[]



stab=0
overlap=4
overlap+=stab
rad=3
bright=False

for i in range(1200):
	img=np.zeros((20,20))
	img2=np.zeros((20,20))
	
	'''
	#CIRCLE SAME AS CLS_2
	strt_x_1=np.random.randint(0+overlap+stab,10-2*rad+1+overlap-stab)
	strt_y_1=np.random.randint(0+overlap+stab,10-2*rad+1+overlap-stab)

	
	rr, cc = draw.disk((strt_x_1+rad,strt_y_1+rad), radius=rad, shape=img.shape)
	img[rr, cc] = np.random.randint(0,255,size=(img[rr, cc].shape[0]))
	#img[rr, cc] = 1
	rr, cc = draw.circle_perimeter(strt_x_1+rad, strt_y_1+rad, rad)
	img[rr, cc] = 255
	'''
	
	strt_x_1=np.random.randint(0+overlap+stab,10-2*rad+1+overlap-stab)
	strt_y_1=np.random.randint(1+overlap+stab,11-2*rad+1+overlap-stab)
	
	start = (strt_x_1,strt_y_1)
	extent = (3,6)
	rr, cc = draw.rectangle(start, extent=extent, shape=img.shape)
	
	
	if bright:
		img[rr, cc] = np.random.randint(250,255,size=(img[rr, cc].shape))
	else:
		img[rr, cc] = np.random.randint(0,255,size=(img[rr, cc].shape))
	
	
	rr, cc = draw.rectangle_perimeter(start, extent=extent, shape=img.shape)
	
	
	img[rr, cc] = 255
	
	
	strt_x_2=np.random.randint(10+2*rad-1-overlap+stab,20-overlap-stab)
	strt_y_2=np.random.randint(10+2*rad-1-overlap+stab,20-overlap-stab)
	
	rr, cc = draw.disk((strt_x_2-rad,strt_y_2-rad), radius=rad, shape=img2.shape)
	

	if bright:
		img2[rr, cc] = np.random.randint(250,255,size=(img2[rr, cc].shape[0]))
	else:
		img2[rr, cc] = np.random.randint(0,255,size=(img2[rr, cc].shape[0]))		


	rr, cc = draw.circle_perimeter(strt_x_2-rad, strt_y_2-rad, rad,shape=img2.shape)
	img2[rr, cc] = 255
	
	
	
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
	
	
	

	
#cls_1=np.array(255*(cls_1/np.max(cls_1)))
#cls_2=np.array(255*(cls_2/np.max(cls_2)))

np.min(cls_1)

cls_1=np.array(cls_1).astype(int)
cls_2=np.array(cls_2).astype(int)

plt.figure(figsize=(10,12),dpi=100)
plt.imshow(cls_1[0].reshape(20,20),cmap=plt.cm.binary_r)
plt.title("Class 1 example",fontsize=15,fontweight="bold")
plt.savefig("./temp/cls_1_example.svg",bbox_inches='tight')
plt.show()
 	
plt.figure(figsize=(10,12),dpi=100)
plt.imshow(cls_2[0].reshape(20,20),cmap=plt.cm.binary_r)
plt.title("Class 2 example",fontsize=15,fontweight="bold")
plt.savefig("./temp/cls_2_example.svg",bbox_inches='tight')
plt.show()


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
plt.figure(figsize = (10, 8), dpi=100)
plt.scatter(XP[:,0],XP[:,1],c=y)
plt.title("PCA", fontweight='bold', fontsize=15)
plt.xlabel("PC 1", fontweight='bold')
plt.ylabel("PC 2", fontweight='bold')
#plt.xlim((-1400,1400))
#plt.ylim((-100,100))
plt.savefig("./temp/PCA2D.svg",bbox_inches='tight')
plt.show()


#3D SCATTER
fig = plt.figure(figsize = (10, 7),dpi=100)
ax = plt.axes(projection ="3d")
ax.set_xlabel('PC 1', fontweight ='bold')
ax.set_ylabel('PC 2', fontweight ='bold')
ax.set_zlabel('PC 3', fontweight ='bold')
#ax.set_xlim((-1500,1500))
#x.set_ylim((-100,100))
#ax.set_zlim((-100,100))
ax.scatter3D(XP[:,0],XP[:,1],XP[:,2],c=y)
plt.savefig("./temp/PCA3D.svg",bbox_inches='tight')
plt.show()




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

'''
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

'''


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





plt.figure(figsize=(10,12),dpi=100)
plt.imshow(mean1_tr,cmap=plt.cm.binary_r)
#plt.plot(range(19),range(19,0,-1),c='red',linewidth=2)
plt.title("Class 1 mean train",fontsize=15,fontweight="bold")
plt.savefig("./temp/cls_1_train_mean.svg",bbox_inches='tight')
plt.show()

plt.figure(figsize=(10,12),dpi=100)
plt.imshow(mean2_tr,cmap=plt.cm.binary_r)
#plt.plot(range(19),range(19,0,-1),c='red',linewidth=2)
plt.title("Class 2 mean train",fontsize=15,fontweight="bold")
plt.savefig("./temp/cls_2_train_mean.svg",bbox_inches='tight')
plt.show()

'''
plt.figure(figsize=(10,12))
plt.imshow(mean1_te,cmap=plt.cm.binary_r)
#plt.plot(range(19),range(19,0,-1),c='red',linewidth=2)
plt.title("Class 1 mean test",fontsize=15,fontweight="bold")
plt.savefig("./temp/cls_1_test_mean.svg",bbox_inches='tight')
plt.show()

	
plt.figure(figsize=(10,12))
plt.imshow(mean2_te,cmap=plt.cm.binary_r)
#plt.plot(range(19),range(19,0,-1),c='red',linewidth=2)
plt.title("Class 2 mean test",fontsize=15,fontweight="bold")
plt.savefig("./temp/cls_2_test_mean.svg",bbox_inches='tight')
plt.show()

plt.figure(figsize=(10,12))
#plt.imshow(mean1_tr+mean2_tr,cmap=plt.cm.binary_r)
plt.plot(range(19),range(19,0,-1),c='red',linewidth=2)
plt.title("Class 1 mean train + Class 2 mean train",fontsize=15,fontweight="bold")
plt.savefig("./temp/total_train_mean",bbox_inches='tight')
plt.show()

plt.figure(figsize=(10,12))
#plt.imshow(mean1_te+mean2_te,cmap=plt.cm.binary_r)
plt.plot(range(19),range(19,0,-1),c='red',linewidth=2)
plt.title("Class 1 mean test + Class 2 mean test",fontsize=15,fontweight="bold")
plt.savefig("./temp/total_test_mean",bbox_inches='tight')
plt.show()
'''

print("DBI for train:",davies_bouldin_score(X,y))
print("DBI for test:",davies_bouldin_score(Xt,yt))


##############MEASURE OVERLAP######################


'''
def get_sub_1(arr):
	return np.fliplr(np.triu(np.fliplr(arr)))
	

def get_sub_2(arr):
	return np.fliplr(np.tril(np.fliplr(arr)))	





c1sub1_te=np.array([get_sub_1(i.reshape(20,20)) for i in cls_1_te])
c2sub1_te=np.array([get_sub_1(i.reshape(20,20)) for i in cls_2_te])
c1sub2_te=np.array([get_sub_2(i.reshape(20,20)) for i in cls_1_te])
c2sub2_te=np.array([get_sub_2(i.reshape(20,20)) for i in cls_2_te])



c1sub1_te_f=np.round(30*c1sub1_te/255)
c2sub1_te_f=np.round(30*c2sub1_te/255)
c1sub2_te_f=np.round(30*c1sub2_te/255)
c2sub2_te_f=np.round(30*c2sub2_te/255)


c1_te_f=np.round(30*cls_1_te/255)
c2_te_f=np.round(30*cls_2_te/255)



c1sub1_te_act=c1sub1_te_f>5
c2sub1_te_act=c2sub1_te_f>5
c1sub2_te_act=c1sub2_te_f>5
c2sub2_te_act=c2sub2_te_f>5

c1_te_act=c1_te_f>5
c2_te_act=c2_te_f>5


c1sub1_te_act_sum=np.array([np.sum(i) for i in c1sub1_te_act])
c1sub2_te_act_sum=np.array([np.sum(i) for i in c1sub2_te_act])

c2sub1_te_act_sum=np.array([np.sum(i) for i in c2sub1_te_act])
c2sub2_te_act_sum=np.array([np.sum(i) for i in c2sub2_te_act])


c1_te_act_sum=np.sum(c1_te_act,axis=1)
c2_te_act_sum=np.sum(c1_te_act,axis=1)



c1_te_overlap=c1sub2_te_act_sum/c1_te_act_sum
c2_te_overlap=c2sub1_te_act_sum/c2_te_act_sum

te_overlap=np.hstack((c1_te_overlap,c2_te_overlap))


print("Total mean overlap:", np.mean(te_overlap))

plt.figure(figsize=(10,12))
plt.hist(te_overlap,bins=(np.arange(0,1.2,0.1)-0.05),color='red',edgecolor='black')
plt.xlabel('Overlap',fontweight='bold',fontsize=12)
plt.ylabel("Counts",fontweight='bold',fontsize=12)
plt.ylim(0,410)
plt.title('Active input neuron overlap per test image histogram',fontweight='bold',fontsize=15)
plt.savefig("./temp/Input_Active_overlap_total_test",bbox_inches='tight')
plt.show()



print("Mean overlap for first class:", np.mean(c1_te_overlap))

plt.figure(figsize=(10,12))
plt.hist(c1_te_overlap,bins=(np.arange(0,1.2,0.1)-0.05),color='red',edgecolor='black')
plt.xlabel('Overlap',fontweight='bold',fontsize=12)
plt.ylabel("Counts",fontweight='bold',fontsize=12)
plt.ylim(0,410)
plt.title('Active input neuron overlap per test image for class 1 histogram',fontweight='bold',fontsize=15)
plt.savefig("./temp/Input_Active_overlap_class_1_test",bbox_inches='tight')
plt.show()




print("Mean overlap for second class:", np.mean(c2_te_overlap))

plt.figure(figsize=(10,12))
plt.hist(c2_te_overlap,bins=(np.arange(0,1.2,0.1)-0.05),color='red',edgecolor='black')
plt.xlabel('Overlap',fontweight='bold',fontsize=12)
plt.ylabel("Counts",fontweight='bold',fontsize=12)
plt.ylim(0,410)
plt.title('Active input neuron overlap per test image for class 2 histogram',fontweight='bold',fontsize=15)
plt.savefig("./temp/Input_Active_overlap_class_2_test",bbox_inches='tight')
plt.show()

'''


'''
img=np.zeros((20,20))

strt_x=np.random.randint(0+overlap+stab,10-2*rad+1+overlap-stab)
strt_y=np.random.randint(0+overlap+stab,10-2*rad+1+overlap-stab)

start = (strt_x,strt_y)
extent = (3,6)
rr, cc = draw.rectangle(start, extent=extent, shape=img.shape)


img[rr, cc] = np.random.randint(0,255,size=(img[rr, cc].shape))


rr, cc = draw.rectangle_perimeter(start, extent=extent, shape=img.shape)


img[rr, cc] = 255

plt.figure(figsize=(10,12))
plt.imshow(img,cmap=plt.cm.binary_r)
plt.title("Class 1 example",fontsize=15,fontweight="bold")
plt.savefig("./temp/cls_1_example",bbox_inches='tight')
plt.show()
'''