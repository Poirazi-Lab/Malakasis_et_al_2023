from mnist import MNIST
import matplotlib.pyplot as plt
import numpy as np
import csv


mndata=MNIST('./MNIST/')
imagelists,labels=mndata.load_training()


#Creating arrays of MNIST images
#imagearrays=[]
#for img in imagelists:
#	imagearrays.append(np.reshape(np.array(img),(28,28)))



grouplist=[]

for i in range(10):
	temp=[]
	groups=[]
	for j in range(60000):
		if labels[j]==i:
			temp.append(imagelists[j])			
		if len(temp)==60:
			groups.append(np.around(np.mean(np.array(temp),axis=0)))
			temp=[]
	groups.append(np.around(np.mean(np.array(temp),axis=0)))
	grouplist.append(groups)



labs=[]
for i in range(10):
	for j in range(len(grouplist[i])):
		labs.append(i)
		

'''
with open('MNIST_centroid_labels', 'w+') as f:
    # create the csv writer
    writer = csv.writer(f)

    # write a row to the csv file
    writer.writerow(labs)

with open('MNIST_centroid_images', 'w+') as ff:
    # create the csv writer
	writer = csv.writer(ff)

    # write a row to the csv file
	for i in range(10):
		for j in grouplist[i]:
			   writer.writerow(j)
'''

#for img in imagelists:
#	imagearrays.append(np.reshape(np.array(img),(28,28)))


#TESTING MY FILES
my_data = np.genfromtxt('MNIST_centroid_images', delimiter=',')

my_data[0]
asd=np.reshape(my_data[0],(28,28))

for i in range(210,220):
	asd=np.reshape(my_data[i],(28,28))
	plt.figure()
	plt.imshow(asd,cmap=plt.cm.binary_r)

