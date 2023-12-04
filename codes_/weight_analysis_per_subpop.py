import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

#df0=pd.read_table("./results/weights_per_epoch_asdf.txt",header=None, error_bad_lines=False)

# Input
data_file = "./results/weights_per_epoch_sub1asdf.txt"

# Delimiter
data_file_delimiter = '\t'

# The max column count a line in the file could have
largest_column_count = 0

# Loop the data lines
with open(data_file, 'r') as temp_f:
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
df = pd.read_csv(data_file, header=None, delimiter=data_file_delimiter, names=column_names)


'''
for i in range(0,len(df.values),10):	
	x=df.values[i]
	plt.figure()
	plt.grid(True)
	plt.hist(x,100,range=(0,1),ec="k")
	plt.ylim(0,400)
	plt.xlabel('Weight Value')
	plt.title('Weight Distribution ' + str(i) + ' iterations')
	#plt.savefig("./temp/Weights" + str(i),bbox_inches='tight')
	plt.show()
'''

# Input
data_file2 = "./results/weights_per_epoch_sub2asdf.txt"


# The max column count a line in the file could have
largest_column_count = 0

# Loop the data lines
with open(data_file2, 'r') as temp_f:
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
df2 = pd.read_csv(data_file2, header=None, delimiter=data_file_delimiter, names=column_names)
# print(df)




sub1 = "mediumslateblue"
sub2 = "firebrick"



for i in range(0,len(df2),10):
	#x0=df0.values[i]	
	x1=df.values[i]	
	x2=df2.values[i]
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
	
	
