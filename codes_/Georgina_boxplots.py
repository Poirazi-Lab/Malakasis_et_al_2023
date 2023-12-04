
layer1 = []
layer2 = []
y = [] 


for name in acc:
    
    # Extract names of models
    name1,name2= name.split('\n')
    name1 = name1.split(' ')[-1] 
    name2 = name2.split(' ')[-1]
    
    # layer 1 = name of activation in layer1
    layer1 += 20*[activation_name[name1]]
    # layer 2 = name of activation in layer2
    layer2 += 20*[activation_name[name2]]
    y += acc[name]

    
df_acc = {'Layer 1' : layer1,
            'Layer 2' : layer2,
            'Accuracy' : y,
            }

df_acc = pd.DataFrame(df_acc)

ax = sns.catplot(x='Layer 1', y='Accuracy', hue='Layer 2', 
            data=df_acc, kind='box', height=4,aspect = 3, legend=True, palette=list(palette.values()),
            legend_out = False)
plt.suptitle('Test set accuracy')
plt.legend(loc=0, prop={'size': 10},title = 'Layer 2',title_fontsize = 8)
