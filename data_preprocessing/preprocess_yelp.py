import numpy as np

data = 'yelp'

social_data = np.loadtxt('dataset/'+data+'/trustnetwork.txt', dtype=np.int64, delimiter='\t')
interaction_data = np.loadtxt('dataset/'+data+'/rating.txt', dtype=np.int64, delimiter='\t')

dataset = interaction_data[interaction_data[:,2] >= 4] # remove score < 4

# shuffle the interacted items for each user
np.random.shuffle(dataset)

# split train and test sets
train_split_ind = int(len(dataset) * 0.8)
        
train_set = dataset[ : train_split_ind]
valid_test_set = dataset[train_split_ind : ]

# split valid set from train set
valid_split_ind = int(len(valid_test_set) * 0.5)

valid_set = valid_test_set[ : valid_split_ind]
test_set = valid_test_set[valid_split_ind : ]

np.savetxt('dataset/'+data+'/train.txt', train_set, fmt='%d',delimiter=' ')
np.savetxt('dataset/'+data+'/valid.txt', valid_set, fmt='%d',delimiter=' ')
np.savetxt('dataset/'+data+'/test.txt', test_set, fmt='%d',delimiter=' ')
np.savetxt('dataset/'+data+'/trust.txt', social_data, fmt='%d',delimiter=' ')
