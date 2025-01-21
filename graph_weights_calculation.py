from collections import defaultdict
import numpy as np


def process(interaction_data, social_data):
    
    user_to_items = defaultdict(set)
    for user_id, item_id, _ in interaction_data:
        user_to_items[user_id].add(item_id)

    trust_weighted = []
    for user1_id, user2_id in social_data:
        if user1_id in user_to_items and user2_id in user_to_items:
            intersection = user_to_items[user1_id] & user_to_items[user2_id]
            union = user_to_items[user1_id] | user_to_items[user2_id]
            similarity = len(intersection) / len(union) if union else 0
        else:
            similarity = 0 # set sim=0 if user not in training set
        trust_weighted.append([user1_id, user2_id, similarity])

    trust_weighted = np.array(trust_weighted)

    return trust_weighted


data = ['lastfm', 'douban', 'yelp']
for name_data in data:
    social_data = np.loadtxt('dataset/' + name_data + '/trust.txt', dtype=np.int64, delimiter=' ')
    
    # all Hrs
    if name_data == 'douban':
        interaction_data = np.loadtxt('dataset/' + name_data + '/rating.txt', dtype=np.int64, delimiter=' ')
    else:
        interaction_data = np.loadtxt('dataset/' + name_data + '/rating.txt', dtype=np.int64, delimiter='\t')

    # weighted trust for SHaRe (only train set)
    interaction_data = np.loadtxt('dataset/' + name_data + '/train.txt', dtype=np.int64, delimiter=' ')
    trust_weighted = process(interaction_data, social_data)
    np.savetxt('dataset/' + name_data + '/weighted_trust.txt', trust_weighted, delimiter=' ', fmt='%d %d %f')

    
