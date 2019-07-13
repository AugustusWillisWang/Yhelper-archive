# This module will extract extra feature from Yelp database for marked review dataset.

import yelpdbreader
import json

# print(yelpdbreader.get_user_data('TYSR6svM9HQoqgBUpQH7GQ'))

def load_metadata(file):
    metadata = []  # video id list
    for line in open(file):
        data = line.strip().split(' ')
        metadata.append(data)
    return metadata

metadata=load_metadata('./data/YelpChi/output_meta_yelpResData_NRYRcleaned.txt')
res=[]
cnt=0
error=0

for i in metadata:
    try:
        features=yelpdbreader.get_user_data(i[2])# i[2] is reviewer ID
        review_features=yelpdbreader.get_review_data(i[1])# i[1] is review ID
        features.update(review_features)
        res.append(features)
        cnt+=1
        if(cnt%100==0):
            print('Current user: %d'%(cnt))
    except:
        res.append({'dirty':1})
        cnt+=1
        print('Error occurred. Current user: %d'%(cnt))
        error+=1

with open('./data/preprocessed/extra_feature_res.json', 'w') as f:
    f.write(json.dumps(res))

print('Preprocess(db) finished. %d error(s) occurred.'%(error))


