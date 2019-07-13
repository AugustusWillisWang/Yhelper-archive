# This module is not used anymore.

# This module will crawl extra features for all terms in the marked dataset. 
# It will be blocked by anti-crawler mechanism in about 20 rounds.

import os
import usercrawler
import json

def load_metadata(file):
    metadata = []  # video id list
    for line in open(file):
        data = line.strip().split(' ')
        metadata.append(data)
    return metadata

def main():
    data_dir = './data/YelpChi/'

    # load data
    print("Loading data...")
    metadata=load_metadata('./data/YelpChi/output_meta_yelpHotelData_NRYRcleaned.txt')

    # Data format for the metadata files:
    # It has 9 columns of data, following are the specification of the important columns used in this work.
    # 1. Date
    # 2. review ID
    # 3. reviewer ID (2)
    # 4. product ID
    # 5. Label (N means genuine review and Y means fake reviews)
    # 9. star rating

    result=[]
    cnt=0
    for i in metadata:
        result.append(usercrawler.get_user_data(i[2]))
        print(usercrawler.get_user_data(i[2]))
        cnt=cnt+1
        if(cnt%100==0):
            print('current user: %d'%(cnt))

    json_out=json.dumps(result)
    with open('./data/preprocessed/socialdata.json', 'w') as f:
        f.write(json_out)
    
if __name__ == "__main__":
    main()