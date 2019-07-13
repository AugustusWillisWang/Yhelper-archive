# encoding=utf8

# This module is the classifier. Import this module to train or use pretrained classifier.
# 
# Usage: 
# import classifier
# classifier.load_model()
# classifier.check(reviews)

# settings-----------------------------
PARA_TUNING=0
# PARA_TUNING=1
# settings-----------------------------

# train dataset-----------------------------
DATA_SET=0 # hotel
# DATA_SET=1 # restaurant
# train dataset-----------------------------

# word2vec ref: https://zhuanlan.zhihu.com/p/24961011
# TODO: WU: word unigram, 
# TODO: WB: word bigrams, 
# TODO: POS denotes part-of-speech tags
# TODO: LIWC 

# * unigram word2vec
# * bigrams word2vec

# "Using only behavioral features (BF) boosts precision by
# about 20% and recall by around 7% in both domains
# resulting in around 14% improvement in F1. Thus,
# behaviors are stronger than linguistic n-grams for
# detecting real-life fake reviews filtered by Yelp."

import os
import math
import pickle
import numpy as np
import vader_awmod
import xgboost as xgb
import json
# from sklearn.metrics import mean_squared_error
# from sklearn.svm import SVR
# from sklearn.neural_network import MLPRegressor
# from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, precision_score, recall_score

import preprocess_review

# def loglize(number):
#     return math.log(number+1)

# def zero_test(number):
#     # @retval 1 if number==0, 0 otherwise
#     if(number < 0.001):
#         if(number > -0.001):
#             return 1
#     return 0


# def zero_vector(width):
#     return [0.0 for i in range(width)]

# model define
# pca = PCA(n_components=20)

basic_senti_classifier = vader_awmod.SentimentIntensityAnalyzer()

tfidf_vectorizer = TfidfVectorizer()

tfidf_classifier = Pipeline([
    ('sgd', SGDClassifier())
])

late_fusion_classifier = Pipeline([
    ('xgb', xgb.XGBClassifier())
])

def load_model():
    global tfidf_classifier
    global late_fusion_classifier
    global tfidf_vectorizer
    model_output1 = open('./model/model1.data', 'rb')
    model_output2 = open('./model/model2.data', 'rb')
    model_output3 = open('./model/tfidf-model.data', 'rb')
    tfidf_classifier=pickle.load(model_output1)
    late_fusion_classifier=pickle.load(model_output2)
    tfidf_vectorizer=pickle.load(model_output3)

def check(reviews):
    '''
    reviews is a dict with the following format:
    reviews: {
        'user_info':{
            'dirty':
            'Funny':
            'Useful':
            'Cool': 
            'firsts':
            'yelp_time_abs': (2020-yelp_time)
            'review_votes':
            'compliment':
            'tips':
            'followers':
        }
        'review': (review_text)
        'grade':
        'friends':
        'photos':
    }
    '''
    review_text=[]
    basic_senti_scores = []

    # get review text
    for i in reviews:
        if(i['user_info']['dirty']):
            review_text.append('')
        else:
            review_text.append(i['review'])

    # preprocess review
    preprocessed_review = []

    for i in review_text:
        postprocess_review=[]
        words = preprocess_review.pre_process(i, preprocess_review.porter)
        if (len(i)>0):
            for word in words:
                if word not in preprocess_review.stops:
                    postprocess_review.append(word)
            preprocessed_review.append(' '.join(words)) 
        else:
            preprocessed_review.append(' ') 

    # get senti score for review
    for instance in preprocessed_review:
        basic_senti_scores.append(basic_senti_classifier.polarity_scores(instance)["compound"])

    print(basic_senti_scores)

    # extract tf-idf features
    tfidf_features = tfidf_vectorizer.transform(preprocessed_review)

    print(tfidf_features)

    # concat features
    basic_features=[]

    cnt=0
    for i in reviews:
        if(i['user_info']['dirty']):
            # basic_features.append([0,0])
            # basic_features.append([0,0,0,0,0,0,0,0,0,0,0,0,0])
            basic_features.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        else:
            i_list=[
                i['grade'], basic_senti_scores[cnt],
                i['user_info']['Funny'], i['user_info']['Useful'], 
                i['user_info']['Cool'], i['user_info']['firsts'],
                i['user_info']['yelp_time_abs'], i['user_info']['review_votes'], i['user_info']['compliment'],
                i['user_info']['tips'], i['friends'], i['user_info']['followers'], i['photos']

                # review features
                ,i['Useful_review'], i['Cool_review'], i['Funny_review'], i['review_votes_review'] 
            ]
            basic_features.append(i_list)
            # review_text.append(i['review'])
        cnt+=1

    # basic_features=get_features(metadata, basic_senti_scores, extra_features)

    # classify
    sgd_predict=tfidf_classifier.predict(tfidf_features).reshape(-1,1)
    basic_features=np.array(basic_features)

    print(basic_features)
    print(sgd_predict)
    print(concat_feature_func([basic_features, sgd_predict]))
    predicts = late_fusion_classifier.predict(concat_feature_func([basic_features, sgd_predict]))
    print(predicts.reshape(1,-1).tolist()[0])
    return predicts.reshape(1,-1).tolist()[0]


def concat_feature_func(feature_list):
    return np.concatenate(feature_list, axis=1)

def load_metadata(file):
    metadata = []  # video id list
    for line in open(file):
        data = line.strip().split(' ')
        metadata.append(data)
    return metadata

def load_review(file):
    review = []  # video id list
    for line in open(file):
        data = line.strip()
        review.append(data)
    return review

def setup_user_to_review_dict(metadata):
    adict={}
    for i in metadata:
        if i[2] in adict:# i[2] is reviewer ID
            adict[i[2]].append(i[1])# add a new review map
        else:
            adict[i[2]]=[i[1]]# add a new review map
    return dict
    
def setup_review_to_user_dict(metadata):
    adict={}
    for i in metadata:
        adict[i[1]]=[i[2]]# add a new review-reviewer map
    return dict

def get_groundtruth_nparray(metadata):
    res=[]
    for i in metadata:
        if(i[4]=='Y'):
            res.append(1)
        else:
            res.append(0)
    res=np.array(res)
    res=res.reshape(-1,1)
    return res

def get_features(metadata, basic_senti_scores, extra_features):
    res=[]
    for i, k, j in zip(metadata, basic_senti_scores, extra_features):
        data=[i[8], k] # only star ranking
        j_list=[]
        if(j['dirty']):
            # j_list=[0,0,0,0,0,0,0,0,0,0,0]        
            j_list=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]        
            # j_list=[0,0,0,0]        
            # j_list=[]        
        else:
            j_list=[
                j['Funny'], j['Useful'], j['Cool'], j['firsts'],
                j['yelp_time_abs'], j['review_votes'], j['compliment'],
                j['tips'], j['friends'], j['followers'], j['photos']

                   # review features
                ,j['Useful_review'], j['Cool_review'], j['Funny_review'], j['review_votes_review'] 
            ]
        data.extend(j_list)
        res.append(data)
    res=np.array(res)
    return res

def load_extra_features(file):
    with open(file, 'r') as f:
        return json.loads(f.read())

def main():
    data_dir = './data/YelpChi/'

    # load data
    print("Loading data...")
    metadata=[]
    review=[]
    preprocessed_review=[]
    extra_features=[]
    if(DATA_SET==0):
        metadata=load_metadata('./data/YelpChi/output_meta_yelpHotelData_NRYRcleaned.txt')
        review=load_review('./data/YelpChi/output_review_yelpHotelData_NRYRcleaned.txt')
        preprocessed_review=load_review('./data/preprocessed/reviews_processed.txt')
        extra_features=load_extra_features('./data/preprocessed/extra_feature.json')
    else:
        metadata=load_metadata('./data/YelpChi/output_meta_yelpResData_NRYRcleaned.txt')
        review=load_review('./data/YelpChi/output_review_yelpResData_NRYRcleaned.txt')
        preprocessed_review=load_review('./data/YelpChi/output_review_yelpResData_NRYRcleaned.txt')
        extra_features=load_extra_features('./data/preprocessed/extra_feature_res.json')

    # user_to_review=setup_user_to_review_dict(metadata)
    # review_to_user=setup_review_to_user_dict(metadata)
    groundtruth=get_groundtruth_nparray(metadata)



    i_cnt=0
    y_cnt=0
    for i in groundtruth:
        i_cnt+=1
        if(i==1):
            y_cnt+=1
    print('There are %d instances, %d (%f percent) of them are fake reviews.'%(i_cnt, y_cnt, float(y_cnt)/i_cnt))


    print("Extracting features...")
    basic_senti_scores = []
    for instance in review:
        basic_senti_scores.append(basic_senti_classifier.polarity_scores(instance)["compound"])

    # Data format for the metadata files:
    # It has 9 columns of data, following are the specification of the important columns used in this work.
    # 1. Date
    # 2. review ID
    # 3. reviewer ID
    # 4. product ID
    # 5. Label (N means genuine review and Y means fake reviews)
    # 9. star rating

    print("Extracting tf-idf features...")
    tfidf_features = tfidf_vectorizer.fit_transform(preprocessed_review)

    # concat features
    print ("Concating features...")
    basic_features=get_features(metadata, basic_senti_scores, extra_features)

    # features=concat_feature_func([basic_feature])
    # TODO: Add Tfidf based classifier
    # TODO: https://blog.csdn.net/TiffanyRabbit/article/details/72650606 降维
    # TODO: Introduce a SVM classifier
    # TODO: Introduce the classifier in the paper

    # TODO: LDA

    print(basic_features)
    # print(review)
    print(groundtruth)

    # TODO: try lightgbm
    # https://lightgbm.readthedocs.io/en/latest/Python-API.html#scikit-learn-api

    if(PARA_TUNING):
        print('Now working in parameter tuning mode.')
        parameters = {
            # 'mlp__max_iter':[150],
            # 'mlp__max_iter':range(110, 160, 10),
        }
    
        # gs_clf = GridSearchCV(image_feature_regressor, parameters, cv=5, iid=False, n_jobs=-1, scoring='neg_mean_squared_error')
        gs_clf = GridSearchCV(late_fusion_classifier, parameters, cv=5, iid=False, n_jobs=-1, scoring='neg_mean_squared_error')
        gs_clf = gs_clf.fit(basic_features, groundtruth)
        print(gs_clf.best_score_)
        for param_name in sorted(parameters.keys()):
            print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
    
        exit(0)


    print("Start training and predict...")
    kf = KFold(n_splits=10, shuffle=True)
    avg_p = 0
    avg_r = 0
    for train, test in kf.split(basic_features):

        # BASIC MODEL--------------------------------
        # train
        tfidf_classifier.fit(tfidf_features[train], groundtruth[train])
        sgd_predict=tfidf_classifier.predict(tfidf_features[train]).reshape(-1,1)
        feature=concat_feature_func([basic_features[train], sgd_predict])
        late_fusion_classifier.fit(feature, groundtruth[train])

        # predict
        sgd_predict=tfidf_classifier.predict(tfidf_features[test]).reshape(-1,1)
        predicts = late_fusion_classifier.predict(concat_feature_func([basic_features[test], sgd_predict]))

        # for dbg
        # print(predicts)
        print(sgd_predict.reshape(1,-1))
        # print(sgd_predict)
        # predicts=sgd_predict

        print(classification_report(groundtruth[test],predicts))
        avg_p += precision_score(groundtruth[test],predicts, average='macro')
        avg_r += recall_score(groundtruth[test],predicts, average='macro')

    print('Average Precision is %f.' %(avg_p/10.0))
    print('Average Recall is %f.' %(avg_r/10.0))
    
    model_output1 = open('./model/model1.data', 'wb')
    model_output2 = open('./model/model2.data', 'wb')
    model_output3 = open('./model/tfidf-model.data', 'wb')

    pickle.dump(tfidf_classifier, model_output1)
    pickle.dump(late_fusion_classifier, model_output2)
    pickle.dump(tfidf_vectorizer, model_output3)
    model_output1.close()
    model_output2.close()
    model_output3.close()

if __name__ == "__main__":
    main()