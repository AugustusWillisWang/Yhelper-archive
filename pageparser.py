# This module is used to crawl reviews from Yelp, then return the classify result.
# It will also be used as the backend of Chrome Ext.
# In real world, the backend of the Ext will just read data from a database.
# You can seen this module as a live example.

import io
import os
import re
import json
import requests
from bs4 import BeautifulSoup

import usercrawler
import classifier

# ref https://cuiqingcai.com/1319.html
# ref https://www.kancloud.cn/wizardforcel/bs4-doc/141641
# ref https://www.crummy.com/software/BeautifulSoup/bs4/doc.zh/#id44

html = io.open('./data/YelpWeb/test.html', 'r',  encoding='utf-8').read()
float_re=re.compile(r'[0-9]+\.?[0-9]*')
uid_re=re.compile(r'.*userid=(.+)')

def to_int(input_str):
    try:
        res = int(input_str)
    except:
        res = 0
    finally:
        return res

def parse_yelp_page(html):
    print('Parsing yelp pages...')
    soup = BeautifulSoup(html, features="html.parser")
    # print soup.prettify()
    reviews=[]
    all_raw_review = soup.find_all('div', class_="review review--with-sidebar")
    for i in all_raw_review:
        areview={}

        # name and uid
        name=i.find(class_="user-display-name js-analytics-click")
        try:
            areview['name']=name.string
            areview['user_url']=name['href']
            # print areview['user_url']
            areview['uid']=re.match(uid_re, areview['user_url']).group(1)
            areview['is_qype']=0
        except AttributeError:
            areview['name']='Qype_User'
            areview['is_qype']=1

        # basic user info
        review_sidebar=i.find('div', class_="review-sidebar")
        review_wrapper=i.find('div', class_="review-wrapper")
        # review_sidebar.descendants
        review_sidebar_content=review_sidebar.find('div', class_="review-sidebar-content")
        ypassport_media_block=review_sidebar_content.find('div', class_="ypassport media-block")
        media_story=ypassport_media_block.find('div', class_="media-story")
        user_passport_stats=media_story.find('ul', class_="user-passport-stats")
        areview['friends']=int(user_passport_stats.find('li', class_="friend-count responsive-small-display-inline-block").b.string)
        areview['reviews']=int(user_passport_stats.find('li', class_="review-count responsive-small-display-inline-block").b.string)
        try:
            areview['photos']=int(user_passport_stats.find('li', class_="photo-count responsive-small-display-inline-block").b.string)
        except AttributeError:
            areview['photos']=int(0)
        try:
            areview['elite']=user_passport_stats.find('li', class_="is-elite responsive-small-display-inline-block").a.string
            areview['is_elite']=1
        except AttributeError:
            areview['is_elite']=0


        # review and grade
        stripped_strings = review_wrapper.div.p.stripped_strings
        review=''
        for i in stripped_strings:
            review+=i
        areview['review']=str(review)
            
        areview['grade']=review_wrapper.div.div.div.div['title']
        areview['grade']=int(float(re.match(float_re, areview['grade']).group()))

        areview['Useful_review']=to_int(review_wrapper.find('span', string='Useful').parent.find('span', class_='count').string)
        areview['Cool_review']=to_int(review_wrapper.find('span', string='Cool').parent.find('span', class_='count').string)
        areview['Funny_review']=to_int(review_wrapper.find('span', string='Funny').parent.find('span', class_='count').string)
        areview['review_votes_review'] = areview['Useful_review']+areview['Cool_review']+areview['Funny_review']
        if(areview['is_qype']!=1):
            areview['user_info']=usercrawler.get_user_data(areview['uid'])
        else:
            areview['user_info']={'dirty':1}

        # print areview['friends']
        # print areview['reviews']
        # print areview['photos']
        # print areview['review']
        # print areview['grade']
        reviews.append(areview)
        # print review_wrapper
        # exit(0)

        print(areview)

    json_out=json.dumps(reviews)
    with open('./data/preprocessed/yelppage.json', 'w') as f:
        f.write(json_out)
    
    return reviews

def update_html(html, classify_result):
    soup = BeautifulSoup(html, features="html.parser")
    all_raw_review = soup.find_all('div', class_="review review--with-sidebar")
    cnt=0
    for i in all_raw_review:
        # notes: soup.new_tag must be called for each new node. 
        # Other wise the tag will be detached from its former position and link to a new position.
        if(classify_result[cnt]==1):
            fake_tag = soup.new_tag('span', class_="fake-review-tip")
            fake_tag.string='This review has a high chance to be a fake review.'
            fake_tag['class']='rating-qualifier'
            i.find('div', class_="biz-rating biz-rating-large clearfix").span.insert_after(fake_tag)
        cnt+=1
        # print (i.find('div', class_="biz-rating biz-rating-large clearfix").prettify())
    new_html=soup.prettify()
    with io.open('./data/preprocessed/new_yelppage.html', 'w', encoding='utf-8') as f:
        f.write(new_html)

    return new_html

    # add
    # <span class="rating-qualifier">
        # This review has a high chance to be a fake review.
    # </span>
    # if the classifier thinks it is a fake review
    # pass



# def soup_filter(tag):
    # return tag.class_="review review--with-sidebar"

def detect_fake_review(url):
    html=requests.get(url).text
    reviews=parse_yelp_page(html)
    classifier.load_model()
    classify_result=classifier.check(reviews)
    return update_html(html, classify_result)


def main():
    # reviews=parse_yelp_page(html)
    # # call classifier
    # classifier.load_model()
    # classify_result=classifier.check(reviews)
    # # classify_result=[1,0,0,0,1,0,1,1]
    # update_html(html, classify_result)

    detect_fake_review('https://www.yelp.com/biz/hanks-cajun-grill-and-oyster-bar-houston?start=120')

    # print all_raw_review

if __name__ == "__main__":
    main()