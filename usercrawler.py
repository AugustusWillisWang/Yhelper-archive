#coding=utf-8

# This module will crawl user information from yelp according to UID.

import io
import re
import json
import requests
from bs4 import BeautifulSoup

# ref https://cuiqingcai.com/1319.html

# ref https://www.yelp.com/dataset/download

def get_user_data(uid):

    auser={}

    url=r'https://www.yelp.com/user_details?userid='+uid
    html=requests.get(url).text
    soup = BeautifulSoup(html, features="html.parser")

    # def detect_attribute(attr_name):
    #     try:    
    #         review_vote=soup.find('li', string=attr_name)
    #         print review_vote.strings
    #         info_level+=1
    #         return int(review_vote.strong.string)
    #     except:
    #         return 0

    info_level=0 # this index shows how much information this account provides

    try:

        def match_attribute(attr_name):
            try:    
                attr_re=re.compile(attr_name + r'[\s, \n]*<strong>(\d+)')
                # attr_re=re.compile(r'<strong>(\d+)')
                res=re.search(attr_re, html).group(1)
                # info_level+=1
                # print(res)
                return int(res)
            except:
                return 0
    
        try:
            histogram_count=soup('td', class_="histogram_count")
            rating_distribution=[] # 5, 4, 3, 2, 1
            for i in histogram_count:
                rating_distribution.append(int(i.string))
    
            if(len(rating_distribution)<5):
                raise AttributeError
            info_level+=1
            auser['rating_distribution']=rating_distribution
        except AttributeError:
            auser['rating_distribution']=[0,0,0,0,0]
            
        auser['Useful']=match_attribute('Useful')
        auser['Funny']=match_attribute('Funny')
        auser['Cool']=match_attribute('Cool')

        auser['review_votes']=0

        auser['review_votes']+=auser['Useful']
        auser['review_votes']+=auser['Funny']
        auser['review_votes']+=auser['Cool']
    
        auser['tips']=match_attribute('Tips')
        auser['updates']=match_attribute('Review Updates')
        auser['firsts']=match_attribute('Firsts')
        auser['followers']=match_attribute('Followers')

        auser['compliment']=0 # this info is not available for ordinary users

        if (auser['review_votes']!=0):
            info_level+=1
        if (auser['tips']!=0):
            info_level+=1
        if (auser['updates']!=0):
            info_level+=1
        if (auser['firsts']!=0):
            info_level+=1
        if (auser['followers']!=0):
            info_level+=1
    
        # TODO Compliments

        # yelp time
        yelp_since=soup.find('h4', string='Yelping Since')
        yelp_time=yelp_since.parent.p.string
        yelp_time=int(re.search('\d+', yelp_time).group())
        yelp_time_abs=2020-yelp_time
    
        auser['yelp_time']=yelp_time
        auser['yelp_time_abs']=yelp_time_abs
    
        # print(yelp_time)
    
        for a in soup(class_='ysection'):
            for b in a(class_='ylist'):
                for c in b('li'):
                    info_level+=1
    
        # print(info_level)
        auser['info_level']=info_level
        auser['dirty']=0 # this user data is not dirty
    
        # TODO: 'View more graphs'
    
        # print soup.a.prettify()
    
        # print auser
    
    except:
        auser['dirty']=1 # there were some problems, data needs cleaning
    
    return auser

def main():
    # get_user_data('cKTA-iJbfrioKWFiDreghw')
    # get_user_data('hN03Wim4nDQ-824277aEsQ')
    # get_user_data('--vbfrIPT3d2Fdmq7M74RA')
    print(get_user_data('IFTr6_6NI4CgCVavIL9k5g'))
    pass


    # print all_raw_review

if __name__ == "__main__":
    main()