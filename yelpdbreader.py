# This module is used to interact with the Yelp database

import sqlite3
import re

conn = sqlite3.connect('./data/yelpHotelData.db')
c = conn.cursor()
print('Opened database successfully.')

dirty_cnt=0

def get_review_data(reviewid):
    db_command=r"""
        SELECT  usefulCount, coolCount, funnyCount
                from review
        WHERE   reviewID='"""+reviewid+r"';"
    cursor = c.execute(db_command)

    res=()
    for i in cursor:
        res=i
    # print(res)
    areview={}
    global dirty_cnt

    try:
        areview['Useful_review']=res[0]
        areview['Cool_review']=res[1]
        areview['Funny_review']=res[2]
        areview['review_votes_review']=res[0]+res[1]+res[2]
        areview['dirty_review']=0 
    except:
        dirty_cnt+=1
        print('Dirty review data detected. Total:%d.'%(dirty_cnt))
        areview['dirty_review']=1 

    # print(db_command)
    return areview


def get_user_data(uid):
    # "SELECT id, name, address, salary  from COMPANY"
    db_command=r"""
        SELECT  yelpJoinDate, friendCount, reviewCount,
                firstCount, usefulCount, coolCount, 
                funnyCount, complimentCount, tipCount, 
                fanCount, reviewerID, name from reviewer
        WHERE   reviewerID='"""+uid+r"';"
    cursor = c.execute(db_command)

    res=()
    for i in cursor:
        res=i
    # print(res)
    auser={}
    yelp_time=int(re.search('\d+', res[0]).group())
    auser['yelp_time']=yelp_time
    auser['yelp_time_abs']=2020-yelp_time
    auser['friends']=res[1]
    auser['reviews']=res[2]
    auser['firsts']=res[3]
    auser['Useful']=res[4]
    auser['Cool']=res[5]
    auser['Funny']=res[6]
    auser['compliment']=res[8]
    auser['tips']=res[8]
    auser['followers']=res[9]

    auser['review_votes']=0
    auser['review_votes']+=auser['Useful']
    auser['review_votes']+=auser['Funny']
    auser['review_votes']+=auser['Cool']

    auser['updates']=0 # we can not get this two attributes from the database
    auser['photos']=0 # we can not get this two attributes from the database
    auser['dirty']=0 

    # print(db_command)
    return auser

# for row in cursor:
#    print "ID = ", row[0]
#    print "NAME = ", row[1]
#    print "ADDRESS = ", row[2]
#    print "SALARY = ", row[3], "\n"

def shutdown_database():
    conn.close()
# print "Operation done successfully";

if __name__=='__main__':
    print(get_user_data(r'TYSR6svM9HQoqgBUpQH7GQ'))