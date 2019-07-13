# coding=utf-8

# This module is the server side of the Yhelper Chrome Extension

from flask import Flask
from flask import request

import pageparser

print(r'[Yhelper Server] Package imported')

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    # return '<h1>Home</h1>'
    # return result_db.detect_fake_review('url')

    return pageparser.detect_fake_review(request.form['url'])

@app.route('/send', methods=['GET', 'POST'])
def send():
    print("Msg received.")
    return '<h1>Received msg.</h1>'+request.form['url']

@app.route('/result', methods=['GET', 'POST'])
def result():
    # return result_db.detect_fake_review('url')
    return pageparser.detect_fake_review('https://www.yelp.com/biz/hanks-cajun-grill-and-oyster-bar-houston?start=120')

# @app.route('/signin', methods=['GET'])

if __name__ == '__main__':
    app.run()