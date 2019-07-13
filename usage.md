[TOC]

# 1. Usage

## 1.1. Model and data set

To re-train the classifier, unzip the data-zip to `./data` folder (if not exist) in the root dir of this project.

## 1.2. Train and evaluate model 

The local version requires `python2.7` with `sklearn`, `xgb`, and `nltk`. To run the chrome extension and the website, a web framework `flask` is also needed. You can install them via pip.

```
pip install sklearn
pip install xgboost
pip install nltk
pip install flask
```

To do preprocess, run:
```
python preprocess_db.py
python preprocess_review.py
```

Warning: `preprocess_review.py` will take a long time to finish.

To train classifier, run:
```
python classifier.py
```

The trained model will be saved in `model1.data`, `model2.data` and `tfidf-model.data` in `data` folder.

To detect fake reviews from a yelp page, call (in python):
```py
import pageparser
pageparser.detect_fake_review(url)
```

To call pre-trained classifier, call (in python):
```py
import classifier
classifier.load_model()
classify_result=classifier.check(reviews)
# see classifier.py for the structure of `reviews`
```

You can run:
```
python pageparser.py
```

To run a built-in fake reviews detection unit test. 

For debug: the result will be saved in:
```
./data/preprocessed/yelppage.json
./data/preprocessed/new_yelppage.htm
```

## 1.3. Website



## 1.4. Browser Extension (Live demo)

The extension is consisted of two parts: the Chrome extension client side and the python flask server side. The client side will collect info (such as url) when user are browsing yelp and send them to the server side. The server side will do query based on the data from client side, and return the prediction result. Then the client will update the website (point out which review is a potential fake review) according to the server's prediction. 

In real world, this extension should query result from server database, and will use Ajax to update webpage dynamically. But for simplify, the server will now crawl data and run classifier when received a query request, and the chrome extension (client) will then update (replace, in fact) the yelp page. 

Now the server will be setup at `Localhost:5000`. 


### 1.4.1. Usage

#### 1.4.1.1. Server Side

```py
# in python 2.7 environment, run:
python server.py
```

The server will output the result in stdout when receiving requests from the extension. 

#### 1.4.1.2. Client Side

```
# input the following URL in Chrome's multifunction bar.
chrome://extensions/

# enable developer mode
# `Load Unpacked`
# choose directory: `......\Yhelper\extension`

# This extension is only available when you are visiting yelp detail page. 
# Click the extension icon and click run script button. 
# It will take some time for the server to crawl and run the classifier.
```

# 2. DOC

See `./doc` in root folder.