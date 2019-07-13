# coding: utf-8
# Author: Lizi
# Liao, Liangming Pan, originaly by C.J. Hutto
# For license information, see LICENSE.TXT

# Modified by Huaqiang Wang
# vader_awmod.py

"""
If you use the VADER sentiment analysis tools, please cite:
Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for
Sentiment Analysis of Social Media Text. Eighth International Conference on
Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.
"""
import io
import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV   
import math
import re
import string
import requests
import json
import scipy
from itertools import product
from inspect import getsourcefile
from os.path import abspath, join, dirname
from sklearn.metrics import classification_report, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


import xgboost as xgb
# from mlxtend.classifier import StackingClassifier

##Constants##

# (empirically derived mean sentiment intensity rating increase for booster words)
B_INCR = 0.293
B_DECR = -0.293

# (empirically derived mean sentiment intensity rating increase for using
# ALLCAPs to emphasize a word)
C_INCR = 0.733

N_SCALAR = -0.74

# for removing punctuation
REGEX_REMOVE_PUNCTUATION = re.compile('[%s]' % re.escape(string.punctuation))

PUNC_LIST = [".", "!", "?", ",", ";", ":", "-", "'", "\"",
             "!!", "!!!", "??", "???", "?!?", "!?!", "?!?!", "!?!?"]
NEGATE = \
    ["aint", "arent", "cannot", "cant", "couldnt", "darent", "didnt", "doesnt",
     "ain't", "aren't", "can't", "couldn't", "daren't", "didn't", "doesn't",
     "dont", "hadnt", "hasnt", "havent", "isnt", "mightnt", "mustnt", "neither",
     "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't",
     "neednt", "needn't", "never", "none", "nope", "nor", "not", "nothing", "nowhere",
     "oughtnt", "shant", "shouldnt", "uhuh", "wasnt", "werent",
     "oughtn't", "shan't", "shouldn't", "uh-uh", "wasn't", "weren't",
     "without", "wont", "wouldnt", "won't", "wouldn't", "rarely", "seldom", "despite"]

# booster/dampener 'intensifiers' or 'degree adverbs'
# http://en.wiktionary.org/wiki/Category:English_degree_adverbs

BOOSTER_DICT = \
    {"absolutely": B_INCR, "amazingly": B_INCR, "awfully": B_INCR, "completely": B_INCR, "considerably": B_INCR,
     "decidedly": B_INCR, "deeply": B_INCR, "effing": B_INCR, "enormously": B_INCR,
     "entirely": B_INCR, "especially": B_INCR, "exceptionally": B_INCR, "extremely": B_INCR,
     "fabulously": B_INCR, "flipping": B_INCR, "flippin": B_INCR,
     "fricking": B_INCR, "frickin": B_INCR, "frigging": B_INCR, "friggin": B_INCR, "fully": B_INCR, "fucking": B_INCR,
     "greatly": B_INCR, "hella": B_INCR, "highly": B_INCR, "hugely": B_INCR, "incredibly": B_INCR,
     "intensely": B_INCR, "majorly": B_INCR, "more": B_INCR, "most": B_INCR, "particularly": B_INCR,
     "purely": B_INCR, "quite": B_INCR, "really": B_INCR, "remarkably": B_INCR,
     "so": B_INCR, "substantially": B_INCR,
     "thoroughly": B_INCR, "totally": B_INCR, "tremendously": B_INCR,
     "uber": B_INCR, "unbelievably": B_INCR, "unusually": B_INCR, "utterly": B_INCR,
     "very": B_INCR,
     "almost": B_DECR, "barely": B_DECR, "hardly": B_DECR, "just enough": B_DECR,
     "kind of": B_DECR, "kinda": B_DECR, "kindof": B_DECR, "kind-of": B_DECR,
     "less": B_DECR, "little": B_DECR, "marginally": B_DECR, "occasionally": B_DECR, "partly": B_DECR,
     "scarcely": B_DECR, "slightly": B_DECR, "somewhat": B_DECR,
     "sort of": B_DECR, "sorta": B_DECR, "sortof": B_DECR, "sort-of": B_DECR}

# check for special case idioms using a sentiment-laden keyword known to VADER
SPECIAL_CASE_IDIOMS = {"the shit": 3, "the bomb": 3, "bad ass": 1.5, "yeah right": -2,
                       "cut the mustard": 2, "kiss of death": -1.5, "hand to mouth": -2}


##Static methods##

def negated(input_words, include_nt=True):
    """ 
    Determine if input contains negation words
    """
    neg_words = []
    neg_words.extend(NEGATE)
    for word in neg_words:
        if word in input_words:
            return True
    if include_nt:
        for word in input_words:
            if "n't" in word:
                return True
    if "least" in input_words:
        i = input_words.index("least")
        if i > 0 and input_words[i - 1] != "at":
            return True
    return False


def normalize(score, alpha=15):
    """
    Normalize the score to be between -1 and 1 using an alpha that
    approximates the max expected value
    """
    norm_score = score / math.sqrt((score * score) + alpha)
    if norm_score < -1.0:
        return -1.0
    elif norm_score > 1.0:
        return 1.0
    else:
        return norm_score


def allcap_differential(words):
    """
    Check whether just some words in the input are ALL CAPS
    :param list words: The words to inspect
    :returns: `True` if some but not all items in `words` are ALL CAPS
    """
    is_different = False
    allcap_words = 0
    for word in words:
        if word.isupper():
            allcap_words += 1
    cap_differential = len(words) - allcap_words 
    if cap_differential > 0 and cap_differential < len(words):
        is_different = True
    return is_different


def scalar_inc_dec(word, valence, is_cap_diff):
    """
    Check if the preceding words increase, decrease, or negate/nullify the
    valence
    """
    scalar = 0.0
    word_lower = word.lower()
    if word_lower in BOOSTER_DICT:
        scalar = BOOSTER_DICT[word_lower]
        if valence < 0:
            scalar *= -1
        # check if booster/dampener word is in ALLCAPS (while others aren't)
        if word.isupper() and is_cap_diff:
            if valence > 0:
                scalar += C_INCR
            else:
                scalar -= C_INCR
    return scalar


def map_to_label(scores):
    labels = []
    for score in scores:
        if score > -0.5 and score < 0.5:
            labels.append(1)
        elif score >= 0.5:
            labels.append(2)
        elif score <= -0.5:
            labels.append(0)
    return labels


class SentiText(object):
    """
    Identify sentiment-relevant string-level properties of input text.
    """

    def __init__(self, text):
        if not isinstance(text, str):
            text = str(text.encode('utf-8'))
        self.text = text
        self.words_and_emoticons = self._words_and_emoticons()
        # doesn't separate words from\
        # adjacent punctuation (keeps emoticons & contractions)
        self.is_cap_diff = allcap_differential(self.words_and_emoticons)

    def _words_plus_punc(self):
        """
        Returns mapping of form:
        {
            'cat,': 'cat',
            ',cat': 'cat',
        }
        """
        no_punc_text = REGEX_REMOVE_PUNCTUATION.sub('', self.text)
        # removes punctuation (but loses emoticons & contractions)
        words_only = no_punc_text.split()
        # remove singletons
        words_only = set(w for w in words_only if len(w) > 1)
        # the product gives ('cat', ',') and (',', 'cat')
        punc_before = {''.join(p): p[1] for p in product(PUNC_LIST, words_only)}
        punc_after = {''.join(p): p[0] for p in product(words_only, PUNC_LIST)}
        words_punc_dict = punc_before
        words_punc_dict.update(punc_after)
        return words_punc_dict

    def _words_and_emoticons(self):
        """
        Removes leading and trailing puncutation
        Leaves contractions and most emoticons
            Does not preserve punc-plus-letter emoticons (e.g. :D)
        """
        wes = self.text.split()
        words_punc_dict = self._words_plus_punc()
        wes = [we for we in wes if len(we) > 1]
        for i, we in enumerate(wes):
            if we in words_punc_dict:
                wes[i] = words_punc_dict[we]
        return wes
        #不是很懂为什么要这样替换. 直接按C的方式检查字符串不行吗....


class SentimentIntensityAnalyzer(object):
    """
    Give a sentiment intensity score to sentences.
    """

    def __init__(self, lexicon_file="sentiment_lexicon.txt"):
        _this_module_file_path_ = abspath(getsourcefile(lambda: 0))
        lexicon_full_filepath = join(dirname(_this_module_file_path_), lexicon_file)
        with io.open(lexicon_full_filepath, encoding='utf-8') as f:
            self.lexicon_full_filepath = f.read()
        self.lexicon = self.make_lex_dict()

    def make_lex_dict(self):
        """
        Convert lexicon file to a dictionary
        """
        lex_dict = {}
        for line in self.lexicon_full_filepath.split('\n'):
            (word, measure) = line.strip().split('\t')[0:2]
            lex_dict[word] = float(measure)
        return lex_dict

    def polarity_scores(self, text):
        """
        Return a float for sentiment strength based on the input text.
        Positive values are positive valence, negative value are negative
        valence.
        """
        sentitext = SentiText(text)
        #text, words_and_emoticons, is_cap_diff = self.preprocess(text)

        sentiments = []
        words_and_emoticons = sentitext.words_and_emoticons
        for item in words_and_emoticons:
            valence = 0
            i = words_and_emoticons.index(item)
            #FIXIT: 这里只处理了单独的词语, 连续的词语只处理了 kind of
            if (i < len(words_and_emoticons) - 1 and item.lower() == "kind" and
                    words_and_emoticons[i + 1].lower() == "of") or \
                    item.lower() in BOOSTER_DICT:
                sentiments.append(valence)
                continue

            sentiments = self.sentiment_valence(valence, sentitext, item, i, sentiments)

        sentiments = self._but_check(words_and_emoticons, sentiments)

        valence_dict = self.score_valence(sentiments, text)

        return valence_dict

    def sentiment_valence(self, valence, sentitext, item, i, sentiments):
        # i 为当前处理到的位置 
        is_cap_diff = sentitext.is_cap_diff #if all words are CAPS
        words_and_emoticons = sentitext.words_and_emoticons
        item_lowercase = item.lower()
        if item_lowercase in self.lexicon:
            # get the sentiment valence
            valence = self.lexicon[item_lowercase]

            # check if sentiment laden word is in ALL CAPS (while others aren't)
            # if not, then strengthen its valence:
            if item.isupper() and is_cap_diff:
                if valence > 0:
                    valence += C_INCR
                else:
                    valence -= C_INCR

            for start_i in range(0, 3):
                # start_i 是在i前的单词. start_i==0时, 是之前的第一个单词
                if i > start_i and words_and_emoticons[i - (start_i + 1)].lower() not in self.lexicon:
                    # dampen the scalar modifier of preceding words and emoticons
                    # (excluding the ones that immediately preceed the item) based
                    # on their distance from the current item.
                    s = scalar_inc_dec(words_and_emoticons[i - (start_i + 1)], valence, is_cap_diff)
                    # 越远, 修饰词的效力越有限
                    if start_i == 1 and s != 0:
                        s = s * 0.95
                    if start_i == 2 and s != 0:
                        s = s * 0.9
                    valence = valence + s
                    valence = self._never_check(valence, words_and_emoticons, start_i, i)
                    if start_i == 2:
                        valence = self._idioms_check(valence, words_and_emoticons, i)

                        # future work: consider other sentiment-laden idioms
                        # other_idioms =
                        # {"back handed": -2, "blow smoke": -2, "blowing smoke": -2,
                        #  "upper hand": 1, "break a leg": 2,
                        #  "cooking with gas": 2, "in the black": 2, "in the red": -2,
                        #  "on the ball": 2,"under the weather": -2}

            valence = self._least_check(valence, words_and_emoticons, i)

        sentiments.append(valence)
        return sentiments

    def _least_check(self, valence, words_and_emoticons, i):
        # check for negation case using "least"
        if i > 1 and words_and_emoticons[i - 1].lower() not in self.lexicon \
           and words_and_emoticons[i - 1].lower() == "least":
            if words_and_emoticons[i - 2].lower() != "at" and words_and_emoticons[i - 2].lower() != "very":
                valence = valence * N_SCALAR
        elif i > 0 and words_and_emoticons[i - 1].lower() not in self.lexicon \
                and words_and_emoticons[i - 1].lower() == "least":
            valence = valence * N_SCALAR
        return valence

    def _but_check(self, words_and_emoticons, sentiments):
        # check for modification in sentiment due to contrastive conjunction 'but'
        if 'but' in words_and_emoticons or 'BUT' in words_and_emoticons:
            try:
                bi = words_and_emoticons.index('but')
            except ValueError:
                bi = words_and_emoticons.index('BUT')
            for sentiment in sentiments:
                si = sentiments.index(sentiment)
                if si < bi:
                    sentiments.pop(si)
                    sentiments.insert(si, sentiment * 0.5)
                elif si > bi:
                    sentiments.pop(si)
                    sentiments.insert(si, sentiment * 1.5)
        return sentiments

    def _idioms_check(self, valence, words_and_emoticons, i):
        onezero = "{0} {1}".format(words_and_emoticons[i - 1], words_and_emoticons[i])

        twoonezero = "{0} {1} {2}".format(words_and_emoticons[i - 2],
                                          words_and_emoticons[i - 1], words_and_emoticons[i])

        twoone = "{0} {1}".format(words_and_emoticons[i - 2], words_and_emoticons[i - 1])

        threetwoone = "{0} {1} {2}".format(words_and_emoticons[i - 3],
                                           words_and_emoticons[i - 2], words_and_emoticons[i - 1])

        threetwo = "{0} {1}".format(words_and_emoticons[i - 3], words_and_emoticons[i - 2])

        sequences = [onezero, twoonezero, twoone, threetwoone, threetwo]

        for seq in sequences:
            if seq in SPECIAL_CASE_IDIOMS:
                valence = SPECIAL_CASE_IDIOMS[seq]
                break

        if len(words_and_emoticons) - 1 > i:
            zeroone = "{0} {1}".format(words_and_emoticons[i], words_and_emoticons[i + 1])
            if zeroone in SPECIAL_CASE_IDIOMS:
                valence = SPECIAL_CASE_IDIOMS[zeroone]
        if len(words_and_emoticons) - 1 > i + 1:
            zeroonetwo = "{0} {1} {2}".format(words_and_emoticons[i], words_and_emoticons[i + 1], words_and_emoticons[i + 2])
            if zeroonetwo in SPECIAL_CASE_IDIOMS:
                valence = SPECIAL_CASE_IDIOMS[zeroonetwo]

        # check for booster/dampener bi-grams such as 'sort of' or 'kind of'
        if threetwo in BOOSTER_DICT or twoone in BOOSTER_DICT:
            valence = valence + B_DECR
        return valence

    def _never_check(self, valence, words_and_emoticons, start_i, i):
        if start_i == 0:
            if negated([words_and_emoticons[i - 1]]):
                valence = valence * N_SCALAR
        if start_i == 1:
            if words_and_emoticons[i - 2] == "never" and\
               (words_and_emoticons[i - 1] == "so" or
                    words_and_emoticons[i - 1] == "this"):
                valence = valence * 1.5
            elif negated([words_and_emoticons[i - (start_i + 1)]]):
                valence = valence * N_SCALAR
        if start_i == 2:
            if words_and_emoticons[i - 3] == "never" and \
               (words_and_emoticons[i - 2] == "so" or words_and_emoticons[i - 2] == "this") or \
               (words_and_emoticons[i - 1] == "so" or words_and_emoticons[i - 1] == "this"):
                valence = valence * 1.25
            elif negated([words_and_emoticons[i - (start_i + 1)]]):
                valence = valence * N_SCALAR
        return valence

    def _punctuation_emphasis(self, sum_s, text):
        # add emphasis from exclamation points and question marks
        ep_amplifier = self._amplify_ep(text)
        qm_amplifier = self._amplify_qm(text)
        punct_emph_amplifier = ep_amplifier + qm_amplifier
        return punct_emph_amplifier

    def _amplify_ep(self, text):
        # check for added emphasis resulting from exclamation points (up to 4 of them)
        ep_count = text.count("!")
        if ep_count > 4:
            ep_count = 4
        # (empirically derived mean sentiment intensity rating increase for
        # exclamation points)
        ep_amplifier = ep_count * 0.292
        return ep_amplifier

    def _amplify_qm(self, text):
        # check for added emphasis resulting from question marks (2 or 3+)
        qm_count = text.count("?")
        qm_amplifier = 0
        if qm_count > 1:
            if qm_count <= 3:
                # (empirically derived mean sentiment intensity rating increase for
                # question marks)
                qm_amplifier = qm_count * 0.18
            else:
                qm_amplifier = 0.96
        return qm_amplifier

    def _sift_sentiment_scores(self, sentiments):
        # want separate positive versus negative sentiment scores
        pos_sum = 0.0
        neg_sum = 0.0
        neu_count = 0
        for sentiment_score in sentiments:
            if sentiment_score > 0:
                pos_sum += (float(sentiment_score) + 1)  # compensates for neutral words that are counted as 1
            if sentiment_score < 0:
                neg_sum += (float(sentiment_score) - 1)  # when used with math.fabs(), compensates for neutrals
            if sentiment_score == 0:
                neu_count += 1
        return pos_sum, neg_sum, neu_count

    def score_valence(self, sentiments, text):
        if sentiments:
            sum_s = float(sum(sentiments))
            # compute and add emphasis from punctuation in text
            punct_emph_amplifier = self._punctuation_emphasis(sum_s, text)
            if sum_s > 0:
                sum_s += punct_emph_amplifier
            elif sum_s < 0:
                sum_s -= punct_emph_amplifier

            compound = normalize(sum_s)
            # discriminate between positive, negative and neutral sentiment scores
            pos_sum, neg_sum, neu_count = self._sift_sentiment_scores(sentiments)

            if pos_sum > math.fabs(neg_sum):
                pos_sum += (punct_emph_amplifier)
            elif pos_sum < math.fabs(neg_sum):
                neg_sum -= (punct_emph_amplifier)

            total = pos_sum + math.fabs(neg_sum) + neu_count
            pos = math.fabs(pos_sum / total)
            neg = math.fabs(neg_sum / total)
            neu = math.fabs(neu_count / total)

        else:
            compound = 0.0
            pos = 0.0
            neg = 0.0
            neu = 0.0

        sentiment_dict = \
            {"neg": round(neg, 3),
             "neu": round(neu, 3),
             "pos": round(pos, 3),
             "compound": round(compound, 4)}

        return sentiment_dict


if __name__ == '__main__':
    # --- examples -------
    sentences = ["You are smart, handsome, and funny.",      # positive sentence example
                 "You are not smart, handsome, nor funny.",   # negation sentence example
                 "You are smart, handsome, and funny!",       # punctuation emphasis handled correctly (sentiment intensity adjusted)
                 "You are very smart, handsome, and funny.",  # booster words handled correctly (sentiment intensity adjusted)
                 "You are VERY SMART, handsome, and FUNNY.",  # emphasis for ALLCAPS handled
                 "You are VERY SMART, handsome, and FUNNY!!!",  # combination of signals - VADER appropriately adjusts intensity
                 "You are VERY SMART, uber handsome, and FRIGGIN FUNNY!!!",  # booster words & punctuation make this close to ceiling for score
                 "The book was good.",         # positive sentence
                 "The book was kind of good.",  # qualified positive sentence is handled correctly (intensity adjusted)
                 "The plot was good, but the characters are uncompelling and the dialog is not great.",  # mixed negation sentence
                 "At least it isn't a horrible book.",  # negated negative sentence with contraction
                 "Make sure you :) or :D today!",     # emoticons handled
                 "Today SUX!",  # negative slang with capitalization emphasis
                 "Today only kinda sux! But I'll get by, lol"  # mixed sentiment example with slang and constrastive conjunction "but"
                 ]

    classifier = SentimentIntensityAnalyzer()

    print("----------------------------------------------------")
    print(" - Analyze typical example cases, including handling of:")
    print("  -- negations")
    print("  -- punctuation emphasis & punctuation flooding")
    print("  -- word-shape as emphasis (capitalization difference)")
    print("  -- degree modifiers (intensifiers such as 'very' and dampeners such as 'kind of')")
    print("  -- slang words as modifiers such as 'uber' or 'friggin' or 'kinda'")
    print("  -- contrastive conjunction 'but' indicating a shift in sentiment; sentiment of later text is dominant")
    print("  -- use of contractions as negations")
    print("  -- sentiment laden emoticons such as :) and :D")
    print("  -- sentiment laden slang words (e.g., 'sux')")
    print("  -- sentiment laden initialisms and acronyms (for example: 'lol') \n")
    print("----------------------------------------------------")
    print('  -- Score scheme -- ')
    print('  * negative sentiment: compound score in [-1, -0.5]')
    print('  * neutral sentiment: compound score in (-0.5, 0.5)')
    print('  * positive sentiment: compound score in [0.5, 1]')
    print("""  -- The 'compound' score is computed by summing the valence scores of each word in the lexicon, adjusted
     according to the rules, and then normalized to be between -1 (most extreme negative) and +1 (most extreme positive).
     This is the most useful metric if you want a single unidimensional measure of sentiment for a given sentence.
     Calling it a 'normalized, weighted composite score' is accurate.""")
    print("""  -- The 'pos', 'neu', and 'neg' scores are ratios for proportions of text that fall in each category (so these
     should all add up to be 1... or close to it with float operation).  These are the most useful metrics if
     you want multidimensional measures of sentiment for a given sentence.""")
    print("----------------------------------------------------")
    print('  -- Example Results -- ')
    for sentence in sentences:
        vs = classifier.polarity_scores(sentence)
        print("{:-<65} {}".format(sentence, str(vs)))

    #print("Press Enter to continue analyzing tweets data...")  # for DEMO purposes...

    print("----------------------------------------------------")

    data_dir = './data'

    print("Loading data...")
    with open(os.path.join(data_dir, 'tweets_processed.txt'), 'r') as f:
        x = np.array(f.readlines())

    extra_infos=[]
    with open(os.path.join(data_dir, 'sentiment_extrainfo.txt'), 'r') as f:
        for line in f.readlines():
            extra_infos.append(json.loads(line))

    log_extra_array=[]
    for i in extra_infos:
        # log_extra_array.append([i['favorite_count_log'],i['retweet_count_log'],i['friends_count_log'],i['followers_count_log']])
        # log_extra_array.append([i['retweet_count_log']])
        # log_extra_array.append([i['favorite_count_log']])
        log_extra_array.append([i['friends_count_log']])
        # log_extra_array.append([i['followers_count_log']])
        # log_extra_array.append([i['followers_count_log'],i['friends_count_log']])
        # log_extra_array.append([])
    log_extra_array=np.array(log_extra_array)

    with open(os.path.join(data_dir, 'labels.txt'), 'r') as f:
        y = np.array([int(line.strip()) for line in f.readlines()])
        # print(y)
        # labels for training emoji data
        # if a tweet does not have emoji, it will be classified as 4
        emoji_sp_label=[]
        for a,b in zip(extra_infos, y):
            if(len(a['emoji'])==0):
                emoji_sp_label.append(4)
            else:
                emoji_sp_label.append(b)
        emoji_sp_label=np.array(emoji_sp_label)
        print(emoji_sp_label)

    with open(os.path.join(data_dir, 'emoji.txt'), 'r') as f:
        emoji = np.array(f.readlines())


    print("Vectorize Emoji...")
    emoji_feats = TfidfVectorizer().fit_transform(emoji)
    # emoji_feats = CountVectorizer().fit_transform(emoji)
    emoji_feats=scipy.sparse.bmat([[emoji_feats]]).tocsr()

    print("Start training and predict...")
    kf = KFold(n_splits=10)
    avg_p = 0
    avg_r = 0
    for train, test in kf.split(y):
        # print(x[test])
        # print(y[test])
        # print(emoji_sp_label[test])

    # train:
    # train-emoji
        # you can use the train data to train your classifiers
        # new_classifier = new_classifiler_model.fit(x[train], y[train])
        # then apply to the test data as below
        
        # have_emoji_feats=np.array([]) 
        # have_y=np.array([])
        # for e, t, r in zip(emoji[train], emoji_feats[train], y[train]):
        #     if(e!='NoEmoji'):
        #         have_emoji_feats.append(t)
        #         have_y.append(r)

        # emoji_model = MultinomialNB().fit(emoji_feats[train], emoji_sp_label[train]) 
        emoji_model = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42,max_iter=4, tol=None).fit(emoji_feats[train], emoji_sp_label[train]) # SVM (with SGD)
    # train-others
        # log_extra_array
        # extra_model = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42,max_iter=4, tol=None).fit(log_extra_array[train], y[train]) # SVM (with SGD)
        
        
    # train xgboost as Stacking Classifier
        predict_scores = []
        for instance in x[train]:
            predict_scores.append(classifier.polarity_scores(instance)["compound"])
        text_predicts = map_to_label(predict_scores)

        # Emoji
        emoji_predicts = emoji_model.predict(emoji_feats[train])

        # Likes, Retweets
        # extra_predicts = extra_model.predict(log_extra_array[train])

        # predict=stacking results
        stacking_input=[]
        log_extra_array_train=[]
        for i in log_extra_array[train]:
            log_extra_array_train.append(i)
        for a,b,c in zip(predict_scores, emoji_predicts, log_extra_array_train):
            line=[]
            line.append(a)
            line.append(b)
            line.extend(c)
            stacking_input.append(line)
        stacking_array=np.array(stacking_input)

        # print(len(text_predicts))
        
        # prepare xgb classifier to combine the 3 classifiers 
        # xgb_model = xgb.XGBRegressor(objective='multi:softmax',num_class=3).fit(stacking_array, y[train])
        xgb_model = xgb.XGBRegressor(objective='multi:softmax',num_class=3, max_depth=5, gamma= 0,eta=0.001).fit(stacking_array, y[train])
        
        # the following code is used for tuning parameter for xgboost
        # stacking_array=np.array(stacking_input)
        # cv_params = {
        #     'gamma': [0],                  
        #     'max_depth': [5,7,10],               
        #     'eta': [0.001, 0.005,0.01,0.1]
        #     }

        # other_params = {'objective' : 'multi:softmax', 'num_class':3 }
        # model = xgb.XGBRegressor(**other_params)
        # # (max_depth=3, learning_rate=0.1, n_estimators=100, 
        # # silent=True, objective='reg:linear', booster='gbtree',
        # #  n_jobs=1, nthread=None, gamma=0, min_child_weight=1, 
        # # max_delta_step=0, subsample=1, colsample_bytree=1,
        # #  colsample_bylevel=1, reg_alpha=0, reg_lambda=1, 
        # # scale_pos_weight=1, base_score=0.5, random_state=0, 
        # # seed=None, missing=None, importance_type='gain', **kwargs)

        # optimized_GBM = GridSearchCV(estimator=xgb_model, param_grid=cv_params, scoring='r2', cv=5, verbose=1, n_jobs=4)# is r2 ok?
        # optimized_GBM.fit(stacking_array, y[train])
        # # evalute_result = gbm_result.grid_scores_
        # print((optimized_GBM.best_params_))
        # print((optimized_GBM.best_score_))

        # exit(0)
        # para-tune ends here

    # test:
        # Text only
        predict_scores = []
        for instance in x[test]:
            predict_scores.append(classifier.polarity_scores(instance)["compound"])
        text_predicts = map_to_label(predict_scores)

        # Emoji
        emoji_predicts = emoji_model.predict(emoji_feats[test])

        # Likes, Retweets
        # extra_predicts = extra_model.predict(log_extra_array[test])

        # Ensamble Classifier
        # predicts just add emoji results

        predicts =[]

        # the 1st version of model
        # for a, b in zip(text_predicts, emoji_predicts):
        #     if(b==2):
        #         predicts.append(2)
        #     else:
        #         predicts.append(a)

        # predicts=text_predicts
        # predicts=extra_predicts

        # predict=stacking results
        stacking_input=[]
        log_extra_array_train=[]
        for i in log_extra_array[test]:
            log_extra_array_train.append(i)
        for a,b,c in zip(predict_scores, emoji_predicts, log_extra_array_train):
            line=[]
            line.append(a)
            line.append(b)
            line.extend(c)
            stacking_input.append(line)
        stacking_array=np.array(stacking_input)

        # using xgb classifier to combine the 3 classifiers 
        predicts=xgb_model.predict(stacking_array)
        # print(predicts)

        print(classification_report(y[test], predicts))
        # print(classification_report(emoji_sp_label[test], emoji_predicts))
        # avg_p += precision_score(emoji_sp_label[test], predicts, average='macro')
        # avg_r += recall_score(emoji_sp_label[test], predicts, average='macro')
        avg_p += precision_score(y[test], predicts, average='macro')
        avg_r += recall_score(y[test], predicts, average='macro')

    print('Average Precision is %f.' % (avg_p / 10.0))
    print('Average Recall is %f.' % (avg_r / 10.0))
