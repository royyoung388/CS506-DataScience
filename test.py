import string
from collections import Counter

from nltk.corpus import wordnet as wn

import matplotlib.pyplot as plt
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from nltk.corpus import sentiwordnet as swn

nltk.download('sentiwordnet')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

counter = Counter()
stemmer = SnowballStemmer("english")
wl = WordNetLemmatizer()

# word = 'overrated'
# print(stemmer.stem(word))
# # w = wl.lemmatize(stemmer.stem(word))
w = word_tokenize('I love u')
print(w)
w = swn.senti_synsets("n't")
print(w.pos_score())
print(w.neg_score())
wn.synset_from_pos_and_offset