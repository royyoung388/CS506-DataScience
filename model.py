import os.path
import pickle
import string
from collections import Counter

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB

nltk.download('stopwords')


def process_text(df):
    df['Summary'].fillna('', inplace=True)
    df['Text'].fillna('', inplace=True)
    df['Summary'] = df['Summary'].apply(stem_sent)
    df['Text'] = df['Text'].apply(stem_sent)
    return df


def stem_sent(text):
    if not text:
        return

    stemmer = SnowballStemmer('english')
    text = text.lower()

    words = word_tokenize(text)
    result = []
    for w in words:
        # remove all punctuation
        if w in string.punctuation or w in stopwords.words('english'):
            continue
        result.append(stemmer.stem(w))
    return ' '.join(result)


def calc_counter(text, vocab_counter):
    counter = Counter(text.split())
    vocab_counter += counter


def one_hot(text, vocabulary):
    vec = [0] * (len(vocabulary) + 1)
    counter = Counter(text.split())
    for k, v in counter.items():
        if k in vocabulary:
            vec[vocabulary[k]] = v
        else:
            vec[0] += v
    return vec


def convert2Id(text, id_dict):
    if text in id_dict:
        return id_dict[text]
    else:
        return 0


if __name__ == '__main__':
    # Load files into DataFrames
    print('loading data...')
    if os.path.exists('./data/X_train_stem.csv'):
        X_train = pd.read_csv("./data/X_train_stem.csv", index_col=0)
        X_submission = pd.read_csv("./data/X_test_stem.csv", index_col=0)
        X_train[['Summary', 'Text']] = X_train[['Summary', 'Text']].astype(str)
        X_submission[['Summary', 'Text']] = X_submission[['Summary', 'Text']].astype(str)
    else:
        X_train = pd.read_csv("./data/X_train.csv", index_col=0)
        X_submission = pd.read_csv("./data/X_test.csv", index_col=0)
        X_train[['Summary', 'Text']] = X_train[['Summary', 'Text']].astype(str)
        X_submission[['Summary', 'Text']] = X_submission[['Summary', 'Text']].astype(str)
        # stem word
        print('stem data')
        X_train = process_text(X_train)
        X_train.to_csv("./data/X_train_stem.csv")
        X_submission = process_text(X_submission)
        X_submission.to_csv("./data/X_test_stem.csv")

    X_train = X_train.iloc[:100]

    # Split training set into training and testing set
    # X_train, X_test, Y_train, Y_test = train_test_split(
    #     X_train.drop(['Score'], axis=1),
    #     X_train['Score'],
    #     test_size=1 / 4.0,
    #     random_state=0
    # )

    # make vocabulary
    # print('building vocabulary...')
    # if os.path.exists('./data/vocabulary.pickle'):
    #     with open("./data/counter.pickle", "rb") as f:
    #         vocab_counter = pickle.load(f)
    #     with open("./data/vocabulary.pickle", "rb") as f:
    #         vocabulary = pickle.load(f)
    # else:
    #     vocab_counter = Counter()
    #     X_train['Summary'].apply(calc_counter, args=(vocab_counter,))
    #     X_train['Text'].apply(calc_counter, args=(vocab_counter,))
    #     # vocabulary index start from 1. 0 used for any unknown word
    #     vocabulary = dict(zip(list(vocab_counter), range(1, len(vocab_counter) + 1)))
    #     print(vocab_counter.most_common(500))
    #     print(list(sorted(vocabulary.items(), key=lambda x: x[1]))[:500])
    #     with open('./data/counter.pickle', 'wb') as f:
    #         pickle.dump(vocab_counter, f)
    #     with open('./data/vocabulary.pickle', 'wb') as f:
    #         pickle.dump(vocabulary, f)

    # # make word one-hot
    # print('building word ont-hot')
    # X_train['OnehotS'] = X_train['Summary'].apply(one_hot, args=(vocabulary,))
    # X_train['OnehotT'] = X_train['Text'].apply(one_hot, args=(vocabulary,))

    # build TFIDF
    print("building TFIDF")
    vectorizeS = TfidfVectorizer(stop_words={'english'}, min_df=10)
    vectorizeT = TfidfVectorizer(stop_words={'english'}, min_df=10)
    tfidfS = vectorizeS.fit_transform(X_train['Summary'])
    X_train['TfidfT'] = vectorizeT.fit_transform(X_train['Text']).toarray()

    # extract year feature
    print('extract year feature')
    X_train['Year'] = pd.to_datetime(X_train['Time'], unit='s').dt.year

    # convert productId and userId to int id
    print("convert product id and user id to int id")
    if os.path.exists('product2Id.pickle'):
        with open('./data/product2Id.pickle', 'wb') as f1, open('./data/user2Id.pickle', 'wb') as f2:
            product2Id = pickle.load(f1)
            User2Id = pickle.load(f2)
    else:
        uniqueP = X_train['ProductId'].unique()
        uniqueS = X_train['UserId'].unique()

        product2Id = dict(zip(uniqueP.tolist(), range(1, len(uniqueP) + 1)))
        User2Id = dict(zip(uniqueS.tolist(), range(1, len(uniqueS) + 1)))

        with open('./data/product2Id.pickle', 'wb') as f1, open('./data/user2Id.pickle', 'wb') as f2:
            pickle.dump(product2Id, f1)
            pickle.dump(User2Id, f2)

    X_train["ProductId"] = X_train["ProductId"].apply(convert2Id, args=(product2Id,))
    X_train["UserId"] = X_train["UserId"].apply(convert2Id, args=(User2Id,))
    X_train.head()

    # train naive bayes
    # Split training set into training and testing set
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_train.drop(['Score'], axis=1),
        X_train['Score'],
        test_size=1 / 4.0,
        random_state=0
    )

    # train model
    print('training naive bayes')
    cnb_s = ComplementNB()
    cnb_t = ComplementNB()

    cnb_s.fit(X_train['TfidfT'], Y_train)
    score = cnb_s.score(X_test['TfidfT'], Y_test)
    print(score)
