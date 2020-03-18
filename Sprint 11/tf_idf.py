import pandas as pd 
from pathlib import Path 
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords as nltk_stopwords
import nltk # nltk.download('stopwords')

path = Path().absolute()

data_train = pd.read_csv('{}/datasets/tweets_lemm_train.csv'.format(path))
train_corpus = data_train['lemm_text'].values.astype('U')
train_labels = data_train['positive']

stopwords = set(nltk_stopwords.words('russian'))
count_tf_idf = TfidfVectorizer(stop_words=stopwords)

tfidf_train = count_tf_idf.fit_transform(train_corpus)

model = LogisticRegression()
model.fit(tfidf_train, train_labels)

data_test = pd.read_csv('{}/datasets/tweets_lemm_test.csv'.format(path))
test_corpus = data_test['lemm_text'].values.astype('U')
tfidf_test = count_tf_idf.transform(test_corpus)

predictions = model.predict(tfidf_test)
submission = pd.DataFrame({'positive': predictions})

submission.to_csv('{}/datasets/predictions'.format(path), index=False)