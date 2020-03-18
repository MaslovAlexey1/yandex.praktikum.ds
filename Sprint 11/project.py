import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.utils import resample
import re
import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preparing():
    path = Path().absolute()
    df = pd.read_csv('{}/datasets/toxic_comments.csv'.format(path))
    return df

def train_valid_test_split(df):
    features_train, features_test, target_train, target_test = train_test_split(df['lemm_text'], df['toxic'], test_size = 0.2, random_state=12345)
    features_train, features_valid, target_train, target_valid = train_test_split(features_train, target_train, test_size = 0.25, random_state=12345)
    return features_train, features_valid, features_test, target_train, target_valid, target_test

def train_valid_test_split_df(df):
    df_train, df_test = train_test_split(df, test_size = 0.2, random_state=12345)
    df_train, df_valid = train_test_split(df_train, test_size = 0.25, random_state=12345)
    return df_train, df_valid, df_test

def features_target_split(df):
    features = df['lemm_text']
    target = df['toxic']
    return features, target

def lemmatize(m, text):
    text = clear_text(text)
    token_words=word_tokenize(text)
    lemm_text = []
    for word in token_words:
        lemm_text.append(m.lemmatize(word, pos='v'))
    return " ".join(lemm_text)

def clear_text(text):
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    return " ".join(text.split())

def downsample(df, ratio):
    df_toxic = df[df['toxic'] == 1]
    df_not_toxic = df[df['toxic'] == 0]
    sample_size = int(df_not_toxic.shape[0] * ratio)
    df_not_toxic = df_not_toxic.sample(sample_size, random_state=12345)
    df_downsampled = pd.concat([df_toxic, df_not_toxic]) 
    return df_downsampled

def upsample(df):
    df_toxic = df[df['toxic'] == 1]
    df_not_toxic = df[df['toxic'] == 0]
    df_toxic = resample(df_toxic, replace=True, n_samples=df_not_toxic.shape[0], random_state=12345)
    df_upsampled = pd.concat([df_toxic, df_not_toxic]).sample(frac=1, random_state=12345).reset_index(drop=True)
    return df_upsampled

    

    
df = preparing()
m = WordNetLemmatizer()
df['lemm_text'] = df['text'].apply(lambda x: lemmatize(m, x))
count_tf_idf = TfidfVectorizer(stop_words=stop_words)

df_train, df_valid, df_test = train_valid_test_split_df(df)
features_train, target_train =  features_target_split(df_train)
features_valid, target_valid = features_target_split(df_valid)
features_test, target_test = features_target_split(df_test)

corpus_train = features_train.values
count_tf_idf.fit(corpus_train)
tf_idf_train = count_tf_idf.transform(corpus_train)

corpus_valid = features_valid.values
tf_idf_valid = count_tf_idf.transform(corpus_valid)




lr_model = LogisticRegression(solver='liblinear')
lr_model.fit(tf_idf_train, target_train)
predictions = lr_model.predict(tf_idf_valid)
f1_score(target_valid, predictions)

lr_model = LogisticRegression(solver='liblinear', class_weight='balanced')
lr_model.fit(tf_idf_train, target_train)
predictions = lr_model.predict(tf_idf_valid)
f1_score(target_valid, predictions)

rf_model = RandomForestClassifier(n_estimators = 20, random_state=12345)
rf_model.fit(tf_idf_train, target_train)
predictions = rf_model.predict(tf_idf_valid)
f1_score(target_valid, predictions)

dt_model = DecisionTreeClassifier(min_samples_leaf= 30, random_state=12345)
dt_model.fit(tf_idf_train, target_train)
predictions = dt_model.predict(tf_idf_valid)
f1_score(target_valid, predictions)



df.lemm_text.values

count_tf_idf.get_feature_names
print(count_tf_idf.get_feature_names())
tf_idf_train.shape
count_tf_idf = TfidfVectorizer()

df_train, df_valid, df_test = train_valid_test_split_df(df)
features_train, target_train =  features_target_split(df_train)
features_valid, target_valid = features_target_split(df_valid)
features_test, target_test = features_target_split(df_test)

corpus_train = features_train.values

count_tf_idf = TfidfVectorizer(stop_words=stop_words, ngram_range=(3, 3))
count_tf_idf.fit(corpus_train)
tf_idf_train = count_tf_idf.transform(corpus_train)
# tf_idf_train.shape
# count_tf_idf.get_feature_names()

corpus_valid = features_valid.values
tf_idf_valid = count_tf_idf.transform(corpus_valid)




lr_model = LogisticRegression(solver='liblinear')
lr_model.fit(tf_idf_train, target_train)
predictions = lr_model.predict(tf_idf_valid)
f1_score(target_valid, predictions)

lr_model = LogisticRegression(solver='sag')
lr_model.fit(tf_idf_train, target_train)
predictions = lr_model.predict(tf_idf_valid)
f1_score(target_valid, predictions)


lr_model = LogisticRegression(solver='liblinear', class_weight='balanced')
lr_model.fit(tf_idf_train, target_train)
predictions = lr_model.predict(tf_idf_valid)
f1_score(target_valid, predictions)

rf_model = RandomForestClassifier(n_estimators = 5, random_state=12345)
rf_model.fit(tf_idf_train, target_train)
predictions = rf_model.predict(tf_idf_valid)
f1_score(target_valid, predictions)

dt_model = DecisionTreeClassifier(random_state=12345, class_weight='balanced')
dt_model.fit(tf_idf_train, target_train)
predictions = dt_model.predict(tf_idf_valid)
f1_score(target_valid, predictions)


df_train[df_train['toxic'] == 1]

stop_words