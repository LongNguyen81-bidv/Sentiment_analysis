# import libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import underthesea as uts
from wordcloud import WordCloud,STOPWORDS
# import pandas_profiling
# import scikit-learn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import glob
import pickle

# Part 1: Build project

# load data
df = pd.read_csv('processed_data.csv', encoding='utf-8')

# Data pre - processing

# remove duplicate
df.drop_duplicates(inplace=True)

# remove missing values
df.dropna(inplace=True)



##LOAD EMOJICON
file = open('files/emojicon.txt', 'r', encoding="utf8")
emoji_lst = file.read().split('\n')
emoji_dict = {}
for line in emoji_lst:
    key, value = line.split('\t')
    emoji_dict[key] = str(value)
file.close()
#################
#LOAD TEENCODE
file = open('files/teencode.txt', 'r', encoding="utf8")
teen_lst = file.read().split('\n')
teen_dict = {}
for line in teen_lst:
    key, value = line.split('\t')
    teen_dict[key] = str(value)
file.close()
###############
#LOAD TRANSLATE ENGLISH -> VNMESE
file = open('files/english-vnmese.txt', 'r', encoding="utf8")
english_lst = file.read().split('\n')
english_dict = {}
for line in english_lst:
    key, value = line.split('\t')
    english_dict[key] = str(value)
file.close()
################
#LOAD wrong words
file = open('files/wrong-word.txt', 'r', encoding="utf8")
wrong_lst = file.read().split('\n')
file.close()
#################
#LOAD STOPWORDS
file = open('files/vietnamese-stopwords.txt', 'r', encoding="utf8")
stopwords_lst = file.read().split('\n')
file.close()


from Library_functions_NLP_TV import *

# load tfidf vectorizer
vectorizer = pickle.load(open('tfidf.pkl', 'rb'))


# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(df['comment'], df['sentiment'], test_size=0.2, random_state=42)

# transform data
X_test_tfidf = vectorizer.transform(X_test)

# load model
model = pickle.load(open('finalized_model_2.sav', 'rb'))

y_pred = model.predict(X_test_tfidf)
cm = confusion_matrix(y_test, y_pred)


# Part 2: Build app

# Title
st.image('download.jpg')

st.title("Trung tâm tin học - ĐH KHTN")
st.header('Data Science and Machine Learning Certificate')
st.markdown('#### Project: Sentiment analysis of Vietnamese comments on Shopee')

# st.video('https://www.youtube.com/watch?v=q3nSSZNOg38&list=PLFTWPHJsZXVVnckL0b3DYmHjjPiRB5mqX')

menu = ['Overview',
        'Build Model', 
        'New Prediction']
choice = st.sidebar.selectbox('Menu', menu)

if choice == 'Overview':
    st.subheader('Overview')
    
    st.write('''
    This project is about sentiment analysis of Vietnamese comments on Shopee. 
    The dataset is collected from Shopee website. 
    The dataset contains 2 columns: rating and comment. 
    The rating is from 1 to 5. 
    The comment is the comment of customers after they buy products on Shopee. 
    The goal of this project is to build a model to predict the sentiment of comments. 
    The sentiment is positive or negative. 
    ''')
    st.write('''
    The dataset has 2 classes: positive and negative. 
    ''')
    st.write('''
    The model is built with Logistic Regression and applying oversampling data:
    - The model has 86% accuracy.
    - The model has 94% precision for the positive class.
    - The model has 85% recall for the positive class.
    - The model has 73% precision for the negative class.
    - The model has 88% recall for the negative class. 
    ''')
elif choice == 'Build Model':
    st.subheader('Build Model')
    st.write('#### Data Preprocessing')
    st.write('##### Show data')
    st.table(df.head())
    # plot bar chart for sentiment
    st.write('##### Bar chart for sentiment')
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df['sentiment'].value_counts().index, df['sentiment'].value_counts().values)
    ax.set_xticks(df['sentiment'].value_counts().index)
    ax.set_xticklabels(['Positive', 'Negative'])
    ax.set_ylabel('Number of comments')
    ax.set_title('Bar chart for sentiment')
    st.pyplot(fig)
    
    # plot wordcloud for positive and negative comments
    st.write('##### Wordcloud for positive comments')
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(WordCloud(width=800, height=400, background_color='white').generate(' '.join(df[df['sentiment'] == 1]['comment'])))
    ax.axis('off')
    st.pyplot(fig)
    
    st.write('##### Wordcloud for negative comments')
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(WordCloud(width=800, height=400, background_color='white').generate(' '.join(df[df['sentiment'] == 0]['comment'])))
    ax.axis('off')
    st.pyplot(fig)
    
    st.write('#### Build model and evaluation:')
    st.write('##### Confusion matrix')
    st.table(cm)
    st.write('##### Classification report')
    st.table(classification_report(y_test, y_pred, 
                                    output_dict=True
                                    ))
    st.write('##### Accuracy')
    # show accuracy as percentage with 2 decimal places
    st.write(f'{accuracy_score(y_test, y_pred)*100:.2f}%')
    
elif choice == 'New Prediction':
    st.subheader('New Prediction')
    st.write('''
    Input a comment and the model will predict the sentiment of the comment. 
    ''')
    comment = st.text_input('Input a comment')
    if st.button('Predict'):
        if comment != '':
            comment = process_text(comment, emoji_dict, teen_dict, wrong_lst)
            comment = covert_unicode(comment)
            comment = process_special_word(comment)
            comment = process_postag_thesea(comment)
            comment = remove_stopword(comment, stopwords=stopwords_lst)
            comment = vectorizer.transform([comment])
            sentiment = model.predict(comment)
            if sentiment == 1:
                st.write('The sentiment of the comment is positive')
            else:
                st.write('The sentiment of the comment is negative')
        else:
            st.write('Please input a comment')