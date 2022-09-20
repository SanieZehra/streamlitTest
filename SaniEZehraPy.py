import streamlit as st
import numpy as np
import pandas as pd
#import nltk
import string
import re
from collections import Counter
pd.options.mode.chained_assignment = None
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('wordnet')
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
import plotly
import plotly.figure_factory as ff
import matplotlib.pyplot as plt


def main():
    st.title('MESSAGES HISTORY')

    #Loading data
    df=pd.read_csv('C:/Users/Dell/Downloads/Asg/SMS_data.csv', encoding= 'unicode_escape')

    #preparing the data
    df["Message_body"] = df["Message_body"].astype(str)
    df["Message_body"]=df["Message_body"].str.lower()

    #removing puncs
    removePunc = string.punctuation
    def remvPunc(text):
        return text.translate(str.maketrans('', '', removePunc))
    df["Message_body"] = df["Message_body"].apply(lambda text: remvPunc(text))

    #removing stopwords
    stopWords = set(stopwords.words('english'))
    def remove_stopwords(text):
        return " ".join([word for word in str(text).split() if word not in stopWords])
    df["Message_body"] = df["Message_body"].apply(lambda text: remove_stopwords(text))

    #Lemmatizing
    lemmatizer = WordNetLemmatizer()
    def lemmatize_words(text):
        return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

    df["Message_body"] = df["Message_body"].apply(lambda text: lemmatize_words(text))

    #A dropdown lets you choose between Spam and Non-spam data and adjusts your visualizations accordingly
    selection = st.selectbox('Sort Messages By',['Spam','Non-Spam'])
    print(selection)

    if selection == 'Spam':
        disp=df[df['Label']=='Spam']
        st.table(disp)
    else:
        disp=df[df['Label']=='Non-Spam']
        st.table(disp)
    
    #Bar Chart1 - A visualization that shows the most common keywords for Non-Spam Messages
    a3=[]
    a4=[]
    nonSpam=df[df['Label']=='Non-Spam']
    count = Counter()
    for text in nonSpam["Message_body"].values:
        for word in text.split():
            count[word] += 1    
    count.most_common(5)
    for letter, counting in count.most_common(5):
        a3.append(letter)
        a4.append(counting)

    st.subheader('The Most Common Keywords in Non-Spam')
    st.write(a3)
    st.bar_chart(data=a4,x=a3)

    #Bar Chart2 - A visualization that shows the most common keywords for Spam messages
    a1=[]
    a2=[]
    Spam=df[df['Label']=='Spam']
    for text in Spam["Message_body"].values:
        for word in text.split():
            count[word] += 1
    count.most_common(5)
    for letter, counting in count.most_common(5):
        a1.append(letter)
        a2.append(counting)
    
    st.subheader('The Most Common Keywords in Spam')
    st.write(a1)
    st.bar_chart(data=a2,x=a1)


    #A visualization that shows the Number of Messages Received over Dates in (Line Chart).
    dateDf= df.groupby('Date_Received')['S. No.'].count()
    #Line Chart
    st.subheader('Number of Messages Received over Dates')
    st.write(dateDf)
    st.line_chart(dateDf)



if __name__== '__main__':
    main()
