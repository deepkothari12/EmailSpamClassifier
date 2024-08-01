import streamlit as st
import pickle
import nltk
import string
from nltk.stem.porter import PorterStemmer
#step
#1-preprocessing
#2-vectorization
#3-classification/prdict
pr = PorterStemmer()
englis_list = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
tfidf = pickle.load(open('vectarization.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))


def transform_text(text):
    text_lower = text.lower()
    text_with_token = nltk.word_tokenize(text_lower)

    y = []
    for i in text_with_token:
        if i.isalnum():
            y.append(i)

    text = y[:] #cloning
    y.clear()
    for i in text:
        if i not in englis_list and i not in string.punctuation :
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(pr.stem(i))
         

    return " ".join(y) #list -> str


st.title("Email Spamm Classifiers")

input_sms = st.text_input("Enter the text or Messages")

if st.button('Predict'):
    transform_text_sms = transform_text(input_sms)

    vector_input = tfidf.transform([transform_text_sms])

    result = model.predict(vector_input)[0]

    if result == 1:
        st.header("Spammmm")

    else :
        st.header("Not Spam")
