import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb
import numpy as np
word_index=imdb.get_word_index()
reverse_word_index=dict([(value,key) for (key,value) in word_index.items()])
model=load_model('rnn_model.h5')
def decode_review(review):
    return ' '.join([reverse_word_index.get(i-3,'?') for i in review])
def preprocess_review(review):
    review=review.split()
    review=[word_index.get(i,2)+3 for i in review]
    padded_review=sequence.pad_sequences([review],maxlen=500)
    return padded_review

def predict_review(review):
    preprocessed_review=preprocess_review(review)
    prediction=model.predict(preprocessed_review)
    sentinement='Positive' if prediction[0][0]>0.5 else 'Negative'
    return sentinement,prediction[0][0]

st.title('Sentiment Analysis')
review=st.text_area("Enter your review")
if st.button('Classify'):
    sentiment,score=predict_review(review)
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Confidence: {score}')
else:
    st.write("Please enter the movie review")