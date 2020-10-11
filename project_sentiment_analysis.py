# -*- coding: utf-8 -*-
"""Project Sentiment Analysis.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1UZu3BrepB01zCAcE9gvrrg3MIVmyNzEu
"""

!pip install nltk

import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

df = pd.read_table('/content/drive/My Drive/smartknower/Machine learning/Sentiment Analysis project/Restaurant_Reviews.tsv')
nltk.download('stopwords')
corpus = []
for i in range(0, 1000):
    comment = re.sub(pattern='[^a-zA-Z]',repl= ' ',string = df['Review'][i])
    comment = comment.lower()
    comment_words = comment.split()
    ps = PorterStemmer()
    comment = [ps.stem(word) for word in comment_words if not word in set(stopwords.words('english'))]
    comment = [ps.stem(word) for word in comment_words]
    comment = ' '.join(comment)
    corpus.append(comment)
tfidf = TfidfVectorizer(max_features=1500)
x = tfidf.fit_transform(corpus).toarray()
y = df.iloc[:, 1].values

pickle.dump(tfidf, open('tfidf-transform.pkl', 'wb'))

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state= 0)

text_model = MultinomialNB(alpha=0.2)
text_model.fit(x_train,y_train)


import pickle
pickle.dump(tfidf, open('tfidf-transform.pkl', 'wb'))
filename = 'restaurant-review.pkl'
pickle.dump(text_model, open(filename, 'wb'))

# Commented out IPython magic to ensure Python compatibility.
# %%writefile app.py
# import numpy as np
# import pickle
# import pandas as pd
# #from flasgger import Swagger
# import streamlit as st 
# 
# from PIL import Image
# 
# #app=Flask(__name__)
# #Swagger(app)
# filename = 'restaurant-review.pkl'
# classifier = pickle.load(open(filename, 'rb'))
# tfidf = pickle.load(open('tfidf-transform.pkl','rb'))
# 
# 
# #@app.route('/')
# def welcome():
#     return "Welcome All"
# 
# #@app.route('/predict',methods=["Get"])
# def predict_note(message):
#     
#     data = [message]
#     vect = tfidf.transform(data).toarray()
#     prediction=classifier.predict(vect)
#     print(prediction)
#     return prediction
# 
# 
# 
# def main():
#     st.title("Restaurant Review Classifier")
#     st.subheader('TFIFD Vectorizer')     
#     st.write('This project is based on Naive Bayes Classifier')
#     html_temp = """
#     <div style="background-color:tomato;padding:10px">
#     <h2 style="color:white;text-align:center;">Restaurant Review Classifier ML App </h2>
#     </div>
#     """
#     st.markdown(html_temp,unsafe_allow_html=True)
#     message = st.text_area("Enter Text","Type Here ..")
#     
#     result=""
#     if st.button("Predict"):
#         result=predict_note(message)
#     st.success('The output is {}'.format(result))
# 
# if __name__=='__main__':
#     main()

!pip install streamlit

!streamlit run app.py

!pip install pyngrok

from pyngrok import ngrok
url = ngrok.connect(port='8501')
url

