import numpy as np
import pickle
import pandas as pd
#from flasgger import Swagger
import streamlit as st 

from PIL import Image

#app=Flask(__name__)
#Swagger(app)
filename = 'restaurant-review.pkl'
classifier = pickle.load(open(filename, 'rb'))
tfidf = pickle.load(open('tfidf-transform.pkl','rb'))


#@app.route('/')
def welcome():
    return "Welcome All"

#@app.route('/predict',methods=["Get"])
def predict_note(message):
    
    data = [message]
    vect = tfidf.transform(data).toarray()
    prediction=classifier.predict(vect)
    print(prediction)
    return prediction



def main():
    st.title("Restaurant Review Classifier")
    st.subheader('TFIFD Vectorizer')     
    st.write('This project is based on Naive Bayes Classifier')
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Restaurant Review Classifier ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    message = st.text_area("Enter Text","Type Here ..")
    
    result=""
    if st.button("Predict"):
        result=predict_note(message)
    st.success('The output is {}'.format(result))

if __name__=='__main__':
    main()
    