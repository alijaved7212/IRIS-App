# Important Libraries
import matplotlib.pyplot as plt
import numpy
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Simple IRIS Flower Prediction App

This web app predicts the **IRIS flower** type!
""")

st.sidebar.header('User input Parameter')

def user_input_features():

   sepal_length = st.sidebar.slider('sepal length', 4.3, 7.9, 5.4)
   sepal_width = st.sidebar.slider('sepal width', 2.0, 4.4, 3.4)
   petal_length = st.sidebar.slider('petal length', 1.0, 6.9, 1.3)
   petal_width = st.sidebar.slider('petal widht', 0.1, 2.5, 0.2)
   data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width }
   features = pd.DataFrame(data, index=[0])
   return features

df = user_input_features()

st.subheader('User input Parameter')
st.write(df)

iris = datasets.load_iris()
X = iris.data
y = iris.target
clf = RandomForestClassifier()
clf.fit(X ,y)
prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Class lables and their corresponding index number')
st.write(iris.target_names)

st.subheader('Prediction')
st.write(iris.target_names[prediction])
# st.write(prediction)

st.subheader('Prediction probablity')
st.write(prediction_proba)
ns.scatterplot





    
    




