#import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

import os

#import seaborn as sns

import streamlit as st

import xgboost as xgb

#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#import torch.optim as optim
import pickle

from sklearn.preprocessing import MinMaxScaler

from collections import defaultdict
from sklearn.linear_model import LinearRegression, Lasso, Ridge

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

#sns.set_style("darkgrid")
#sns_p = sns.color_palette('Paired')

# ==============================================

import joblib
from PIL import Image
import requests
from io import BytesIO

# ==============================================

# PyTorch Model




# ==============================================

xg_reg = xgb.XGBRegressor(objective = "reg:squarederror", max_depth = 10,
                       n_estimators = 30, seed = 420)
xg_reg.load_model('xgb_bmw_reg.json')

enc = joblib.load('encoder.joblib')
enc_pt = joblib.load('encoder_pytorch.joblib')
minmax = joblib.load('minmax.joblib')

rfr = joblib.load('rfr.joblib')

# ========================================================

response = requests.get("https://streamlit-data.s3-us-west-1.amazonaws.com/bmw.jpg")
img = Image.open(BytesIO(response.content))

#response = requests.get("https://streamlit-data.s3-us-west-1.amazonaws.com/x1.jpg")
#x1 = Image.open(BytesIO(response.content)).resize((200, 200))

#response = requests.get("https://streamlit-data.s3-us-west-1.amazonaws.com/x2.png")
#x2 = Image.open(BytesIO(response.content)).resize((200, 200))


#response = requests.get("https://streamlit-data.s3-us-west-1.amazonaws.com/s1.png")
#s1 = Image.open(BytesIO(response.content)).resize((200, 200))

#response = requests.get("https://streamlit-data.s3-us-west-1.amazonaws.com/s2.png")
#s2 = Image.open(BytesIO(response.content)).resize((200, 200))

#response = requests.get("https://streamlit-data.s3-us-west-1.amazonaws.com/s3.jpg")
#s3 = Image.open(BytesIO(response.content)).resize((200, 200))

#response = requests.get("https://streamlit-data.s3-us-west-1.amazonaws.com/s4.png")
#s4 = Image.open(BytesIO(response.content)).resize((200, 200))

#response = requests.get("https://streamlit-data.s3-us-west-1.amazonaws.com/s5.png")
#s5 = Image.open(BytesIO(response.content)).resize((200, 200))

#response = requests.get("https://streamlit-data.s3-us-west-1.amazonaws.com/s6.jpg")
#s6 = Image.open(BytesIO(response.content)).resize((200, 200))

#response = requests.get("https://streamlit-data.s3-us-west-1.amazonaws.com/s7.png")
#s7 = Image.open(BytesIO(response.content)).resize((200, 200))

#response = requests.get("https://streamlit-data.s3-us-west-1.amazonaws.com/s8.png")
#s8 = Image.open(BytesIO(response.content)).resize((200, 200))

#response = requests.get("https://streamlit-data.s3-us-west-1.amazonaws.com/x3.jpg")
#x3 = Image.open(BytesIO(response.content)).resize((200, 200))

#response = requests.get("https://streamlit-data.s3-us-west-1.amazonaws.com/x3.jpg")
#x4 = Image.open(BytesIO(response.content)).resize((200, 200))

#response = requests.get("https://streamlit-data.s3-us-west-1.amazonaws.com/x5.jpeg")
#x5 = Image.open(BytesIO(response.content)).resize((200, 200))

#response = requests.get("https://streamlit-data.s3-us-west-1.amazonaws.com/m2.jpg")
#m2 = Image.open(BytesIO(response.content)).resize((200, 200))

#response = requests.get("https://streamlit-data.s3-us-west-1.amazonaws.com/m3.png")
#m3 = Image.open(BytesIO(response.content)).resize((200, 200))

#response = requests.get("https://streamlit-data.s3-us-west-1.amazonaws.com/m4.png")
#m4 = Image.open(BytesIO(response.content)).resize((200, 200))

#response = requests.get("https://streamlit-data.s3-us-west-1.amazonaws.com/m5.jpg")
#m5 = Image.open(BytesIO(response.content)).resize((200, 200))

response = requests.get("https://streamlit-data.s3-us-west-1.amazonaws.com/m6.png")
m6 = Image.open(BytesIO(response.content)).resize((200, 200))

#response = requests.get("https://streamlit-data.s3-us-west-1.amazonaws.com/i3.jpeg")
#i3 = Image.open(BytesIO(response.content)).resize((200, 200))

#response = requests.get("https://streamlit-data.s3-us-west-1.amazonaws.com/i8.jpg")
#i8 = Image.open(BytesIO(response.content)).resize((200, 200))

#response = requests.get("https://streamlit-data.s3-us-west-1.amazonaws.com/x6.jpg")
#x6 = Image.open(BytesIO(response.content)).resize((200, 200))

#response = requests.get("https://streamlit-data.s3-us-west-1.amazonaws.com/x7.jpg")
#x7 = Image.open(BytesIO(response.content)).resize((200, 200))

#response = requests.get("https://streamlit-data.s3-us-west-1.amazonaws.com/z3.png")
#z3 = Image.open(BytesIO(response.content)).resize((200, 200))

#response = requests.get("https://streamlit-data.s3-us-west-1.amazonaws.com/z4.jpg")
#z4 = Image.open(BytesIO(response.content)).resize((200, 200))




#image = Image.open('/home/nareg/Downloads/cmb-1.jpg')
st.image(img, use_column_width = True)
# ===============================================



st.markdown('## **BMW Sale Price Prediction**')

st.markdown('This app provides fair sale price for second hand BMW cars in USA. Various algorithms have been trained for this purpose.')
st.markdown('Start by choosing an algorithm from the left panel.')
st.markdown('-------------------------------------------------------------------------------')



algo = st.sidebar.selectbox('Select Algorithm', pd.DataFrame({'algos': ['XGBoost', 'Neural Network', 'RandomForestRegressor']}))


#year = st.text_area("enter your age")

model_ = st.selectbox('Model', [' 1 Series', ' 2 Series', ' 3 Series', ' 4 Series', ' 5 Series',
       ' 6 Series', ' 7 Series', ' 8 Series', ' M2', ' M3', ' M4', ' M5',
       ' M6', ' X1', ' X2', ' X3', ' X4', ' X5', ' X6', ' X7', ' Z3', ' Z4', ' i3', ' i8'])


if model_ == ' X1':

    response = requests.get("https://streamlit-data.s3-us-west-1.amazonaws.com/x1.jpg")
    x1 = Image.open(BytesIO(response.content)).resize((200, 200))
    st.image(x1)

elif model_ == ' X2':

    response = requests.get("https://streamlit-data.s3-us-west-1.amazonaws.com/x2.png")
    x2 = Image.open(BytesIO(response.content)).resize((200, 200))
    st.image(x2)

elif model_ == ' X6':

    response = requests.get("https://streamlit-data.s3-us-west-1.amazonaws.com/x6.jpg")
    x6 = Image.open(BytesIO(response.content)).resize((200, 200))
    st.image(x6)

elif model_ == ' X7':

    response = requests.get("https://streamlit-data.s3-us-west-1.amazonaws.com/x7.jpg")
    x7 = Image.open(BytesIO(response.content)).resize((200, 200))
    st.image(x7)


elif model_ == ' 1 Series':

    response = requests.get("https://streamlit-data.s3-us-west-1.amazonaws.com/s1.png")
    s1 = Image.open(BytesIO(response.content)).resize((200, 200))
    st.image(s1)

elif model_ == ' 2 Series':

    response = requests.get("https://streamlit-data.s3-us-west-1.amazonaws.com/s2.png")
    s2 = Image.open(BytesIO(response.content)).resize((200, 200))
    st.image(s2)

elif model_ == ' 3 Series':

    response = requests.get("https://streamlit-data.s3-us-west-1.amazonaws.com/s3.jpg")
    s3 = Image.open(BytesIO(response.content)).resize((200, 200))
    st.image(s3)

elif model_ == ' 4 Series':

    response = requests.get("https://streamlit-data.s3-us-west-1.amazonaws.com/s4.png")
    s4 = Image.open(BytesIO(response.content)).resize((200, 200))
    st.image(s4)

elif model_ == ' 5 Series':

    response = requests.get("https://streamlit-data.s3-us-west-1.amazonaws.com/s5.png")
    s5 = Image.open(BytesIO(response.content)).resize((200, 200))
    st.image(s5)

elif model_ == ' 6 Series':

    response = requests.get("https://streamlit-data.s3-us-west-1.amazonaws.com/s6.jpg")
    s6 = Image.open(BytesIO(response.content)).resize((200, 200))
    st.image(s6)

elif model_ == ' 7 Series':

    response = requests.get("https://streamlit-data.s3-us-west-1.amazonaws.com/s7.png")
    s7 = Image.open(BytesIO(response.content)).resize((200, 200))
    st.image(s7)

elif model_ == ' 8 Series':

    response = requests.get("https://streamlit-data.s3-us-west-1.amazonaws.com/s8.png")
    s8 = Image.open(BytesIO(response.content)).resize((200, 200))
    st.image(s8)

elif model_ == ' X3':

    response = requests.get("https://streamlit-data.s3-us-west-1.amazonaws.com/x3.jpg")
    x3 = Image.open(BytesIO(response.content)).resize((200, 200))
    st.image(x3)

elif model_ == ' X4':

    response = requests.get("https://streamlit-data.s3-us-west-1.amazonaws.com/x3.jpg")
    x4 = Image.open(BytesIO(response.content)).resize((200, 200))
    st.image(x4)

elif model_ == ' X5':

    response = requests.get("https://streamlit-data.s3-us-west-1.amazonaws.com/x5.jpeg")
    x5 = Image.open(BytesIO(response.content)).resize((200, 200))
    st.image(x5)

elif model_ == ' M2':

    response = requests.get("https://streamlit-data.s3-us-west-1.amazonaws.com/m2.jpg")
    m2 = Image.open(BytesIO(response.content)).resize((200, 200))
    st.image(m2)

elif model_ == ' M3':

    response = requests.get("https://streamlit-data.s3-us-west-1.amazonaws.com/m3.png")
    m3 = Image.open(BytesIO(response.content)).resize((200, 200))
    st.image(m3)

elif model_ == ' M4':

    response = requests.get("https://streamlit-data.s3-us-west-1.amazonaws.com/m4.png")
    m4 = Image.open(BytesIO(response.content)).resize((200, 200))
    st.image(m4)

elif model_ == ' M5':

    response = requests.get("https://streamlit-data.s3-us-west-1.amazonaws.com/m5.jpg")
    m5 = Image.open(BytesIO(response.content)).resize((200, 200))
    st.image(m5)

elif model_ == ' i3':

    response = requests.get("https://streamlit-data.s3-us-west-1.amazonaws.com/i3.jpeg")
    i3 = Image.open(BytesIO(response.content)).resize((200, 200))
    st.image(i3)

elif model_ == ' i8':

    response = requests.get("https://streamlit-data.s3-us-west-1.amazonaws.com/i8.jpg")
    i8 = Image.open(BytesIO(response.content)).resize((200, 200))
    st.image(i8)

elif model_ == ' Z3':

    response = requests.get("https://streamlit-data.s3-us-west-1.amazonaws.com/z3.png")
    z3 = Image.open(BytesIO(response.content)).resize((200, 200))
    st.image(z3)


elif model_ == ' Z4':

    response = requests.get("https://streamlit-data.s3-us-west-1.amazonaws.com/z4.jpg")
    z4 = Image.open(BytesIO(response.content)).resize((200, 200))
    st.image(z4)



#===================================================================================================================


#transmission_ = st.radio('Transmission', ['Automatic', 'Manual', 'Semi-Auto'])

#fueltype_ = st.radio('Fuel Type', ['Diesel', 'Electric', 'Hybrid', 'Other', 'Petrol'])

year_ = st.slider('Select Year', min_value = 1996, max_value = 2020)

mileage_ = st.number_input('Enter Mileage (Press Enter)')

mpg_ = st.number_input('Enter MPG (Press Enter)')

esize_ = st.slider('Select Engine Size', min_value = 0.0, max_value = 6.8, step = 0.1)

tax_ = st.number_input('Enter Tax Per Year ($)')

#df = pd.DataFrame({'model': [model_], 'year': [year_], 'transmission': [transmission_], 'mileage': [mileage_], 'fuelType': [fueltype_], 'tax': [tax_], 'mpg': [mpg_], 'engineSize': [esize_]},#
                    #    index = ['Your BMW'])

col1, col2 = st.beta_columns(2)

with col1:

    transmission_ = st.radio('Transmission', ['Automatic', 'Manual', 'Semi-Auto'])

with col2:

    fueltype_ = st.radio('Fuel Type', ['Diesel', 'Electric', 'Hybrid', 'Other', 'Petrol'])


df = pd.DataFrame({'model': [model_], 'year': [year_], 'transmission': [transmission_], 'mileage': [mileage_], 'fuelType': [fueltype_], 'tax': [tax_], 'mpg': [mpg_], 'engineSize': [esize_]},
                        index = ['Your BMW'])


if algo == 'XGBoost':

    df1 = pd.DataFrame({'year': [year_], 'mileage': [mileage_], 'tax': [tax_],
                    'mpg': [mpg_], 'engineSize': [esize_]})
    df2 = pd.DataFrame(enc.transform(pd.DataFrame({'model': [model_], 'fuelType': [fueltype_],
                                               'transmission': [transmission_]})).toarray(), index = df1.index)
    X_input = pd.concat([df2, df1], axis=1)

    xgb_pred = xg_reg.predict(X_input)[0]

    "-------"

    "## **Predicted Price**: ", xg_reg.predict(X_input)[0], "$"
    st.markdown("The XGB model has been trained with more than 7000 cars and has an $R^2 \simeq 0.96$ for the testing set.")


if algo == 'RandomForestRegressor':

    df1 = pd.DataFrame({'year': [year_], 'mileage': [mileage_], 'tax': [tax_],
                    'mpg': [mpg_], 'engineSize': [esize_]})
    df2 = pd.DataFrame(enc.transform(pd.DataFrame({'model': [model_], 'fuelType': [fueltype_],
                                               'transmission': [transmission_]})).toarray(), index = df1.index)
    X_input = pd.concat([df2, df1], axis=1)

    rfr_pred = rfr.predict(X_input)[0]

    "-------"

    "## **Predicted Price**: ", rfr.predict(X_input)[0], "$"
    st.markdown("The RandomForestRegressor model has been trained with more than 7000 cars and has an $R^2 \simeq 0.963$ for the testing set.")


if algo == 'Neural Network':

    "-------"
    "See Github page"

#    df1 = pd.DataFrame({'year': [year_], 'mileage': [mileage_], 'tax': [tax_],
                       # 'mpg': [mpg_], 'engineSize': [esize_]})
#    df2 = pd.DataFrame(enc_pt.transform(pd.DataFrame({'model': [model_], 'fuelType': [fueltype_],
                                        #           'transmission': [transmission_]})).toarray(), index = df1.index)
#    X_input = pd.concat([df2, df1], axis=1)

#    X_input = X_input.astype('float32').values

    #X_input = torch.from_numpy(X_input.values)
    #minmax = MinMaxScaler()
#    X_input = minmax.transform(X_input)

#    pt_model = torch.load('bmw_pytorch.pt')
#    pt_pred = pt_model(torch.from_numpy(X_input)).flatten().detach().numpy()[0]

#    "## **Predicted Price**: ", pt_pred
#    st.markdown("The Neural Netrwok model has an $R^2 \simeq 0.93$.")



st.markdown('---------------------------------------------------------------------------')



st.markdown('---------------------------------------------------------------------------')


st.markdown('-------------------------------------------------------------------------------')

st.markdown("""
                Made by [Nareg Mirzatuny](https://github.com/NaregM)

Source code: [GitHub](
                https://github.com/NaregM/heroku_bmw_price_prediction)

""")
st.markdown('-------------------------------------------------------------------------------')
