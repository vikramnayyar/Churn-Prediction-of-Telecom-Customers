#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 16:09:06 2021

@author: Vikram Nayyar
"""
import pandas as pd
import streamlit as st
import pickle as pkl

df = pd.read_csv("../data/clean_data.csv")

st.title('Churn Prediction for Telecom Customers')

st.write("This app is based on 15 inputs that predict wheather a customer will deposit or not? Using this app, a bank can identify specific customer segments; that will make deposits.")
st.write("Please use the following form to get started!")
st.markdown('<p class="big-font">(NOTE: For convinience, usual values are pre-selected in the form.)</p>', unsafe_allow_html=True)


# selecting tenure
st.subheader("Select Customer's Tenure")
tenure = st.slider("", min_value = 0, max_value = 72, 
                         step = 1, value = 14)    # Slider does not tolerate dtype value mismatch df.age.max() was thus not used.
#st.write("Selected Age:", selected_age)

def extract_dict(df, col):
    feature = df[col].astype('category')
    dict_val = dict(enumerate(feature.cat.categories))
    dict_inv = {a:b for b,a in dict_val.items()}
    
    return dict_inv


def encode(df, col, selected_item):  # Encode the job entered by user
        g_dict = extract_dict(df, col)
        dict_job = g_dict
        return dict_job.get(selected_item, 'No info available')


#%% selecting MonthlyCharges
st.subheader("Select Customer's Monthly Charges")

charges_list = df.MonthlyCharges.unique()
charges_list = sorted(charges_list, key = lambda x:float(x))

# selected = st.selectbox("", charges_list)

selected = st.select_slider("", options = charges_list)

monthly_charges = encode(df, "MonthlyCharges", selected)  


#%% selecting TotalCharges

st.subheader("Select Customer's Total Charges")

charges_list = df.TotalCharges.unique()
charges_list = sorted(charges_list, key = lambda x:float(x))

selected = st.select_slider("", options = charges_list)

total_charges = encode(df, "TotalCharges", selected)  


#%% selecting online security
st.subheader("Select Customer's Online Security Status")
selected = st.radio("", df["OnlineSecurity"].unique(), index = 2)    # index = 3 removed
#st.write("Selected Job:", selected)

# Using function for encoding
online_security = encode(df, "OnlineSecurity", selected)  


# selecting gender
st.subheader("Select Customer's Gender")
selected = st.radio("", df["gender"].unique(), index = 1)    # index = 3 removed
#st.write("Selected Job:", selected)

# Using function for encoding
gender = encode(df, "gender", selected)  



# selecting SeniorCitizen
st.subheader("Select Customer's Senior Citizen Status")
selected = st.radio("", df["SeniorCitizen"].unique(), index = 1)    # index = 3 removed
#st.write("Selected Job:", selected)

# Using function for encoding
senior_citizen = encode(df, "SeniorCitizen", selected)  



# selecting partner
st.subheader("Select Customer's Partner Status")
selected = st.radio("", df["Partner"].unique(), index = 1)    # index = 3 removed
#st.write("Selected Job:", selected)

# Using function for encoding
partner = encode(df, "Partner", selected)  


# selecting dependents
st.subheader("Select Customer's Dependent Status")
selected = st.radio("", df["Dependents"].unique(), index = 1)    # index = 3 removed
#st.write("Selected Job:", selected)

# Using function for encoding
dependents = encode(df, "Dependents", selected)  


# selecting MutlipleLines
st.subheader("Select Customer's Multiple Lines Status")
selected = st.radio("", df["MultipleLines"].unique(), index = 0)    # index = 3 removed

# Using function for encoding
multiple_lines = encode(df, "MultipleLines", selected)  


# selecting InternetService
st.subheader("Select Customer's Internet Service Status")
selected = st.radio("", df["InternetService"].unique(), index = 1)    # index = 3 removed
#st.write("Selected Job:", selected)

# Using function for encoding
internet_service = encode(df, "InternetService", selected)  


# selecting OnlineBackup
st.subheader("Select Customer's Online Backup Status")
selected = st.radio("", df["OnlineBackup"].unique(), index = 1)    # index = 3 removed

# Using function for encoding
online_backup = encode(df, "OnlineBackup", selected)  



# selecting DeviceProtection
st.subheader("Select Customer's Device Protection Status")
selected = st.radio("", ['No', 'Yes', 'No internet service'], index = 1)    # index = 3 removed

# Using function for encoding
device_protection = encode(df, "DeviceProtection", selected)  



# selecting TechSupport
st.subheader("Select Customer's Tech Support Status")
selected = st.radio("", ['Yes', 'No', 'No internet service'], index = 2)    # index = 3 removed

# Using function for encoding
tech_support = encode(df, "TechSupport", selected)  



# selecting StreamingTV
st.subheader("Select Customer's Streaming TV Status")
selected = st.radio("", ['Yes', 'No', 'No internet service'], index = 2, key =1)    # index = 3 removed

# Using function for encoding
streaming_tv = encode(df, "StreamingTV", selected)  




# selecting StreamingMovies
st.subheader("Select Customer's Streaming Movies Status")
selected = st.radio("", ['Yes', 'No', 'No internet service'], index = 2, key = 9)    # index = 3 removed

# Using function for encoding
streaming_mov = encode(df, "StreamingMovies", selected)  




# selecting Contract
st.subheader("Select Customer's Contract Status")
selected = st.radio("", df["Contract"].unique(), index = 2)    # index = 3 removed

# Using function for encoding
contract = encode(df, "Contract", selected)  





# selecting PaperlessBilling
st.subheader("Select Customer's Paperless Billing Status")
selected = st.radio("", df["PaperlessBilling"].unique(), index = 1, key=0)    # index = 3 removed

# Using function for encoding
paperless_billing = encode(df, "PaperlessBilling", selected)  



# selecting PaymentMethod
st.subheader("Select Customer's Payment Method Status")
selected = st.radio("", df["PaymentMethod"].unique(), index = 1)    # index = 3 removed

# Using function for encoding
payment_method = encode(df, "PaymentMethod", selected)  




## Adding Features

feature_1 = monthly_charges**(0.2) + senior_citizen**(0.35) - online_security**(0.1)


pickle_in = open("../model/model.pkl","rb")
classifier = pkl.load(pickle_in)


prediction = classifier.predict([[gender, partner, dependents, tenure, 
                                  multiple_lines, internet_service, online_security, online_backup, 
                                  device_protection, tech_support, streaming_tv, streaming_mov, 
                                  contract, paperless_billing, payment_method, monthly_charges, 
                                  total_charges, feature_1]])   # index is causing problem



# Adding Predict Button
predict_button = st.button('Predict')
# st.write(predict_button)

if predict_button:
    if(prediction == 1):
        st.success('This customer segment will Churn')
    else:
        st.success('This customer segment will NOT Churn')    

st.write('\n')
about = st.expander('More about app')
about.write("https://github.com/vikramnayyar/Customer-Identification-for-Bank-Marketing/blob/main/README.md")
