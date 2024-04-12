import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, RocCurveDisplay
import pickle

# Load your pre-trained models
with open('models/model_LR.pkl', 'rb') as f:
    model_LR = pickle.load(f)
with open('models/model_RFC.pkl', 'rb') as f:
    model_RFC = pickle.load(f)
with open('models/model_SVC.pkl', 'rb') as f:
    model_SVC = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("./data/credit_risk_dataset.csv")
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    label_enc = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = label_enc.fit_transform(df[col])
    return df

df = load_data()

# Selecting features as per the earlier preprocessing in your analysis
features = ['person_home_ownership', 'loan_grade', 'loan_percent_income', 'cb_person_default_on_file']
target = 'loan_status'

# Main panel
st.title('Loan Risk Analysis Dashboard')

# Input Features
st.header('Input Features')
input_data = {}
input_data['person_home_ownership'] = st.radio('Home Ownership', df['person_home_ownership'].unique(), index=int(df['person_home_ownership'].mode()))
input_data['loan_grade'] = st.radio('Loan Grade', df['loan_grade'].unique(), index=int(df['loan_grade'].mode()))
input_data['loan_percent_income'] = st.slider('Loan Percent Income', float(df['loan_percent_income'].min()), float(df['loan_percent_income'].max()), float(df['loan_percent_income'].median()))
input_data['cb_person_default_on_file'] = st.radio('Credit Bureau Default on File', df['cb_person_default_on_file'].unique(), index=int(df['cb_person_default_on_file'].mode()))

option = st.selectbox(
    'Choose model to evaluate:',
    ('Logistic Regression', 'Random Forest Classifier', 'Support Vector Machine')
)

# Button to classify
if st.button('Classify'):
    # Prepare input data for prediction
    input_df = pd.DataFrame([input_data])
    # We need to apply the same scaling as the model training
    input_scaled = scaler.transform(input_df)

    # Make prediction
    if option == 'Logistic Regression':
        prediction = model_LR.predict(input_scaled)
    elif option == 'Random Forest Classifier':
        prediction = model_RFC.predict(input_scaled)
    elif option == 'Support Vector Machine':
        prediction = model_SVC.predict(input_scaled)
    
    # Display prediction
    st.subheader('Prediction')
    st.write('The predicted class is: {}'.format(prediction[0]))

# Display the DataFrame
if st.checkbox('Show raw data'):
    st.subheader('Raw Data')
    st.write(df)

# Streamlit commands to run the app
# streamlit run your_script_name.py