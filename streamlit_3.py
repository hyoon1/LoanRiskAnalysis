import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

st.set_page_config(page_title="Loan Risk Analysis", layout="wide")

# Load pre-trained models and scaler
with open('models/model_LR.pkl', 'rb') as f:
    model_LR = pickle.load(f)
with open('models/model_RFC.pkl', 'rb') as f:
    model_RFC = pickle.load(f)
with open('models/model_SVC.pkl', 'rb') as f:
    model_SVC = pickle.load(f)
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

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
features = ['person_home_ownership', 'loan_grade', 'loan_percent_income', 'cb_person_default_on_file']
target = 'loan_status'

st.title('Loan Risk Analysis Dashboard')
st.subheader('Data Overview')

if st.checkbox('Show DataFrame', value=True):
    st.dataframe(df)  # Display the DataFrame as is

# Collapsible Data Description Section
with st.expander("About Dataset"):
    st.markdown("""
    Detailed data description of Credit Risk dataset:
    
    | Feature Name                 | Description                               |
    |------------------------------|-------------------------------------------|
    | `person_age`                 | Age                                       |
    | `person_income`              | Annual Income                             |
    | `person_home_ownership`      | Home ownership                            |
    | `person_emp_length`          | Employment length (in years)              |
    | `loan_intent`                | Loan intent                               |
    | `loan_grade`                 | Loan grade                                |
    | `loan_amnt`                  | Loan amount                               |
    | `loan_int_rate`              | Interest rate                             |
    | `loan_status`                | Loan status (0 is Approved, 1 is Declined)|
    | `loan_percent_income`        | Percent income                            |
    | `cb_person_default_on_file`  | Historical default                        |
    | `cb_person_cred_hist_length` | Credit history length                     |
    """)

selected_index = st.number_input('Enter row index to display and predict:', min_value=0, max_value=len(df)-1, value=0, step=1)
selected_row = df.iloc[selected_index]
st.write('Selected Data:')
st.dataframe(selected_row.to_frame().T, height=150)

if st.button('Predict'):
    input_data = selected_row[features].to_numpy().reshape(1, -1)
    input_scaled = scaler.transform(input_data)
    
    # Predict using all models
    pred_lr = model_LR.predict(input_scaled)[0]
    pred_rfc = model_RFC.predict(input_scaled)[0]
    pred_svc = model_SVC.predict(input_scaled)[0]
    
    # Majority voting
    predictions = np.array([pred_lr, pred_rfc, pred_svc])
    prediction, count = np.unique(predictions, return_counts=True)
    majority_vote = prediction[np.argmax(count)]
    
    prediction_text = 'Approved' if majority_vote == 0 else 'Declined'
    true_label_text = 'Approved' if selected_row[target] == 0 else 'Declined'
    
    st.write(f'Majority Predicted Class: {prediction_text}')
    st.write(f'True Label: {true_label_text}')
