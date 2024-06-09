#!/usr/bin/env python
# coding: utf-8

# In[26]:


import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import xgboost as xgb

st.title("Autism Child Data Analysis and Model Testing")

# Function to load data
def load_data():
    df = pd.read_csv('combined.csv')
    return df

# Your custom preprocessing function
def custom_preprocessing(df):
    x = df.drop("ASD_traits", axis=1)
    y = df["ASD_traits"]

    # One Hot Encoding for the categorical variables
    x = pd.get_dummies(x, columns=["Ethnicity", "Who_completed_the_test", "Age_Years", "Qchat_10_Score"], drop_first=True)

    # Label Encoding to convert categorical values into numerical values
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y)
    x['Sex'] = le.fit_transform(x['Sex'])
    x['Jaundice'] = le.fit_transform(x['Jaundice'])
    x['Family_mem_with_ASD'] = le.fit_transform(x['Family_mem_with_ASD'])

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    x_scaled = sc.fit_transform(x)
    
    return x_scaled, y

# Function to train the model
def train_model(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=69)
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    return model, accuracy, precision, recall, f1

# Function to get user input and preprocess it
def user_input():
    inputs = {}
    inputs['A1'] = st.slider("A1_Score", 0, 1, 0)
    inputs['A2'] = st.slider("A2_Score", 0, 1, 0)
    inputs['A3'] = st.slider("A3_Score", 0, 1, 0)
    inputs['A4'] = st.slider("A4_Score", 0, 1, 0)
    inputs['A5'] = st.slider("A5_Score", 0, 1, 0)
    inputs['A6'] = st.slider("A6_Score", 0, 1, 0)
    inputs['A7'] = st.slider("A7_Score", 0, 1, 0)
    inputs['A8'] = st.slider("A8_Score", 0, 1, 0)
    inputs['A9'] = st.slider("A9_Score", 0, 1, 0)
    inputs['A10'] = st.slider("A10_Score", 0, 1, 0)
    inputs['Age_Years'] = st.number_input("Age_Years", min_value=0, max_value=100, step=1)
    inputs['Sex'] = st.selectbox("Sex", ['M', 'F'])
    inputs['Ethnicity'] = st.selectbox("Ethnicity", ['Group1', 'Group2'])  # Replace with actual options
    inputs['Jaundice'] = st.selectbox("Jaundice", ['Yes', 'No'])
    inputs['Family_mem_with_ASD'] = st.selectbox("Family_mem_with_ASD", ['Yes', 'No'])
    inputs['Qchat_10_Score'] = st.selectbox("Qchat_10_Score", [0, 1])  # Update with actual options
    inputs['Who_completed_the_test'] = st.selectbox("Who_completed_the_test", ['Parent', 'Self'])  # Replace with actual options

    inputs_df = pd.DataFrame([inputs])

    # Convert categorical inputs to numeric values
    le = LabelEncoder()
    inputs_df['Sex'] = le.fit_transform(inputs_df['Sex'])
    inputs_df['Jaundice'] = le.fit_transform(inputs_df['Jaundice'])
    inputs_df['Family_mem_with_ASD'] = le.fit_transform(inputs_df['Family_mem_with_ASD'])
    
    inputs_df = pd.get_dummies(inputs_df, columns=["Ethnicity", "Who_completed_the_test", "Age_Years", "Qchat_10_Score"], drop_first=True)
    
    # Standardize the input data
    sc = StandardScaler()
    inputs_scaled = sc.fit_transform(inputs_df)
    
    return inputs_scaled

# Load and preprocess data
df = load_data()
x, y = custom_preprocessing(df)

# Train the model
model, accuracy, precision, recall, f1 = train_model(x, y)

# Display model performance
st.write(f"Model Accuracy: {accuracy}")
st.write(f"Model Precision: {precision}")
st.write(f"Model Recall: {recall}")
st.write(f"Model F1 Score: {f1}")

# Get user input and make prediction
st.subheader("Predict ASD")
input_data = user_input()
if st.button("Predict"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.write("The person is likely to have ASD.")
    else:
        st.write("The person is unlikely to have ASD.")
import os

# Get the current working directory
current_directory = os.getcwd()
print("Current Directory:", current_directory)

# List all files and directories in the current directory
files_and_directories = os.listdir(current_directory)
print("Files and Directories in '", current_directory, "':")
print(files_and_directories)


# In[ ]:




