import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('diabetes.csv')

df = load_data()

# Sidebar for user input
st.sidebar.header('User Input Parameters')

def user_input_features():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 20, 1)
    glucose = st.sidebar.slider('Glucose', 0, 200, 120)
    blood_pressure = st.sidebar.slider('Blood Pressure', 0, 122, 70)
    skin_thickness = st.sidebar.slider('Skin Thickness', 0, 99, 20)
    insulin = st.sidebar.slider('Insulin', 0, 846, 79)
    bmi = st.sidebar.slider('BMI', 0.0, 67.1, 20.0)
    diabetes_pedigree_function = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.5, 0.5)
    age = st.sidebar.slider('Age', 21, 81, 33)
    data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': diabetes_pedigree_function,
        'Age': age
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Main panel
st.title('Diabetes Prediction App')

# Display user input
st.subheader('User Input parameters')
st.write(input_df)

# Model training
X = df.drop(columns=['Outcome'])
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Prediction
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader('Prediction')
st.write('Diabetes' if prediction[0] == 1 else 'No Diabetes')

st.subheader('Prediction Probability')
st.write(prediction_proba)

# Visualization
st.subheader('Data Visualization')

if st.checkbox('Show raw data'):
    st.write(df)

st.subheader('Correlation Heatmap')
corr = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
st.pyplot(plt)

st.subheader('Pairplot')
sns.pairplot(df, hue='Outcome')
st.pyplot(plt)
