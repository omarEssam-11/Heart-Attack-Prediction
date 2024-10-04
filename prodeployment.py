import pandas as pd
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load the model, feature names, and country list
rfc = pickle.load(open('Heart_attack.sav', 'rb'))
trained_feature_names = pickle.load(open('trained_feature_names.pkl', 'rb'))
trained_country_list = pickle.load(open('trained_country_list.pkl', 'rb'))


st.title('Heart Attack Prediction')
st.info('Easy Application For Heart Attack Prediction')

st.sidebar.header('Feature Selection')

# Define text input fields
Age = st.text_input('Age')
Sex = st.selectbox('Sex', ('Male', 'Female'))  # Added sex input
Cholesterol = st.text_input('Cholesterol')
Blood_Pressure = st.text_input('Blood Pressure (format: systolic/diastolic)')
Heart_Rate = st.text_input('Heart Rate')
Diabetes = st.text_input('Diabetes')
Family_History = st.text_input('Family History')
Smoking = st.text_input('Smoking')
Obesity = st.text_input('Obesity')
Alcohol_Consumption = st.text_input('Alcohol Consumption')
Exercise_Hours_Per_Week = st.text_input('Exercise Hours Per Week')
Diet = st.selectbox('Diet', ('Unhealthy', 'Average', 'Healthy'))
Previous_Heart_Problems = st.text_input('Previous Heart Problems')
Medication_Use = st.text_input('Medication Use')
Stress_Level = st.text_input('Stress Level')
Sedentary_Hours_Per_Day = st.text_input('Sedentary Hours Per Day')
Income = st.text_input('Income')
BMI = st.text_input('BMI')
Triglycerides = st.text_input('Triglycerides')
Physical_Activity_Days_Per_Week = st.text_input('Physical Activity Days Per Week')
Sleep_Hours_Per_Day = st.text_input('Sleep Hours Per Day')
Country = st.selectbox('Country', ['India','Italy','Japan','South Africa','New Zealand','Vietnam',
                                   'Spain','United Kingdom','France','Brazil','China','Canada','Germany',
                                   'Argentina','Australia','Thailand','Colombia','United States','Nigeria','South Korea'])

Age = (Age)
Cholesterol = (Cholesterol)
Heart_Rate = (Heart_Rate)
Blood_Pressure_Systolic = (Blood_Pressure)
Blood_Pressure_Diastolic = (Blood_Pressure)
Exercise_Hours_Per_Week = (Exercise_Hours_Per_Week)
Sedentary_Hours_Per_Day = (Sedentary_Hours_Per_Day)
Income = (Income)
BMI = (BMI)
Triglycerides = (Triglycerides)
Physical_Activity_Days_Per_Week = (Physical_Activity_Days_Per_Week)
Sleep_Hours_Per_Day = (Sleep_Hours_Per_Day)
Stress_Level = (Stress_Level)

data_input = {
    'Age': [Age],
    'Sex': [Sex],
    'Cholesterol': [Cholesterol],
    'Heart Rate': [Heart_Rate],
    'Diabetes': [Diabetes],
    'Family History': [Family_History],
    'Smoking': [Smoking],
    'Obesity': [Obesity],
    'Alcohol Consumption': [Alcohol_Consumption],
    'Exercise Hours Per Week': [Exercise_Hours_Per_Week],
    'Diet': [Diet],
    'Previous Heart Problems': [Previous_Heart_Problems],
    'Medication Use': [Medication_Use],
    'Stress Level': [Stress_Level],
    'Sedentary Hours Per Day': [Sedentary_Hours_Per_Day],
    'Income': [Income],
    'BMI': [BMI],
    'Triglycerides': [Triglycerides],
    'Physical Activity Days Per Week': [Physical_Activity_Days_Per_Week],
    'Sleep Hours Per Day': [Sleep_Hours_Per_Day],
    'Country': [Country],
    'Blood_Pressure': [Blood_Pressure]
}

df = pd.DataFrame(data_input)

print(df.head())

def process_blood_pressure(bp_value):
    try:
        systolic = int(str(bp_value).split('/')[0])
        diastolic = int(str(bp_value).split('/')[1])
        return systolic, diastolic
    except (ValueError, IndexError):
        return None, None

df['systolic_pressure'], df['diastolic_pressure'] = zip(*df['Blood_Pressure'].apply(process_blood_pressure))

if df['systolic_pressure'].isnull().any() or df['diastolic_pressure'].isnull().any():
    st.sidebar.error('Invalid Blood Pressure format. Please enter values in the format systolic/diastolic.')
else:
    df.drop(columns='Blood_Pressure', axis=1, inplace=True)

    df = pd.get_dummies(df, columns=['Sex'])

    df['Diet'] = df['Diet'].apply(lambda x: {'Unhealthy': 0, 'Average': 1, 'Healthy': 2}[x])

    df['Country'] = df['Country'].apply(lambda x: trained_country_list.index(x) + 1)

    to_std = ['Age','Cholesterol','Heart Rate', 'Exercise Hours Per Week','Sedentary Hours Per Day','Income','BMI','Triglycerides', 'systolic_pressure','diastolic_pressure']
    to_minmax = ['Stress Level','Sleep Hours Per Day','Physical Activity Days Per Week']

    std_scaler = StandardScaler()
    minmax_scaler = MinMaxScaler()

    df[to_std] = std_scaler.fit_transform(df[to_std])
    df[to_minmax] = minmax_scaler.fit_transform(df[to_minmax])

    df = df.reindex(columns=trained_feature_names, fill_value=0)



    prediction = rfc.predict(df)[0]

    con = st.sidebar.button('Confirm')

    if con:
        if prediction == 0:
            st.sidebar.write('The Patient Has No Heart Attack')
            st.sidebar.image('https://almasdare.com/wp-content/uploads/2019/11/heart.jpg', width=150)
        else:
            st.sidebar.write('The Patient Has Heart Attack')
            st.sidebar.image('https://static.dw.com/image/61450388_1004.webp', width=150)
