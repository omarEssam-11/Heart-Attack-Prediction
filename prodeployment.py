import pandas as pd
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load the model, feature names, and country list
rfc = pickle.load(open('Heart_attack.sav', 'rb'))
trained_feature_names = pickle.load(open('trained_feature_names.pkl', 'rb'))
trained_country_list = pickle.load(open('trained_country_list.pkl', 'rb'))


st.title('Heart Attack Prediction')
st.info('Your Health, Our Priority – Predict Your Heart Attack Risk in Just a Few Clicks!')

st.sidebar.header('Predict')

# Define text input fields
# Input Fields
Age = st.slider('Age', min_value=1, max_value=99, value=30, help="Your age in years.")
Sex = st.selectbox('Sex', ('Male', 'Female'))
Cholesterol = st.slider('Cholesterol', min_value=100, max_value=400, value=200, help="Cholesterol in mg/dL")
Blood_Pressure = st.text_input('Blood Pressure (format: systolic/diastolic)')
Heart_Rate = st.slider('Heart Rate', min_value=20, max_value=200, value=75)
BMI = st.slider('BMI', min_value=10.0, max_value=50.0, step=0.1, value=22.0)

# Yes/No Questions
Diabetes = st.selectbox('Do you have diabetes?', ('No', 'Yes'))
Family_History = st.selectbox('Do you have a family history of heart disease?', ('No', 'Yes'))
Smoking = st.selectbox('Do you smoke?', ('No', 'Yes'))
Obesity = st.selectbox('Are you considered obese?', ('No', 'Yes'))
Alcohol_Consumption = st.selectbox('Do you consume alcohol?', ('No', 'Yes'))
Exercise_Hours_Per_Week = st.slider('Exercise Hours Per Week', min_value=0, max_value=20, value=3, help="Hours of exercise per week")
Previous_Heart_Problems = st.selectbox('Have you had previous heart problems?', ('No', 'Yes'))
Medication_Use = st.selectbox('Do you use any medication regularly?', ('No', 'Yes'))
Stress_Level = st.slider('Stress Level', min_value=0, max_value=10, value=5, help="Rate your stress level from 0 (no stress) to 10 (high stress)")
Sedentary_Hours_Per_Day = st.slider('Sedentary Hours Per Day', min_value=0, max_value=24, value=8, help="How many hours a day do you spend sitting or lying down?")
Income = st.text_input('Income', help="Your annual income in USD")
Triglycerides = st.slider('Triglycerides', min_value=50, max_value=900, value=150, help="Your triglyceride levels in mg/dL")
Physical_Activity_Days_Per_Week = st.slider('Physical Activity Days Per Week', min_value=0, max_value=7, value=3, help="Number of days per week you engage in physical activity")
Sleep_Hours_Per_Day = st.slider('Sleep Hours Per Day', min_value=0, max_value=24, value=7, help="How many hours of sleep do you get per day?")
Diet = st.selectbox('How would you describe your diet?', ('Unhealthy', 'Average', 'Healthy'))
Country = st.selectbox('Country', ['India', 'Italy', 'Japan', 'South Africa', 'New Zealand', 'Vietnam', 'Spain', 'United Kingdom', 'France', 'Brazil', 'China', 'Canada', 'Germany', 'Argentina', 'Australia', 'Thailand', 'Colombia', 'United States', 'Nigeria', 'South Korea'])


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

# Transform Yes/No inputs into 0/1
Diabetes = 1 if Diabetes == 'Yes' else 0
Family_History = 1 if Family_History == 'Yes' else 0
Smoking = 1 if Smoking == 'Yes' else 0
Obesity = 1 if Obesity == 'Yes' else 0
Alcohol_Consumption = 1 if Alcohol_Consumption == 'Yes' else 0
Previous_Heart_Problems = 1 if Previous_Heart_Problems == 'Yes' else 0
Medication_Use = 1 if Medication_Use == 'Yes' else 0

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
    'Blood Pressure': [Blood_Pressure]
}

data_set= pd.read_csv('https://raw.githubusercontent.com/alperrkilic/heart-attack-prediction/main/heart_attack_prediction_dataset.csv')

data_set.drop(['Patient ID', 'Hemisphere', 'Continent','Heart Attack Risk'], axis=1, inplace=True)

df = pd.DataFrame(data_input)

df=pd.concat([df,data_set])

def process_blood_pressure(bp_value):
    try:
        systolic = int(str(bp_value).split('/')[0])
        diastolic = int(str(bp_value).split('/')[1])
        return systolic, diastolic
    except (ValueError, IndexError):
        return None, None

df['systolic_pressure'], df['diastolic_pressure'] = zip(*df['Blood Pressure'].apply(process_blood_pressure))

if df['systolic_pressure'].isnull().any() or df['diastolic_pressure'].isnull().any():
    st.sidebar.error('Invalid Blood Pressure format. Please enter values in the format systolic/diastolic.')
else:
    df.drop(columns='Blood Pressure', axis=1, inplace=True)

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
