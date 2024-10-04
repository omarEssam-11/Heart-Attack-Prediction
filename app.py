# Libraries
import pandas as pd
import numpy as np
import pickle
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, classification_report

# Reading the dataset
df = pd.read_csv('https://raw.githubusercontent.com/alperrkilic/heart-attack-prediction/main/heart_attack_prediction_dataset.csv')

# Drop unnecessary columns
df.drop(['Patient ID', 'Hemisphere', 'Continent'], axis=1, inplace=True)
# Map Country to risk ranking
coun = (df[df['Heart Attack Risk'] == 1]['Country'].value_counts() / 
        (df[df['Heart Attack Risk'] == 1]['Country'].value_counts() + 
         df[df['Heart Attack Risk'] == 0]['Country'].value_counts()) * 100).sort_values(ascending=True).index.to_list()

# Save country risk ranking
pickle.dump(coun, open('trained_country_list.pkl', 'wb'))

def risk_map(x):
    return coun.index(x) + 1

df['Country'] = df['Country'].apply(risk_map)

# Clean Blood Pressure column
def handle_blood_pressure_systolic(value):
    value = str(value).split('/')
    return int(value[0])

def handle_blood_pressure_diastolic(value):
    value = str(value).split('/')
    return int(value[1])

df['systolic_pressure'] = df['Blood Pressure'].apply(handle_blood_pressure_systolic)
df['diastolic_pressure'] = df['Blood Pressure'].apply(handle_blood_pressure_diastolic)
df.drop(columns='Blood Pressure', axis=1, inplace=True)

# Encoding categorical columns
df = pd.get_dummies(df, columns=['Sex'])

def diet_map(x):
    return {'Unhealthy': 0, 'Average': 1, 'Healthy': 2}[x]

df['Diet'] = df['Diet'].apply(diet_map)

# Map Country to risk ranking
coun = (df[df['Heart Attack Risk'] == 1]['Country'].value_counts() / 
        (df[df['Heart Attack Risk'] == 1]['Country'].value_counts() + 
         df[df['Heart Attack Risk'] == 0]['Country'].value_counts()) * 100).sort_values(ascending=True).index.to_list()

def risk_map(x):
    return coun.index(x) + 1

df['Country'] = df['Country'].apply(risk_map)

# Scaling the numerical columns
to_std = ['Age','Cholesterol','Heart Rate', 'Exercise Hours Per Week','Sedentary Hours Per Day','Income','BMI','Triglycerides', 'systolic_pressure','diastolic_pressure']
to_minmax = ['Stress Level','Sleep Hours Per Day','Physical Activity Days Per Week']

std_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()

df[to_std] = std_scaler.fit_transform(df[to_std])
df[to_minmax] = minmax_scaler.fit_transform(df[to_minmax])

# Over-sampling to balance the dataset
runs = RandomOverSampler(sampling_strategy=1)
df, df['Heart Attack Risk'] = runs.fit_resample(df, df['Heart Attack Risk'])

# Splitting the data
x = df.drop('Heart Attack Risk', axis=1)
y = df['Heart Attack Risk']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

# Training the RandomForest model
rfc = RandomForestClassifier(bootstrap=False, max_depth=100, max_features=2, min_samples_leaf=2, min_samples_split=10, n_estimators=750)
rfc.fit(x_train, y_train)

# Save the trained model and feature names
pickle.dump(rfc, open('Heart_attack.sav', 'wb'))
pickle.dump(x_train.columns.to_list(), open('trained_feature_names.pkl', 'wb'))

# Evaluation function
def evaluate_and_plot(model, x_test, y_test):
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f'Accuracy: {accuracy:.3f}')
    print(f'AUC: {auc:.3f}')
    print(f'Precision: {precision:.3f}')
    print(f'Recall: {recall:.3f}')
    print(classification_report(y_test, y_pred))

# Evaluate the model
evaluate_and_plot(rfc, x_test, y_test)
