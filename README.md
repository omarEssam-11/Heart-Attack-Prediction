# Heart Attack Prediction



## Introduction
Heart disease is one of the leading causes of death worldwide. Early detection and intervention are crucial in reducing the mortality rate. This project leverages machine learning techniques to predict the risk of a heart attack using patient health data. The objective is to build a model that can provide real-time predictions, helping healthcare professionals make better-informed decisions.

# About

This project aims to predict the likelihood of a heart attack using a data set that includes various health and lifestyle factors. The prediction is based on the identification of key features that contribute significantly to the assessment of heart attack risk.

## Features

The dataset includes the following features:


- **Patient ID:** Unique identifier for each patient
- **Age:** Age of the patient
- **Sex:** Gender of the patient (Male/Female)
- **Cholesterol:** Cholesterol levels of the patient
- **Blood Pressure:** Blood pressure of the patient (systolic/diastolic)
- **Heart Rate:** Heart rate of the patient
- **Diabetes:** Whether the patient has diabetes (Yes/No)
- **Family History:** Family history of heart-related problems (1: Yes, 0: No)
- **Smoking:** Smoking status of the patient (1: Smoker, 0: Non-smoker)
- **Obesity:** Obesity status of the patient (1: Obese, 0: Not obese)
- **Alcohol Consumption:** Level of alcohol consumption by the patient (None/Light/Moderate/Heavy)
- **Exercise Hours Per Week:** Number of exercise hours per week
- **Diet:** Dietary habits of the patient (Healthy/Average/Unhealthy)
- **Previous Heart Problems:** Previous heart problems of the patient (1: Yes, 0: No)
- **Medication Use:** Medication usage by the patient (1: Yes, 0: No)
- **Stress Level:** Stress level reported by the patient (1-10)
- **Sedentary Hours Per Day:** Hours of sedentary activity per day
- **Income:** Income level of the patient
- **BMI:** Body Mass Index (BMI) of the patient
- **Triglycerides:** Triglyceride levels of the patient
- **Physical Activity Days Per Week:** Days of physical activity per week
- **Sleep Hours Per Day:** Hours of sleep per day
- **Country:** Country of the patient
- **Continent:** Continent where the patient resides
- **Hemisphere:** Hemisphere where the patient resides
- **Heart Attack Risk (Outcome):** Presence of heart attack risk (1: Yes, 0: No)

## Project Overview
This project consists of:
- **Data preprocessing**: Cleaning and preparing the clinical dataset for training.
- **Feature engineering**: Selecting the most important features that impact heart attack risk.
- **Model building**: Training and evaluating multiple machine learning models.
- **Model deployment**: Building a web application for real-time predictions using the trained model.

## Features
- Predict heart attack risk using clinical health indicators such as cholesterol, blood pressure, and other metrics.
- A user-friendly web interface for inputting patient data and receiving predictions.
- Models optimized through hyperparameter tuning to enhance accuracy.
  

1. Clone the repository:
   ```bash
   git clone https://github.com/omarEssam-11/heart-attack-prediction.git
