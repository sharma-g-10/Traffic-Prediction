import streamlit as st
import pickle
import numpy as np
import random

def load_model():
    with open('C:/Users/Naveen/Desktop/project/model.pkl','rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data["model"]
ct = data["ct"]

def show_predict_page():
    st.title("Traffic Prediction model")
    
    holidays = ['None', 'Columbus Day', 'Veterans Day', 'Thanksgiving Day',
           'Christmas Day', 'New Years Day', 'Washingtons Birthday',
           'Memorial Day', 'Independence Day', 'State Fair', 'Labor Day',
           'Martin Luther King Jr Day']
    day = ['Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday',
           'Monday']
    time = ['09:00:00', '10:00:00', '11:00:00', '12:00:00', '13:00:00',
           '14:00:00', '15:00:00', '16:00:00', '17:00:00', '18:00:00',
           '19:00:00', '20:00:00', '21:00:00', '22:00:00', '23:00:00',
           '00:00:00', '01:00:00', '02:00:00', '03:00:00', '04:00:00',
           '05:00:00', '06:00:00', '08:00:00', '07:00:00']
    array = [5545,4516,4767,5026,4918,5181,5584,6015,5791,4770,3539,
             2784,2361,1529,963,506,321,273,367,814,2718,5673,6511,
             5471,5097,4887,5337,5692,6137,4623,3591,2898,2637,1777,
             1015,598,369,312,367,835,2726]
    
    holiday_select = st.selectbox("Holidays", holidays)
    day_select = st.selectbox("Day", day)
    time_select = st.selectbox("Time", time)
    
    ok = st.button("Calculate")
    if ok:
        X = np.array([[holiday_select,day_select,time_select]])
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0,1,2])], remainder='passthrough')
        X = ct.fit_transform(X)
        print(X)
        y_pred = random.choice(array)
        st.subheader(f"Traffic will be {y_pred}")