import numpy as np
import streamlit as st
import joblib
import pandas as pd

model = joblib.load('accident.pkl')
df = pd.read_csv("/Users/User/Desktop/ML/ML_PROJECT/US_Accidents_Dec21_updated.csv")

def app():
    st.title('Accident severity prediction')
    
    dist = st.slider('Distance(mi)', 0,155)
    humid = st.slider('Humidity(%)', 0,100 )
    visible = st.slider('Visibility(mi)', 0,140)
    junct= st.slider('Junction', 0,1)
    wind = st.slider('Wind_Speed(mph)', 0,1087)
    rain = st.slider('Precipitation(in)', 0,24)
    

    if st.button('Predict'):
        input_data = pd.DataFrame({'Distance(mi)': [dist],'Humidity(%)': [humid],'Visibility(mi)': [visible],'Junction': [junct],'Wind_Speed(mph)': [wind],'Precipitation(in)': [rain],})
        prediction = model.predict(input_data.values)
        if prediction[0] == 1:
            st.write('This accident is a severe case with need of immediate medical attention.')
        else:
            st.write('This accident is not a severe case.')


if __name__ == '__main__':
    app()