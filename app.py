import streamlit as st
import numpy as np
import mlem
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    st.title("Esame")

    path="https://frenzy86.s3.eu-west-2.amazonaws.com/python/data/formart_house.csv"

    df=pd.read_csv(path)
    
    df=df.iloc[:-1,:]
    df=df.astype(float)


    fig=plt.figure(figsize=(12,8))
    plt.title('Heatmap')
    sns.heatmap(df.corr(),annot=True,cmap="Blues")
    st.pyplot(fig)

    ##################################
    modelup=mlem.api.load('model_.mlem')

    number1 = st.number_input('Insert a number1', min_value=1,max_value=500)
    number2 = st.number_input('Insert a number2',min_value=1,max_value=500)
    number3 = st.number_input('Insert a number3',min_value=1,max_value=500)
    number4 = st.number_input('Insert a number4',min_value=1,max_value=500) 
    number5 = st.number_input('Insert a number5',min_value=1,max_value=500)
    number6 = st.number_input('Insert a number6',min_value=1,max_value=500)
    number7 = st.number_input('Insert a number7',min_value=1,max_value=500)
    number8 = st.number_input('Insert a number8',min_value=1,max_value=500)
    number9 = st.number_input('Insert a number9',min_value=1,max_value=500)
    number10 = st.number_input('Insert a number10',min_value=1,max_value=500)
    number11 = st.number_input('Insert a number11',min_value=1,max_value=500)
    number12 = st.number_input('Insert a number12',min_value=1,max_value=500)
    number13 = st.number_input('Insert a number13',min_value=1,max_value=500)   

    pred = modelup.predict([[number1, number2,number3, number4,number5,number6,number7,number8,number9, number10,number11,number12,number13]]) [0]
    st.write(round(pred,1))

if __name__=="__main__":
    main()   
