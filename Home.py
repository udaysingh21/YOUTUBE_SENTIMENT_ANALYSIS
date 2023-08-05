import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


st.title('You Tube Sentiment Analysis')

# load the dataset1
@st.cache_data
def load_data1():
    df = pd.read_csv("datasets/UScomments.csv",low_memory=False,on_bad_lines='skip')
    return df

with st.spinner('Loading Data 1...'):
    df = load_data1()

if st.checkbox("Show Dataset 1"):
    st.subheader("Dataset")
    st.write(df)

# loading the dataset2
@st.cache_data
def load_data2():
    df = pd.read_csv("datasets/USvideos.csv",low_memory=False,on_bad_lines='skip')
    return df

with st.spinner('Loading Data 2...'):
    df = load_data2()

if st.checkbox("Show Dataset 2", key="show_dataset_checkbox"):
    st.subheader("Dataset")
    st.write(df)

