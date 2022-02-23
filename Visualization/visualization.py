# General import section
import pandas as pd #to work with dataframes
import streamlit as st #streamlit backend

# Visualization import section
import seaborn as sns #for plotting
import matplotlib.pyplot as plt #to configure plots

# Data Preview
def Heatmap(data_obj):
    fig = plt.figure(figsize=(16, 6))
    sns.heatmap(data_obj.df.corr(), vmin=-1, vmax=1, annot=True, fmt='.2%').set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12)
    st.pyplot(fig)

# Data Preparation: Outlier recognition
def DoubleBoxPlot(initdf, dataframe, column):
    fig, axes = plt.subplots(2,1, sharex=True, figsize=(10,4))
    sns.boxplot(initdf[column], ax=axes[0], color='skyblue', orient="v")
    axes[0].set_title("Original dataframe")
    sns.boxplot(dataframe[column], ax=axes[1], color='green', orient="v")
    axes[1].set_title("Resulting dataframe")
    fig.tight_layout()
    st.pyplot(fig)

def Histogram(dataframe, column):
    fig = plt.figure(figsize=(10, 4))
    sns.histplot(data=dataframe, x=column)
    st.pyplot(fig)

def ScatterPlot(initdf, dataframe, column1, column2):
    fig = plt.figure(figsize=(10, 4))
    sns.scatterplot(data=initdf, x=column1, y=column2)
    sns.scatterplot(data=dataframe, x=column1, y=column2)
    plt.legend(["Outliers","Original"])
    st.pyplot(fig)

# Data Preparation: Smoothing
def doubleLinePlot(initdf, dataframe, column):
    fig = plt.figure(figsize=(10, 4))
    sns.lineplot(y = column, x = [i for i in range(len(initdf[column]))], data = initdf)
    sns.lineplot(y = column, x = [i for i in range(len(dataframe[column]))], data = dataframe)
    st.pyplot(fig)

# Unused
def linePlot_Out_recogn(dataframe, column):
    fig = plt.figure(figsize=(10, 4))
    sns.lineplot(y = column, x = [i for i in range(len(dataframe[column]))], data = dataframe)
    st.pyplot(fig)

def BoxPlot(dataframe, column):
    fig = plt.figure(figsize=(10, 4))
    sns.boxplot(y = dataframe[column], orient='v')
    st.pyplot(fig)