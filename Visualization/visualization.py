# General import section
import streamlit as st #streamlit backend

# Visualization import section
import seaborn as sns #for plotting
import matplotlib.pyplot as plt #to configure plots

# Data Preview
def Heatmap(data_obj):
    """Heatmap

    :param data_obj: DataObject instance
    :type data_obj: __main__.DataObject
    """
    fig = plt.figure(figsize=(16, 6))
    sns.heatmap(data_obj.df.corr(), vmin=-1, vmax=1, annot=True, fmt='.2%').set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12)
    st.pyplot(fig)

# Data Preparation: Outlier recognition
def DoubleBoxPlot(initdf, dataframe, column):
    """Boxplot that compares the selected column against the same one
       in original dataset.

    :param initdf: initial dataframe
    :type initdf: pandas.core.frame.DataFrame
    :param dataframe: current dataframe with changes
    :type dataframe: pandas.core.frame.DataFrame
    :param column: a column to be plotted against the original data
    :type column: str
    """
    fig, axes = plt.subplots(2,1, sharex=True, figsize=(10,4))
    sns.boxplot(initdf[column], ax=axes[0], color='skyblue', orient="v")
    axes[0].set_title("Original dataframe")
    sns.boxplot(dataframe[column], ax=axes[1], color='green', orient="v")
    axes[1].set_title("Resulting dataframe")
    fig.tight_layout()
    st.pyplot(fig)

def Histogram(dataframe, column):
    """Histogram

    :param dataframe: current dataframe with changes
    :type dataframe: pandas.core.frame.DataFrame
    :param column: a column to be plotted against the original data
    :type column: str
    """
    fig = plt.figure(figsize=(10, 4))
    sns.histplot(data=dataframe, x=column)
    st.pyplot(fig)

def ScatterPlot(initdf, dataframe, column1, column2):
    """Scatter plot

    :param initdf: initial dataframe
    :type initdf: pandas.core.frame.DataFrame
    :param dataframe: current dataframe with changes
    :type dataframe: pandas.core.frame.DataFrame
    :param column1: a column 1 to be plotted
    :type column1: str
    :param column2: a column 2 to be plotted
    :type column2: str
    """
    fig = plt.figure(figsize=(10, 4))
    sns.scatterplot(data=initdf, x=column1, y=column2)
    sns.scatterplot(data=dataframe, x=column1, y=column2)
    plt.legend(["Outliers","Original"])
    st.pyplot(fig)

# Data Preparation: Smoothing
def doubleLinePlot(initdf, dataframe, column):
    """Lineplot comparing changed data to an unchanged one

    :param initdf: initial dataframe
    :type initdf: pandas.core.frame.DataFrame
    :param dataframe: current dataframe with changes
    :type dataframe: pandas.core.frame.DataFrame
    :param column: a column to be plotted against the original data
    :type column: str
    """
    fig = plt.figure(figsize=(10, 4))
    sns.lineplot(y = column, x = [i for i in range(len(initdf[column]))], data = initdf)
    sns.lineplot(y = column, x = [i for i in range(len(dataframe[column]))], data = dataframe)
    st.pyplot(fig)

# Data Preparation: Interpolation
def interpolation_subplot(initdf, dataframe, column, method):
    """_summary_

    :param initdf: initial dataframe
    :type initdf: pandas.core.frame.DataFrame
    :param dataframe: current dataframe with changes
    :type dataframe: pandas.core.frame.DataFrame
    :param column: a column to be plotted against the original data
    :type column: str
    :param method: interpolation method used
    :type method: str
    """
    error = 0
    plt.rcParams.update({'xtick.bottom': False})
    #fig, axes = plt.subplots(1, 1, sharex=True, figsize=(20, 20))
    fig = plt.figure(figsize=(10, 4))
    if  method=='linear_fill':
        #initdf[column].plot(title='Actual', ax=axes[0], label='Actual', color='green', style=".-")
        dataframe[column].plot(title="{} (MSE: ".format(method) + str(error) + ")",  label='{}'.format(method),
                                  color='deeppink',
                                   style=".-")
        st.pyplot(fig)   
    elif method=='cubic_fill':
        #initdf[column].plot(title='Actual', ax=axes[0], label='Actual', color='green', style=".-")
        dataframe[column].plot(title="{} (MSE: ".format(method) + str(error) + ")",  label='{}'.format(method),
                                  color='deeppink',
                                   style=".-")
        st.pyplot(fig)                                                    
    else:
        #initdf[column].plot(title='Actual', ax=axes[0],  label='Actual', color='green', style=".-")
        dataframe[column].plot(title="{} (MSE: ".format(method) + str(error) + ")", label='{}'.format(method),
                                  color='deeppink',
                                   style=".-")
        st.pyplot(fig)        


# Unused
def linePlot_Out_recogn(dataframe, column):
    """Lineplot outlier recognition

    :param dataframe: current dataframe with changes
    :type dataframe: pandas.core.frame.DataFrame
    :param column: a column to be plotted against the original data
    :type column: str
    """
    fig = plt.figure(figsize=(10, 4))
    sns.lineplot(y = column, x = [i for i in range(len(dataframe[column]))], data = dataframe)
    st.pyplot(fig)

def BoxPlot(dataframe, column):
    """Boxplot

    :param dataframe: current dataframe with changes
    :type dataframe: pandas.core.frame.DataFrame
    :param column: a column to be plotted against the original data
    :type column: str
    """
    fig = plt.figure(figsize=(10, 4))
    sns.boxplot(y = dataframe[column], orient='v')
    st.pyplot(fig)