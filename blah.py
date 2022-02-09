# Streamlit main
import streamlit as st
# Data manipulation
import pandas as pd
# Data profiling
import pandas_profiling as pp
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import streamlit.components.v1 as components
# Visualization 
import matplotlib.pyplot as plt
import seaborn as sns
# Verification
from io import StringIO

###################################################
#################### MAIN #########################
###################################################

# Global variables:
df = None                                                                     # Sets dataframe to None when no data file is selected
filesize = 0                                                                  # File size tracker. Default is 0 when no file is selected

# Outlier recognition ("_or") variables
# None by default
column_select_or = None                                                       # Selected column
std_coeff = None                                                              # Standard deviation coefficient
filtered_or = None                                                            # Filtered outlier recognition dataframe

def file_picker() -> tuple:
    # File picker sidebar title
    st.sidebar.title('User Inputs')

    # File picker widget in sidebar
    filename = st.sidebar.file_uploader("Upload a data file", type=(["csv", "data"]))                   # Accepts .csv and .data
    if filename is not None:                                                                            # Work with global variables 'df' and 'filesize'
        global df, filesize                                                                             # Try coma as a separator and proc an error if separator is different
        try:                                                                                            # Read data into a dataframe while recognizing a separator using 're'
            dataframe = pd.read_csv(filename, sep=';|,', decimal=',', engine='python')                  # Exception for ';' separator
            df = dataframe                                                                              # Below steps are needed to avoid Streamlit memory error
            filesize = dataframe.size
        except:
            filename.seek(0)
            filename = filename.read()
            filename = str(filename,'utf-8')
            filename = StringIO(filename)
            dataframe = pd.read_csv(filename, sep=';', decimal=',', index_col = False)
            df = dataframe
            filesize = dataframe.size

        #Output variable from file uploader
        st.subheader("Original dataframe")
        st.dataframe(df)
        st.write(df.shape)
        st.write(df.describe())
        #st_profile_report(ProfileReport(df))

def explore_dataset():
    pass

def generate_df_report():
    st.sidebar.subheader("Dataset exploration")

    df_report = st.sidebar.checkbox("Tick the box if you want to explore the data")

    if df_report:
        with st.expander("REPORT", expanded=True):
            report = pp.ProfileReport(df, title="Blah-blah").to_html()
            components.html(report, height=1000, width=1000, scrolling=True)

            #st_profile_report(ProfileReport(df))

            # report = pp.ProfileReport(df, title="Blah-blah").to_file("output.html")
            # #report.html.style.full_width=True
            # #components.html(report, height=1000, width=820, scrolling=True)
            # st_profile_report(report)

            # report = pp.ProfileReport(df, title="Blah-blah").to_html()
            # report.html.style.full_width=True
            # components.html(report, height=1000, width=820, scrolling=True)

            # report = df.profile_report(title="Blah-blah", config.html.full_width=True).to_html()
            # #report.full_width=True
            # components.html(report)
            # #components.html(report, height=1000, width=820, scrolling=True)

            # report = pp.ProfileReport(df, title="Blah-blah", config_file="config.yaml").to_html()
            # components.html(report, height=1000, width=820, scrolling=True)


def data_prep_sidebar():
    prep_selection = st.sidebar.selectbox(label="Select data preparation method",
                    options=("",
                        "Outlier recognition",
                            "Interpolation",
                            "Smoothing"))
    return prep_selection

def data_prep_outlier_inputs():
    global column_select_or, std_coeff
    std_coeff = st.number_input("Enter standard deviation coefficient (multiplier) ", 0.0, 3.0, 2.0, 0.1)
    columns_list = list(df.select_dtypes(exclude=['object']).columns)   
    with st.form(key='outlier form'):
        column_select_or = st.selectbox(
            label='Select a column',
            options=columns_list
        )
        submit_button = st.form_submit_button(label='Submit')
    if submit_button:
        return column_select_or, std_coeff

def Outlier_recognition():
    global df
    mean = df[column_select_or].mean()
    std = df[column_select_or].std()
    fromVal = mean - std_coeff * std
    toVal = mean + std_coeff * std 
    global filtered_or
    filtered_or = df[(df[column_select_or] >= fromVal) & (df[column_select_or] <= toVal)]
    return filtered_or

def linePlot_Out_recogn():
    fig = plt.figure(figsize=(10, 4))
    sns.lineplot(y = column_select_or, x = [i for i in range(len(filtered_or[column_select_or]))], data = filtered_or)
    st.pyplot(fig)

def main():
    # add title
    st.header('Streamlit Testing')
    # add high level site inputs
    file_picker()
    generate_df_report()
    # with st.expander("REPORT", expanded=True):
    #         st_profile_report(pr)
    if filesize > 0:
        if data_prep_sidebar() == "Outlier recognition":
            data_prep_outlier_inputs()
            Outlier_recognition()
            st.write(filtered_or)
            st.write(filtered_or.shape)
            st.write(column_select_or)
            linePlot_Out_recogn()

main()