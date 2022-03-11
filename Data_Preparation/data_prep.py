# General import section
import pandas as pd #to work with dataframes
import streamlit as st #streamlit backend
import numpy as np #to work with numerical arrays
import os #to work with files
from streamlit_autorefresh import st_autorefresh #library for autorefresher https://libraries.io/pypi/streamlit-autorefresh


def import_dset(data_obj):
    """Checking if the processed dataset already exist, then current dataframe will be equal to that. 
       If it's not, initial dataframe will be generated into initial.csv file, and current dataframe will be equal to initial.

    :param data_obj: DataObject instance
    :type data_obj: __main__.DataObject
    :return: pandas dataframe object
    :rtype: pandas.core.frame.DataFrame
    """

    try:
        a = pd.read_csv('Smoothing_and_Filtering//Filtered Dataset.csv', index_col = None)
        if a.equals(data_obj.df) == False:
            current_df = a
        else:
            current_df = data_obj.df
            current_df.to_csv("Smoothing_and_Filtering//initial.csv", index=False)
    except:
        current_df = data_obj.df
        current_df.to_csv("Smoothing_and_Filtering//initial.csv", index=False)

    return current_df


def main(data_obj):
    """Data Preparation main

    :param data_obj: DataObject instance
    :type data_obj: __main__.DataObject
    """    """"""

    # Header  
    st.header("Data Preparation")
    st.info("""
               Here you can rename and/or drop columns.
               \nA field "Column to delete" is a multi-selector. You can choose more than one column to delete at once. 
               \nDon't forget to press 'Submit' each time to apply changes!
            """)

    # Dataframe assignement from data object
    current_df = import_dset(data_obj)

    # Reset dataframe 
    if st.sidebar.button("Reset dataframe to the initial one"):
        current_df = pd.read_csv('Smoothing_and_Filtering//initial.csv', index_col = None)
        # Check if file exists and remove it if it does
        if os.path.isfile("Smoothing_and_Filtering//Filtered Dataset.csv"):
            os.remove("Smoothing_and_Filtering//Filtered Dataset.csv")
        st.sidebar.success("Success!")
          
    cc1, cc2, cc3 = st.columns(3)
    
    # Display current dataframe
    with cc1:
        st.write("Dataframe display:")
        st.write(current_df)
    # The form for renaming columns       
    with cc2:
        st.write(" ")
        st.write(" ")
        st.write(" ")
        with st.form(key="form"):
            # Selecting the column to change 
            col_to_change = st.selectbox("Column to change", current_df.columns)
            new_col_name = st.text_input("New name", value="")
            submit_button = st.form_submit_button(label='Submit changes')
        
        # Submitting changes   
        if submit_button:     
            current_df = current_df.rename(columns={col_to_change: new_col_name})
            current_df.to_csv("Smoothing_and_Filtering//Filtered Dataset.csv", index=False)
            st_autorefresh(interval=50, limit=2, key="fizzbuzzcounter")      
            
    # The form for deleting columns            
    with cc3:
        st.write(" ")
        st.write(" ")
        st.write(" ")

        # Selecting the columns to delete  
        with st.form(key="form1"):
            col_to_delete  = st.multiselect('Columns to delete', current_df.columns)
            submit_button1 = st.form_submit_button(label='Submit changes')

        # Submitting changes 
        if submit_button1:        
            current_df = current_df.drop(columns=col_to_delete)
            current_df.to_csv("Smoothing_and_Filtering//Filtered Dataset.csv", index=False)
            st_autorefresh(interval=50, limit=2, key="fizzbuzzcounter")
                               

# Main
if __name__ == "__main__":
    main()