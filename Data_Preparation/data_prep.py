import pandas as pd
import streamlit as st
import numpy as np
import os
from streamlit_autorefresh import st_autorefresh #https://libraries.io/pypi/streamlit-autorefresh


def import_dset(data_obj):
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
    st.header("Data Preparation")
    st.info("""
               Here you can rename and/or drop columns.
               \nA field "Column to delete" is a multi-selector. You can choose more than one column to delete at once. 
               \nDon't forget to press 'Submit' each time to apply changes!
            """)
    current_df = import_dset(data_obj)

    if st.sidebar.button("Reset dataframe to the initial one"):
        current_df = pd.read_csv('Smoothing_and_Filtering//initial.csv', index_col = None)
        if os.path.isfile("Smoothing_and_Filtering//Filtered Dataset.csv"):
            os.remove("Smoothing_and_Filtering//Filtered Dataset.csv")
        st.sidebar.success("Success!")
          
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        st.write("Dataframe display:")
        st.write(current_df)    
    with cc2:
        st.write(" ")
        st.write(" ")
        st.write(" ")
        with st.form(key="form"):
            col_to_change = st.selectbox("Column to change", current_df.columns)
            new_col_name = st.text_input("New name", value="")
            submit_button = st.form_submit_button(label='Submit changes')
            #st_autorefresh(interval=500, limit=2, key="fizzbuzzcounter1")
          
        if submit_button:
                
            current_df = current_df.rename(columns={col_to_change: new_col_name})
            current_df.to_csv("Smoothing_and_Filtering//Filtered Dataset.csv", index=False)
            st_autorefresh(interval=50, limit=2, key="fizzbuzzcounter")      
            
             
    with cc3:
        st.write(" ")
        st.write(" ")
        st.write(" ")
        with st.form(key="form1"):
            #col_to_delete = st.selectbox("Column to delete", current_df.columns)
            col_to_delete  = st.multiselect('Columns to delete', current_df.columns)
            submit_button1 = st.form_submit_button(label='Submit changes')
            #st_autorefresh(interval=500, limit=2, key="fizzbuzzcounter2")

        if submit_button1:
        
            current_df = current_df.drop(columns=col_to_delete)
            current_df.to_csv("Smoothing_and_Filtering//Filtered Dataset.csv", index=False)
            st_autorefresh(interval=50, limit=2, key="fizzbuzzcounter")
                               
             
if __name__ == "__main__":
    main()




# def main(data_obj):
#     cc1, cc2, cc3 = st.columns(3)
#     with cc1:
#         st.write(data_obj.df.columns)
    
#     with cc2:
#         with st.form(key="form"):
#             col_to_change = st.selectbox("Column to change", df.columns)
#             new_col_name = st.text_input("New name", value="")
#             submit_button = st.form_submit_button(label='Submit')

#         if submit_button:
#             df = df.rename(columns={col_to_change: new_col_name})

    


# if __name__ == "__main__":
#     main()