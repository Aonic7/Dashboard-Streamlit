import streamlit as st
import os

def main():
    """Welcome main
    """    
    st.header("Welcome")
    st.subheader("What is this?")
    st.write("""
             You find yourselves in the app for Data Analytics Dashboard developed by
             the students of MAIT 21/22 program of TH Köln.

             It allows you to:
             - Upload your own datasets
             - Rename and drop columns within them
             - Smooth, filter and interpolate
             - Perform advanced machine learning with customizable parameters on you data
             - Visualize and download the results
             """)
    st.subheader("Navigation")
    st.markdown("""
                0. **Welcome:** This is where you currently are
                1. **Data Preview:**  You can have a look at your dataset in general and spot some correlations between the features
                2. **Data Preparation:** Drop and/or rename single/multiple columns, don't forget to submit changes
                3. **Smoothing and Filtering:** Use a multitude of tools to trim or adjust your data to increase its quality. Don't forget to save and finalize the results! Even if you didn't change anything :)
                4. **Classification:** You can perform several classification methods (e.g. Random Forest) and get results as visualization and datasheet.  
                5. **Regression:** Predict the next data points using Neural Networks, Random Forest and other algorithms
                """)
    st.info("The experience from the workflow is the best when all the pages are navigated in sequence!")
    st.subheader("Source code")
    st.markdown("It can be found via navigating to the menu in the top right corner and pressing 'View App Source' or by using [this link](https://github.com/Aonic7/Dashboard-Streamlit).")

    # To delete leftover files from the previous runs
    if os.path.isfile("Smoothing_and_Filtering//Preprocessing dataset.csv"):
        os.remove("Smoothing_and_Filtering//Preprocessing dataset.csv")
            
    if os.path.isfile("Smoothing_and_Filtering//Filtered Dataset.csv"):
        os.remove("Smoothing_and_Filtering//Filtered Dataset.csv")

    if os.path.isfile("Smoothing_and_Filtering//initial.csv"):
        os.remove("Smoothing_and_Filtering//initial.csv")