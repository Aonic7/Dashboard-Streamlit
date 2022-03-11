# General import section
import pandas as pd #to work with dataframes
import streamlit as st #streamlit backend
from io import StringIO #to read data files as .csv correctly
import os #to work with files

# Streamlit main page configuration
st.set_page_config(page_title="MAIT 21/22 Data Analytics Dashboard",
                    page_icon=None,
                    layout="wide",
                    initial_sidebar_state="expanded",
                    menu_items=None)

# App import
import Welcome_Page
import Data_Preview
import Data_Preparation
import Smoothing_and_Filtering
import Regression
import Classification

# Data object class
class DataObject():
    """
    Data object class holds a dataframe and its byte size.
    """
    def __init__(self, df=None, filesize=None):
      """The constructor for DataObject class

      :param df: pandas dataframe object, defaults to None
      :type df: pandas.core.frame.DataFrame, optional
      :param filesize: byte size of pandas dataframe, defaults to None
      :type filesize: numpy.int32, optional
      """
      self.df = df
      self.filesize = filesize


# Interface class        
class Interface():
    """
    Interface class contains a file picker and a side bar. It also handles the import of a data object.
    """
    def __init__(self):
      """The constructor for Interface class.
      """
      pass
    
    def side_bar(cls, dt_obj):
      """Sidebar configuration and file picker

      :param dt_obj: pandas dataframe object
      :type dt_obj: pandas.core.frame.DataFrame
      """
      # Accepts .csv and .data
      filename = st.sidebar.file_uploader("Upload a data file", type=(["csv", "data"]))                   
      if filename is not None: #file uploader selected a file      
        try: #most datasets can be read using standard 'read_csv'                                                                                           
            dt_obj.df = pd.read_csv(filename, sep=';|,', decimal=',', engine='python')                                                                     
            dt_obj.filesize = dt_obj.df.size
        except: #due to a different encoding some datafiles require additional processing
            filename.seek(0)
            filename = filename.read()
            filename = str(filename,'utf-8')
            filename = StringIO(filename)
            #now the standard 'read_csv' should work
            dt_obj.df = pd.read_csv(filename, sep=';', decimal=',', index_col = False)
            dt_obj.filesize = dt_obj.df.size
      
        # Side bar navigation menu with a select box
        menu = ['Welcome Page','Data Preview', 'Data Preparation', 'Smoothing and filtering',
                                                 'Classification', 'Regression']
        navigation = st.sidebar.selectbox(label="Select menu", options=menu)

        # Apps

        # Landing page
        if navigation == 'Welcome Page':
          with st.container():
           Welcome_Page.welcome()

        # Runs 'Data Preview' app
        if navigation == 'Data Preview':
          with st.container():
           Data_Preview.data_preview(dt_obj)

        # Runs 'Data Preparation' app
        if navigation == 'Data Preparation':
          with st.container():
           Data_Preparation.data_prep(dt_obj)

        # Runs 'Smoothing and filtering' app
        if navigation == 'Smoothing and filtering':
          Smoothing_and_Filtering.smoothing_and_filtering(dt_obj)
        
        # Runs 'Classification' app
        if navigation == 'Classification':
          Classification.classification(dt_obj)  

        # Runs 'Regression' app
        if navigation == 'Regression':
          Regression.regression(dt_obj)
      
      # Initial welcome page when there is no file selected
      else:
        Welcome_Page.welcome()
        # It deletes Preprocessing and initial datasets from the last run
        if os.path.isfile("Smoothing_and_Filtering//Preprocessing dataset.csv"):
           os.remove("Smoothing_and_Filtering//Preprocessing dataset.csv")
        if os.path.isfile("Smoothing_and_Filtering//initial.csv"):
           os.remove("Smoothing_and_Filtering//initial.csv")

def main():
  """
  Main and its Streamlit configuration
  """

  # Creating an instance of the original dataframe data object                   
  data_main = DataObject()
  # Creating an instance of the main interface
  interface = Interface()
  interface.side_bar(data_main)


# Run Main
if __name__ == '__main__':
  main()