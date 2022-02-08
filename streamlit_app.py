import pandas as pd
import streamlit as st
from io import StringIO
from data_preview import data_preview




class DataObject():
    def __init__(self, df=None, filesize=None):
      self.df = df
      self.filesize = filesize
            
class Interface():
    def __init__(self):
      pass
    
    @classmethod
    def side_bar(cls, dt_obj):
      filename = st.sidebar.file_uploader("Upload a data file", type=(["csv", "data"]))                   # Accepts .csv and .data
      if filename is not None:                                                                            # Work with global variables  and 'filesize                                                                            # Try coma as a separator and proc an error if separator is different
        try:                                                                                            # Read data into a dataframe while recognizing a separator using 're'
            dt_obj.df = pd.read_csv(filename, sep=';|,', decimal=',', engine='python')                  # Exception for ';' separator                                                                       
            dt_obj.filesize = dt_obj.df.size
        except:
            filename.seek(0)
            filename = filename.read()
            filename = str(filename,'utf-8')
            filename = StringIO(filename)
            dt_obj.df = pd.read_csv(filename, sep=';', decimal=',', index_col = False)
            dt_obj.filesize = dt_obj.df.size

        data_prev_button = st.sidebar.button('Data Preview')
        data_prep_button = st.sidebar.button('Data Preparation')
        classification_button = st.sidebar.button('Classification')
        regression_button = st.sidebar.button('Regression')

        if data_prev_button:
          data_preview(dt_obj)

        if data_prep_button:
          st.header("DATA PREPARATION")
        if classification_button:
          st.header("CLASSIFICATION")
        if regression_button:
          st.header('REGRESSION')
        
        #menu = ['Data Preview', 'Data Preparation', 'Classification', 'Regression']
         

def main():
  data_main = DataObject()
  interface = Interface()
  
  st.title('Dashboard')
  interface.side_bar(data_main)

  if data_main.df is not None:
    pass

  
main()
