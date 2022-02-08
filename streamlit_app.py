import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import seaborn as sns
from io import StringIO
from pandas.api.types import is_numeric_dtype

class DataObject():
    def __init__(self, df_raw=None):

      self.df_raw = df_raw
      
      filename = st.sidebar.file_uploader("Upload a data file", type=(["csv", "data"]))                   # Accepts .csv and .data
      if filename is not None:                                                                            # Work with global variables 'df_raw' and 'filesize                                                                            # Try coma as a separator and proc an error if separator is different
        try:                                                                                            # Read data into a dataframe while recognizing a separator using 're'
            self.df_raw = pd.read_csv(filename, sep=';|,', decimal=',', engine='python')                  # Exception for ';' separator                                                                       
            self.filesize = self.df_raw.size
        except:
            filename.seek(0)
            filename = filename.read()
            filename = str(filename,'utf-8')
            filename = StringIO(filename)
            self.df_raw = pd.read_csv(filename, sep=';', decimal=',', index_col = False)

            
class Interface(DataObject):
    def __init__(self):
      if data_main.df_raw is not None:
        self.show_original_df = st.sidebar.checkbox("Show original dataframe")
        if self.show_original_df:
            st.subheader("Original dataframe")
            st.dataframe(data_main.df_raw)
            st.write(data_main.df_raw.shape)
    
    def show_dataframe(self, dataframe):
      if dataframe is not None:
        self.dataframe = st.write(dataframe)

data_main = DataObject()
Interface()