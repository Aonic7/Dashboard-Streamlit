import pandas as pd
import streamlit as st
from io import StringIO
from data_preview import data_preview_run
from data_preparation import data_preparation_run



class DataObject():
    def __init__(self, df=None, filesize=None):
      self.df = df
      self.filesize = filesize

            
class Interface():
    def __init__(self):
      pass
    
    # @classmethod
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

        menu = ['Data Preview', 'Data Preparation', 'Classification', 'Regression']
        navigation = st.sidebar.selectbox(label="Select menu", options=menu)

        if navigation == 'Data Preview':
          with st.container():
           data_preview_run(dt_obj)


        if navigation == 'Data Preparation':
          data_preparation_run(dt_obj)
        
        if navigation == 'Classification':
          st.header("CLASSIFICATION")
        if navigation == 'Regression':
          st.header('REGRESSION')
         

def main():
  st.set_page_config(page_title="MAIT 21/22 Data Analytics Dashboard",
                     page_icon=None,
                     layout="wide",
                     initial_sidebar_state="expanded",
                     menu_items=None)
  data_main = DataObject()
  interface = Interface()
  interface.side_bar(data_main)

  st.dataframe(pd.read_csv("Prepared Dataset.csv"))

if __name__ == '__main__':
  main()