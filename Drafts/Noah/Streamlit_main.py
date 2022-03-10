# General import section
import pandas as pd  # to work with dataframes
import streamlit as st  # streamlit backend
from io import StringIO  # to read data files as .csv correctly

# Dashboard apps

# App import
# from Data_Preview import data_preview
import Streamlit_classification
import Streamlit_regression


class DataObject():
    """
    Data object class holds a dataframe and its byte size.
    """

    def __init__(self, df=None, filesize=None):
        """The constructor for DataObject class.

        Args:
            df (pandas.core.frame.DataFrame, optional): pandas dataframe object. Defaults to None.
            filesize (numpy.int32, optional): byte size of pandas dataframe. Defaults to None.
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

        Args:
            dt_obj (pandas.core.frame.DataFrame): pandas dataframe object.
        """
        # Accepts .csv and .data
        filename = st.sidebar.file_uploader("Upload a data file", type=(["csv", "data"]))
        if filename is not None:  # file uploader selected a file
            try:  # most datasets can be read using standard 'read_csv'
                dt_obj.df = pd.read_csv(filename, sep=';|,', decimal=',', engine='python')
                dt_obj.filesize = dt_obj.df.size
            except:  # due to a different encoding some datafiles require additional processing
                filename.seek(0)
                filename = filename.read()
                filename = str(filename, 'utf-8')
                filename = StringIO(filename)
                # now the standard 'read_csv' should work
                dt_obj.df = pd.read_csv(filename, sep=';', decimal=',', index_col=False)
                dt_obj.filesize = dt_obj.df.size

            # Side bar navigation menu with a select box
            menu = ['Classification',"Regression"]
            navigation = st.sidebar.selectbox(label="Select menu", options=menu)

            # Runs 'Classification' app
            if navigation == 'Classification':
                Streamlit_classification.main(dt_obj)

            if navigation == 'Regression':
                Streamlit_regression.main(dt_obj)



def main():
    """
    Main and its Streamlit configuration
    """
    st.set_page_config(page_title="CLassification Dashboard",
                       page_icon=None,
                       layout="wide",
                       initial_sidebar_state="expanded",
                       menu_items=None)
    # Creating an instance of the original dataframe data object
    data_main = DataObject()
    # Creating an instance of the main interface
    interface = Interface()
    interface.side_bar(data_main)


# Run Main
if __name__ == '__main__':
    main()