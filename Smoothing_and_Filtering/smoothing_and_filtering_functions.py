# General import section
import pandas as pd #to work with dataframes
import streamlit as st #streamlit backend
import numpy as np #to work with arrays and series
import scipy.stats as stats #to work with z-score

from scipy.signal import medfilt, savgol_filter
from scipy.interpolate import interp1d
class Remove_Outliers():
    def __init__(self):
        pass

    # Removing outliers using a standard deviation
    def removeOutlier(df, columnName, n):
        """Remove outliers using a standard deviation in range from 1.0 to 3.0.

        Args:
            df (pandas.core.frame.DataFrame): input dataframe
            columnName (str): a column of the input dataframe
            n (float): standard deviation coefficient

        Returns:
            pandas.core.frame.DataFrame: filtered dataframe
        """
        mean = df[columnName].mean()
        std = df[columnName].std()  
        fromVal = mean - n * std 
        toVal = mean + n * std 
        filtered = df[(df[columnName] >= fromVal) & (df[columnName] <= toVal)] #apply the filtering formula to the column
        return filtered

    # Removing outliers using quantile range
    def removeOutlier_q(df, columnName, n1, n2):
        """Remove outliers using a quantile range from 0.0 to 1.0

        Args:
            df (pandas.core.frame.DataFrame): input dataframe
            columnName (str): a column of the input dataframe
            n1 (float): lower quantile
            n2 (float): upper quantile

        Returns:
            pandas.core.frame.DataFrame: filtered dataframe
        """
        lower_quantile, upper_quantile = df[columnName].quantile([n1, n2]) #quantiles are generally expressed as a fraction (from 0 to 1)
        filtered = df[(df[columnName] > lower_quantile) & (df[columnName] < upper_quantile)]
        return filtered

    # Removing outliers using z-score
    def removeOutlier_z(df, columnName, n):
        """Remove outliers using z-score.

        Args:
            df (pandas.core.frame.DataFrame): input dataframe
            columnName (str): a column of the input dataframe
            n (float): standard deviation coefficient

        Returns:
            pandas.core.frame.DataFrame: filtered dataframe
        """
        z = np.abs(stats.zscore(df[columnName])) #find the Z-score for the column
        filtered = df[(z < n)] #apply the filtering formula to the column
        return filtered

class Converter():
   
    def dateTime_converter(df):
      for col in df.columns:
           if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_datetime(df[col])
                    return df
                except ValueError:
                    pass   

class Smoothing():
    def __init__(self):
        pass
    

    def median_filter(df, columnName, filter_length):
        """Data smoothing using median filter function

        Args:
            df (pandas.core.frame.DataFrame): input dataframe
            columnName (str): a column of the input dataframe
            filter_length (int): filter length parameter 

        Returns:
            pandas.core.frame.DataFrame: df_var dataframe
        """
        
        df_var = df.copy()
        medfilt_tair = medfilt(df_var[columnName], filter_length)
        df_var[columnName] = medfilt_tair
        return df_var
    

    def moving_average(df, columnName, filter_length):
        """Data smoothing using moving average function

        Args:
            df (pandas.core.frame.DataFrame): input dataframe
            columnName (str): a column of the input dataframe
            filter_length (int): filter length parameter 

        Returns:
            pandas.core.frame.DataFrame: df_var dataframe
        """
        df_var = df.copy()
        df_var[columnName] = df[columnName].rolling(filter_length).mean()
        df_var.dropna(inplace=True)
        df_var.reset_index(drop=True)

        return df_var

    def savitzky_golay(df, columnName, filter_length, order):
        """Data smoothing using savitzky-golay function

        Args:
            df (pandas.core.frame.DataFrame): input dataframe
            columnName (str): a column of the input dataframe
            filter_length (int): filter length parameter 
            order (int): order of the polynomial to fit the function

        Returns:
            pandas.core.frame.DataFrame: df_var dataframe
        """
        df_var = df.copy()
        df_var[columnName] = savgol_filter(df_var[columnName], filter_length, order)
        df_var.dropna(inplace=True)
        df_var.reset_index(drop=True)

        return df_var

class TimeSeriesOOP():
    def __init__(self, current_df, selected_column, time_column):
        current_df=current_df.set_index("{}".format(time_column))
        self.df = current_df
        self.process_dataframe()
        
        #self.df_ffill = self.df.ffill( )  # df.ffill-pandas func to forward fill missing values
        #self.df_bfill = self.df.bfill( )  # df.ffill-pandas func to backward fill missing values
        
    def process_dataframe(self):  # make separate func if you need more processing
        self.df = self.df.resample('15min').mean( )
        
    
    def int_df_ffill(self):
        return self.df.ffill()  
    
    def int_df_bfill(self):
        return self.df.bfill()  
    
    def make_interpolation_liner(self, column_of_interest):
        # 4. Linear Interpolation ------------------
        self.df['rownum'] = np.arange(self.df.shape[0])  # df.shape[0]-gives number of row count
        df_nona = self.df.dropna(subset=[column_of_interest])  # df.dropna- Remove missing values.
        f = interp1d(df_nona['rownum'], df_nona[column_of_interest], kind='linear')
        self.df[column_of_interest] = f(self.df['rownum'])
        #self.df = self.df.dropna() 
        return self.df

    def make_interpolation_cubic(self, column_of_interest):
        # 5. Cubic Interpolation --------------------
        self.df['rownum'] = np.arange(self.df.shape[0]) 
        df_nona1 = self.df.dropna(subset=[column_of_interest]) 
        f2 = interp1d(df_nona1['rownum'], df_nona1[column_of_interest], kind='cubic')
        self.df[column_of_interest] = f2(self.df['rownum'])
        return self.df
    
    def make_interpolations(self, column_of_interest):
        # 4. Linear Interpolation ------------------
        self.df['rownum'] = np.arange(self.df.shape[0])  # df.shape[0]-gives number of row count
        df_nona = self.df.dropna(subset=[column_of_interest])  # df.dropna- Remove missing values.
        f = interp1d(df_nona['rownum'], df_nona[column_of_interest])
        self.df[column_of_interest] = f(self.df['rownum'])
        # 5. Cubic Interpolation --------------------
        f2 = interp1d(df_nona['rownum'], df_nona[column_of_interest], kind='cubic')
        self.df[column_of_interest] = f2(self.df['rownum'])

    # def get_liner_data(self):
    #     return self.df

    # def draw_all(self, column_of_interest):
    #     self.df_ffill = self.int_df_ffill() # df.ffill-pandas func to forward fill missing values
    #     self.df_bfill = self.int_df_bfill() # df.ffill-pandas func to backward fill missing values
    #     self.make_interpolations(column_of_interest)

    #     fig, axes = plt.subplots(5, 1, sharex=True, figsize=(20, 20))
    #     plt.rcParams.update({'xtick.bottom': False})
    #     error = 0
        
    #     # 1. Actual -------------------------------
    #     self.df[column_of_interest].plot(title='Actual', ax=axes[0], label='Actual', color='green', style=".-")
       
    #     # 2. Forward Fill --------------------------
    #     self.df_ffill[column_of_interest].plot(title='Forward Fill (MSE: ' + str(error) + ")", ax=axes[1],
    #                                             label='Forward Fill', style=".-")
        
    #     #3. Backward Fill -------------------------
    #     self.df_bfill[column_of_interest].plot(title="Backward Fill (MSE: " + str(error) + ")", ax=axes[2],
    #                                             label='Back Fill',
    #                                            color='purple', style=".-")

    #     # # # 4. Linear Interpolation ------------------
    #     self.df[column_of_interest].plot(title="Linear Fill (MSE: " + str(error) + ")", ax=axes[3], label='Cubic Fill',
    #                                  color='red',
    #                                 style=".-")

    #     # # 5. Cubic Interpolation --------------------
    #     self.df[column_of_interest].plot(title="Cubic Fill (MSE: " + str(error) + ")", ax=axes[4], label='Cubic Fill',
    #                               color='deeppink',
    #                                style=".-")
    #     st.pyplot(fig)         