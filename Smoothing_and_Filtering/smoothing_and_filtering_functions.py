# General import section
import pandas as pd #to work with dataframes
import numpy as np #to work with arrays and series
import scipy.stats as stats #to work with z-score

from scipy.signal import medfilt, savgol_filter
class Remove_Outliers():
    """
    This class is used to provide methods for removing outliers from a given dataset.
    """ 
    def __init__(self):
        pass

    # Removing outliers using a standard deviation
    def removeOutlier(df, columnName, n):
        """
        RemoveOutlier - method based on mean and standard deviation. 
        The function determines the mean and standard deviation of the value in a given column. Further, following the formula, 
        it determines the boundary values and removes outliers.
        Remove outliers using a coefficient of standard deviation (n) in range from 1.0 to 3.0.
        
        :param df (pandas.core.frame.DataFrame): input dataframe
        :param columnName (str): a column of the input dataframe
        :param n (float): standard deviation coefficient    
        :return pandas.core.frame.DataFrame: filtered dataframe   
        """
        mean = df[columnName].mean()
        std = df[columnName].std()  
        fromVal = mean - n * std 
        toVal = mean + n * std 
        filtered = df[(df[columnName] >= fromVal) & (df[columnName] <= toVal)] #apply the filtering formula to the column
        return filtered

    # Removing outliers using quantile range
    def removeOutlier_q(df, columnName, n1, n2):
        """
        removeOutlier_q - method based on the removal of data that lie below the lower quantile and above the upper quantile.
        Remove outliers using a quantile range from 0.0 to 1.0
        
        :param df (pandas.core.frame.DataFrame): input dataframe
        :param columnName (str): a column of the input dataframe
        :param n1 (float): lower quantile
        :param n2 (float): upper quantile
        :return pandas.core.frame.DataFrame: filtered dataframe
        """
        lower_quantile, upper_quantile = df[columnName].quantile([n1, n2]) #quantiles are generally expressed as a fraction (from 0 to 1)
        filtered = df[(df[columnName] > lower_quantile) & (df[columnName] < upper_quantile)]
        return filtered

    # Removing outliers using z-score
    def removeOutlier_z(df, columnName, n):
        """
        removeOutlier_z - method based on Z-score. The z-score define the number of standard deviations above and below the mean.

        :param df (pandas.core.frame.DataFrame): input dataframe
        :param columnName (str): a column of the input dataframe
        :param n (float): standard deviation coefficient
        :return pandas.core.frame.DataFrame: filtered dataframe
        """
        z = np.abs(stats.zscore(df[columnName])) #find the Z-score for the column
        filtered = df[(z < n)] #apply the filtering formula to the column
        return filtered

# Timeseries converter
class Converter():
    """
    Class checks columns of dataset for datetime data and then converts it to the datetime datatype
    """
   
    def dateTime_converter(df):
        """Timeseries converter

        :param df: input dataframe
        :type df: pandas.core.frame.DataFrame
        :return: converted dataframe
        :rtype: pandas.core.frame.DataFrame
        """

        for col in df.columns:
            # Checks if the column is a object 
            if df[col].dtype == 'object':
                try:
                    # Converting suitable columns into Datetime format 
                    df[col] = pd.to_datetime(df[col])
                except:
                    pass
            else:
                df[col] = df[col]
        return df


class Smoothing():
    """
    This class is used to provide methods for smoothing a given dataset.
    """ 
    def __init__(self):
        pass
    

    def median_filter(df, columnName, filter_length):
        """Data smoothing using median filter function

        :param df: input dataframe
        :type df: pandas.core.frame.DataFrame
        :param columnName: a column of the input dataframe
        :type columnName: str
        :param filter_length: filter length parameter
        :type filter_length: int
        :return: smoothed dataframe (median filter)
        :rtype: pandas.core.frame.DataFrame
        """
        
        df_var = df.copy()
        medfilt_tair = medfilt(df_var[columnName], filter_length)
        df_var[columnName] = medfilt_tair
        return df_var

    def moving_average(df, columnName, filter_length):
        """Data smoothing using moving average function

        :param df: input dataframe
        :type df: pandas.core.frame.DataFrame
        :param columnName: a column of the input dataframe
        :type columnName: str
        :param filter_length: filter length parameter
        :type filter_length: int
        :return: smoothed dataframe (moving average)
        :rtype: pandas.core.frame.DataFrame
        """
        df_var = df.copy()
        df_var[columnName] = df[columnName].rolling(filter_length).mean()
        df_var.dropna(inplace=True)
        df_var.reset_index(drop=True)

        return df_var

    def savitzky_golay(df, columnName, filter_length, order):
        """Data smoothing using savitzky-golay function

        :param df: input dataframe
        :type df: pandas.core.frame.DataFrame
        :param columnName: a column of the input dataframe
        :type columnName: str
        :param filter_length: filter length parameter 
        :type filter_length: int
        :param order: order of the polynomial to fit the function
        :type order: int
        :return: smoothed dataframe (savitzky-golay)
        :rtype: pandas.core.frame.DataFrame
        """
        df_var = df.copy()
        df_var[columnName] = savgol_filter(df_var[columnName], filter_length, order)
        df_var.dropna(inplace=True)
        df_var.reset_index(drop=True)

        return df_var


# Interpolation class 
class TimeSeriesOOP():
    """
    This class is used to provide methods for interpolation of a given dataset with time-series.
    Interpolation is a method of finding new data points based on the range of a discrete set of known data points.
    """ 
  
    def __init__(self, current_df, column_of_interest, time_column, col_group):
        """Dataframe initialization 

        :param current_df: input dataframe
        :type current_df: pandas.core.frame.DataFrame
        :param column_of_interest: a column of the input dataframe
        :type column_of_interest: str
        :param time_column: a column of the input dataframe 
        :type time_column: DateTime
        :param col_group: list of columns for groupping
        :type col_group: list
        """
        current_df=current_df.set_index("{}".format(time_column))
        self.df = current_df
        # Resampling the datframe 
        self.process_dataframe(col_group, column_of_interest)

    # Resampling the dataframe     
    def process_dataframe(self, col_group, column_of_interest):
        """Resampling the dataframe

        :param col_group: list of columns for groupping
        :type col_group: list
        :param column_of_interest: column selected for processing
        :type column_of_interest: str
        """
        #self.df = self.df.resample('15min').mean( )
        self.df = self.df.groupby(col_group).resample('15min')[column_of_interest].mean().reset_index()
        self.df = self.df.iloc[:, [0, 1, 3, 2]] 

    # Forward fill    
    def int_df_ffill(self, column_of_interest, col_group):
        """Forward fill - It fills missing values with previous data. Forward-fill propagates the last observed 
        non-null value forward until another non-null value is encountered.

        :param column_of_interest: column selected for processing
        :type column_of_interest: str
        :param col_group: list of columns for groupping
        :type col_group: list
        :return: processed dataframe
        :rtype: pandas.core.frame.DataFrame
        """
        self.df_ffill = self.df[[column_of_interest]].ffill() #df.ffill-pandas func to forward fill missing values
        self.df[column_of_interest] = self.df_ffill[column_of_interest]
        self.df = self.df.set_index(col_group[0])

        return self.df 
    
    # Backward fill  
    def int_df_bfill(self, column_of_interest, col_group):
        """Backward fill - Backward filling means fill missing values with next data point. backward-fill propagates 
        the first observed non-null value backward until another non-null value is met. 

        :param column_of_interest: column selected for processing
        :type column_of_interest: str
        :param col_group: list of columns for groupping
        :type col_group: list
        :return: processed dataframe
        :rtype: pandas.core.frame.DataFrame
        """
        self.df_bfill = self.df[[column_of_interest]].bfill()  #df.bfill-pandas func to backward fill missing values
        self.df[column_of_interest] = self.df_bfill[column_of_interest]
        self.df = self.df.set_index(col_group[0])
        return self.df  
    
    # Linear Interpolation 
    def make_interpolation_liner(self, column_of_interest, col_group):
        """Linear Interpolation - In this, the points are simply joined by straight line segments. Each segment 
        (bounded by two data points) can be interpolated independently.

        :param column_of_interest: column selected for processing
        :type column_of_interest: str
        :param col_group: list of columns for groupping
        :type col_group: list
        :return: processed dataframe
        :rtype: pandas.core.frame.DataFrame
        """

        self.df_linear_fill = self.df[[column_of_interest]].interpolate(method='linear')
        self.df[column_of_interest] = self.df_linear_fill[column_of_interest]
        self.df = self.df.set_index(col_group[0])
        
        return self.df
    
    # Cubic Interpolation
    def make_interpolation_cubic(self, column_of_interest, col_group):
        """Cubic interpolation offers true continuity between the segments. As such it 
        requires more than just the two endpoints of the segment but also the two points on either side of them.

        :param column_of_interest: column selected for processing
        :type column_of_interest: str
        :param col_group: list of columns for groupping
        :type col_group: list
        :return: processed dataframe
        :rtype: pandas.core.frame.DataFrame
        """

        self.df_cubic_fill = self.df[[column_of_interest]].interpolate(method='cubic')
        self.df_cubic_fill[self.df_cubic_fill[[column_of_interest]] < 0] = 0 #converting negative values to zero
        self.df[column_of_interest] = self.df_cubic_fill[column_of_interest]
        self.df = self.df.set_index(col_group[0])

        return self.df