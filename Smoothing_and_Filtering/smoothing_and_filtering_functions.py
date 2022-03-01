# General import section
import pandas as pd #to work with dataframes
import streamlit as st #streamlit backend
import numpy as np #to work with arrays and series
import scipy.stats as stats #to work with z-score

from scipy.signal import medfilt, savgol_filter

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