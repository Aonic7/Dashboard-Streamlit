# # Generate dataset
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#accesing the file
df = pd.read_csv(r'C:\Users\Admin\OOP\TimeSeries\dataset.csv', parse_dates=['LastUpdated'], index_col='LastUpdated')

#defining class "TimeSeriesOOP"
class TimeSeriesOOP:
    def __init__(self, path_to_csv, date_column=None, index_column=None):
        self.df = self.csv_reader(path_to_csv, date_column=date_column, index_column=index_column)
        self.process_dataframe( )
        self.df_ffill = self.df.ffill( )  # df.ffill-pandas func to forward fill missing values
        self.df_bfill = self.df.bfill( )  # df.ffill-pandas func to backward fill missing values

    @staticmethod 
    def csv_reader(path_to_csv, date_column=None, index_column=None):
        dataframe = pd.read_csv(path_to_csv, parse_dates=[date_column],
                                index_col=index_column)
        return dataframe

    def process_dataframe(self):  # make separate func if you need more processing
        self.df = self.df.resample('15min').mean( )

    def make_interpolations(self, column_of_interest):
        # 4. Linear Interpolation ------------------
        self.df['rownum'] = np.arange(self.df.shape[0])  # df.shape[0]-gives number of row count
        df_nona = self.df.dropna(subset=[column_of_interest])  # df.dropna- Remove missing values.
        f = interp1d(df_nona['rownum'], df_nona[column_of_interest])
        self.df['linear_fill'] = f(self.df['rownum'])

        # 5. Cubic Interpolation --------------------
        f2 = interp1d(df_nona['rownum'], df_nona[column_of_interest], kind='cubic')
        self.df['cubic_fill'] = f2(self.df['rownum'])

    def draw_all(self, column_of_interest):
        self.make_interpolations(column_of_interest=column_of_interest)

        fig, axes = plt.subplots(5, 1, sharex=True, figsize=(20, 20))
        plt.rcParams.update({'xtick.bottom': False})
        error = 0

        # 1. Actual -------------------------------
        self.df[column_of_interest].plot(title='Actual', ax=axes[0], label='Actual', color='green', style=".-")

        # 2. Forward Fill --------------------------
        self.df_ffill[column_of_interest].plot(title='Forward Fill (MSE: ' + str(error) + ")", ax=axes[1],
                                               label='Forward Fill', style=".-")

        # 3. Backward Fill -------------------------
        self.df_bfill[column_of_interest].plot(title="Backward Fill (MSE: " + str(error) + ")", ax=axes[2],
                                               label='Back Fill',
                                               color='purple', style=".-")

        # 4. Linear Interpolation ------------------
        self.df['linear_fill'].plot(title="Linear Fill (MSE: " + str(error) + ")", ax=axes[3], label='Cubic Fill',
                                    color='red',
                                    style=".-")

        # 5. Cubic Interpolation --------------------
        self.df['cubic_fill'].plot(title="Cubic Fill (MSE: " + str(error) + ")", ax=axes[4], label='Cubic Fill',
                                   color='deeppink',
                                   style=".-")
        plt.show( )

#using the class
time_series_visualiser = TimeSeriesOOP(r'C:\Users\Admin\OOP\TimeSeries\dataset.csv',
date_column='LastUpdated', index_column='LastUpdated')
col_of_interest = 'Occupancy'
time_series_visualiser.draw_all(column_of_interest=col_of_interest)       