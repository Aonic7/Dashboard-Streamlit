# # Generate dataset
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

#accesing the file
df = pd.read_csv(r'C:\Users\shaha\Desktop\MAIT 1st Sem\OOP\Codes\VS_codes\dataset.csv', parse_dates=['LastUpdated'], index_col='LastUpdated')

#defining class "TimeSeriesOOP"
class TimeSeriesOOP:

    def __init__(self, path_to_csv, date_column=None, index_column=None): #constructor definition
        self.df = self.csv_reader(path_to_csv, date_column=date_column, index_column=index_column)
        self.process_dataframe()
        
    @staticmethod 
    def csv_reader(path_to_csv, date_column=None, index_column=None):
        dataframe = pd.read_csv(path_to_csv, parse_dates=[date_column],
                                index_col=index_column)
        return dataframe

    def process_dataframe(self):  # make separate func if you need more processing
        self.df = self.df.groupby([col_of_code, col_of_capacity]).resample('15min')[col_of_interest].mean().reset_index() #grouping by the system code and capacity, resampling the the LastUpdated and getting Nan values for Occupancy
        self.df = self.df.iloc[:, [0, 1, 3, 2]]    
        
        
    # def  make_interpolation_forward(self, col_of_interest):  
        
    #     self.df_ffill = self.df[[col_of_interest]].ffill()
    #     self.df[col_of_interest] = self.df_ffill[col_of_interest]
    #     self.df = self.df.set_index(col_of_code)
    #     self.df.to_csv('C:/Users/shaha/Desktop/MAIT 1st Sem/OOP/Codes/Interpolation/forward_fill.csv')

    # def  make_interpolation_backward(self, col_of_interest):

    #     self.df_bfill = self.df[[col_of_interest]].bfill()
    #     self.df[col_of_interest] = self.df_bfill[col_of_interest]
    #     self.df = self.df.set_index(col_of_code)
    #     self.df.to_csv('C:/Users/shaha/Desktop/MAIT 1st Sem/OOP/Codes/Interpolation/backward_fill.csv')
    

    # def  make_interpolation_linear(self, col_of_interest):
    
    #     self.df_linear_fill = self.df[[col_of_interest]].interpolate(method='linear')
    #     self.df[col_of_interest] = self.df_linear_fill[col_of_interest]
    #     self.df = self.df.set_index(col_of_code)
    #     self.df.to_csv('C:/Users/shaha/Desktop/MAIT 1st Sem/OOP/Codes/Interpolation/linear.csv')
        

    def  make_interpolation_cubic(self, col_of_interest):
    
        self.df_cubic_fill = self.df[[col_of_interest]].interpolate(method='cubic')
        self.df_cubic_fill[self.df_cubic_fill[[col_of_interest]] < 0] = 0 #converting negative values to zero
        self.df[col_of_interest] = self.df_cubic_fill[col_of_interest]
        self.df = self.df.set_index(col_of_code)
        self.df.to_csv('C:/Users/shaha/Desktop/MAIT 1st Sem/OOP/Codes/Interpolation/cubic.csv')
           

    # def draw_all(self, col_of_interest):
    #     self.make_interpolations(col_of_interest=col_of_interest)

    #     fig, axes = plt.subplots(5, 1, sharex=True, figsize=(50,50))
    #     plt.rcParams.update({'xtick.bottom': False})
    #     error = 0

    # # 1. Actual -------------------------------
    #     self.df[col_of_interest].plot(title='Actual', ax=axes[0], label='Actual', linewidth=0.5, color='green', style=".-")

    # # 2. Forward Fill --------------------------
    #     self.df_ffill[col_of_interest].plot(title='Forward Fill (MSE: ' + str(error) + ")", ax=axes[1],
    #                                            label='Forward Fill', style=".-")

    # # 3. Backward Fill -------------------------
    #     self.df_bfill[[col_of_interest]].plot(title="Backward Fill (MSE: " + str(error) + ")", ax=axes[2],
    #                                            label='Back Fill',
    #                                            color='purple', style=".-")

    # #4. Linear Interpolation ------------------
    #     self.df_linear_fill.plot(title="Linear Fill (MSE: " + str(error) + ")", ax=axes[3], label='Cubic Fill',
    #                                 color='red',
    #                                 style=".-")

    # # 5. Cubic Interpolation --------------------
    #     self.df_cubic_fill.plot(title="Cubic Fill (MSE: " + str(error) + ")", ax=axes[4], label='Cubic Fill',
    #                                color='deeppink',
    #                                style=".-")
    #     plt.show( ) 
    

#using the class
col_of_interest = 'Occupancy'
col_of_code = 'SystemCodeNumber'
col_of_capacity = 'Capacity'

time_series_visualiser = TimeSeriesOOP(r'C:\Users\shaha\Desktop\MAIT 1st Sem\OOP\Codes\VS_codes\dataset.csv',
date_column='LastUpdated', index_column='LastUpdated')


#time_series_visualiser.make_interpolation_forward(col_of_interest=col_of_interest)
#time_series_visualiser.make_interpolation_backward(col_of_interest=col_of_interest)
#time_series_visualiser.make_interpolation_linear(col_of_interest=col_of_interest)
time_series_visualiser.make_interpolation_cubic(col_of_interest=col_of_interest)
      