# # Data Visualization
# Import necessary libraries.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as MSE
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV
from typing import NamedTuple
import math

import streamlit as st


# User Inputs:
                # Syscodenumber
                # Date
                # Time

# Output:
                # R2 -score
                # RMSE
                # Plot of predicted and optimal Value
                # Occupancy Rate 
                # Occupancy for a particular systemcodenumber at a given time and date

# Methods():
                # DataPrep(): Occupancy rate and binning for the minute column obtained from the given time
                # fullmodel(): Initialization of features and target for the whole dataset
                # model(): fitting and prediction on the whole training and test dataset
                # SelectedSysCode(): Initialization of features and target on the basis of filtered dataframe accordingly to the given systemcodenumber 
                # UserSelectmodel():
                # Results(): To display the R2 score, RMSE and the prediction plot

class Timeseries:
    
    # instance attribute
    def __init__(self,df,Input,base,target,time_col,sysCodeNo):
        self.df=df
        self.Input = Input
        self.base=base
        self.target=target
        self.time_col=time_col
        self.sysCodeNo=sysCodeNo

    #
    def model(self,estimator, test_size):
        self.estimator=estimator 
        self.test_size=test_size  
        # Split the data into training set and testing set
        X_train,X_test,Y_train,self.Y_test=train_test_split(self.X,self.Y,test_size=test_size,random_state=123)

        # Create a model
        reg=RandomForestRegressor(n_estimators=estimator)

        c = self.RandomizedSearchoptim()
        # st.write(c)

        rf_reg=RandomizedSearchCV(estimator= reg ,param_distributions = c, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

        # Fitting training data to the model
        rf_reg.fit(X_train,Y_train)
        self.Y_pred = rf_reg.predict(X_test)

        # st.write(self.Y_test, self.Y_pred)
        return(self.Y_pred,self.Y_test)

        
    def UserSelectedmodel(self,estimator, test_size):
        
        self.estimator=estimator 
        self.test_size=test_size  
        # Split the data into training set and testing set
        X_train1,self.X_test1,Y_train1,self.Y_test1 = train_test_split(self.X1,self.Y1,test_size=test_size,random_state=123)
        # st.write(X_train1,Y_train1)

        # Create a model
        reg=RandomForestRegressor(n_estimators=estimator)
        b = self.RandomizedSearchoptim()
        
        rf_reg=RandomizedSearchCV(estimator= reg ,param_distributions = b, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

        # Fitting training data to the model
        rf_reg.fit(X_train1,Y_train1)
        self.Y_pred1 = rf_reg.predict(self.X_testIP)
        # self.Y_pred1 = round(self.Y_pred1,0)
        
        st.write( 'Rate: ', self.Y_pred1, '%')
        # df.to_excel('Filtered1.xlsx', sheet_name='Filtered Data')

        Total = self.df[self.base].unique()
        o = (Total * self.Y_pred1)/100
        st.write('Original value: ', math.floor(o))

    def RandomizedSearchoptim(self):
        # Find the best parameters for the model
        max_features = ['auto', 'sqrt']                                                  # Number of features to consider at every split
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]                     # Maximum number of levels in tree
        max_depth.append(None)
        min_samples_split = [2, 5, 10]                                                   # Minimum number of samples required to split a node
        min_samples_leaf = [1, 2, 4]                                                     # Minimum number of samples required at each leaf node
        bootstrap = [True, False]                                                        # Method of selecting samples for training each tree
       
        # Create the random grid
        random_grid = {'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'bootstrap': bootstrap
        }
        return random_grid

    def DataPrep(self):
       
       # Convert LastUpdated column into Time and Date column. Add new columns for occupancy rate in percentage and Day of Week.
        self.df['Rate'] = round((100.0*self.df[self.target])/self.df[self.base],1)
        nxm=np.shape(self.df)
        n=nxm[0]
        m=nxm[1]
        date_time_column=self.df[self.time_col]
        X=np.ndarray(shape=(n,6),dtype=float, order='F')
        
        for a in range(0,n):

            date_time_obj = date_time_column.iloc[a]
            year=date_time_obj.year
            month=date_time_obj.month
            day=date_time_obj.day
            hour=date_time_obj.hour
            minute=date_time_obj.minute
            second=date_time_obj.second
            X[a,0]=year
            X[a,1]=month
            X[a,2]=day
            X[a,3]=hour
            X[a,4]=minute
            X[a,5]=second

        df_internal=pd.DataFrame(data=X,columns=["year","month","day", "hour","minute","second"])
        self.df.reset_index(drop=True, inplace=True)
        df_internal.reset_index(drop=True, inplace=True)
        self.df=pd.concat([self.df,df_internal],axis=1)
        

        #Categorizing the occupancy rate: Bining
        bins = [0,14,29,44,59]
        labels =[1,2,3,4]
        self.df['newmin'] = pd.cut(self.df['minute'], bins,labels=labels,include_lowest= True)

    def fullmodel(self):
       
       # To initialize the features and self.target 
        self.X = self.df.loc[:,[self.base,'year','month','day','hour','newmin']]
        #self.X.to_csv('DatasetYY2.csv')
        self.Y= self.df['Rate'] 
            
    
    def SelectedSysCode(self,Input):
            # Grouping the minute column into two bins(one being 0-30 mins and other being 30-60 mins)
            if((Input.Time.minute >= 0) & (Input.Time.minute < 15)):
                newmin = 1
            elif((Input.Time.minute >= 15) & (Input.Time.minute < 30)):
                newmin = 2
            elif((Input.Time.minute >= 30 ) & (Input.Time.minute < 45)):
                newmin = 3
            else:
                newmin = 4       

            self.df = self.df[self.df[self.sysCodeNo] == Input.Syscodenumber] #Filtering dataframe according to user input
            
            self.df.drop(self.df.filter(regex="Unname"),axis=1, inplace=True)           
            
            self.X_testIP = pd.DataFrame(columns=['year','month','day','hour','newmin'])
            self.X_testIP.loc[0, ['year']]     = Input.Date.year 
            self.X_testIP.loc[0, ['month']]    = Input.Date.month
            self.X_testIP.loc[0, ['day']]      = Input.Date.day
            self.X_testIP.loc[0, ['hour']]     = Input.Time.hour
            self.X_testIP.loc[0, ['newmin']]   = newmin

            # To initialize the features and self.target 
            self.X1 = self.df.loc[:,['year','month','day','hour','newmin']]

            self.Y1= self.df['Rate']
     
    
    def Results (self):
        
        # R-Square value
        r2 = r2_score(self.Y_test, self.Y_pred)
        st.write('R-squared score of Predicted model: ', round(r2, 5))
        st.write('RMSE of Predicted model: ',MSE(self.Y_test,self.Y_pred)**(0.5))
        
        # To display the 2D-plot for the actual vs predicted values
        df1=pd.DataFrame({'Y_Actual':self.Y_test,'Y_Pred':self.Y_pred})
        fig = plt.figure(figsize=(10, 4))
        sns.scatterplot(x='Y_Actual',y='Y_Pred',data=df1,label='Real Prediction')
        sns.lineplot(x='Y_Actual',y='Y_Actual',data=df1,color='red',alpha=0.5,label='Optimal Prediction')
        plt.title('Y_Actual vs Y_Pred')
        plt.legend()
        st.pyplot(fig)
    
        

class rf_Inputs(NamedTuple):
    Syscodenumber:  str
    Date:           datetime
    Time:           datetime


# Import data setsss 
# df = pd.read_csv(r"C:\Users\Yash Uday Pisat\Desktop\dataset.csv")

# # st.write('Before removing inconsistence data:',df.shape)
# df.dropna(inplace = True)
# df.drop_duplicates(keep='first',inplace=True) 

# df = df[df['Occupancy']>=0 ]
# df = df[df['Capacity']>=0 ]
# false_data = df[df['Occupancy']> df['Capacity']]
# df = pd.concat([df, false_data]).drop_duplicates(keep=False)
# # st.write('After removing inconsistence data:',df.shape)


# dat=datetime(2016,10,10)
# s=datetime(dat.year,dat.month,dat.day,15,15)


# Input= rf_Inputs('Shopping',dat,s)

# df['LastUpdated']=pd.to_datetime(df['LastUpdated'])

"'Capacity' = base column"
"'Occupancy' = target column"

# obj= Timeseries(df,Input,'Capacity','Occupancy','LastUpdated','SystemCodeNumber') 

# obj.DataPrep()
# obj.fullmodel()
# obj.model(500, 0.3)
# obj.Results()
# obj.SelectedSysCode(Input)
# obj.UserSelectedmodel(500, 0.3)

