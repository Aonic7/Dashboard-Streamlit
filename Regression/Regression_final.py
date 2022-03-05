# Import libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
# import statsmodels.api as sm
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as MSE
import streamlit as st #streamlit backend

# Class variables: Y_test, Y_pred,reg, X_train,Y_train

# Methods:
#          model() -- Splits the data into train and test dataset, fits the training data to the model and uses RandomForestRegressor to get the prediction for 
#                     testing data using the best parameter values obtained by performing grid search optimisation 
#          result() -- To get the Root mean squared error(RMSE) and R-squared score for the model
#          prediction_plot() -- To display the 2D-plot for the actual vs predicted values
#          gridSearchCV()  -- For hyperparameter tuning 

# User inputs:
#          estimator -- a paramter in the RandomForestRegressor which denotes the number of trees (Range : 10 to 1000 )
#          test_size -- proportion of the original dataset to be included in the test split  (Range: 0% to 100 % )

# Outputs:
#          Root mean squared error(RMSE)
#          R-squared score
#          2-D plot : Optimal prediction vs real prediction

class Regressor:

    Y_test=None
    Y_pred=None
    reg=None
    X_train=None
    Y_train=None

    # instance attribute
    def __init__(self,X,Y):
        self.X = X
        self.Y = Y

    def model(self,estimator, test_size):
        self.estimator=estimator 
        self.test_size=test_size  
        # Split the data into training set and testing set
        X_train,X_test,Y_train,self.Y_test=train_test_split(self.X,self.Y,test_size=test_size,random_state=123)      
        
        # Create a model
        reg=RandomForestRegressor(n_estimators=estimator)

        # Fitting training data to the model
        reg.fit(X_train,Y_train)

        self.Y_pred = reg.predict(X_test)
        b=[]
        b=self.gridSearchCV(reg, X_train, Y_train)
        # re-create the model
        reg=RandomForestRegressor(n_estimators=estimator,max_depth=b[0],min_samples_split=b[1],min_samples_leaf=b[2],max_features=b[3])

        # Fitting training data to the model
        reg.fit(X_train,Y_train)
        self.Y_pred = reg.predict(X_test)
        return(self.Y_pred)

    def result(self,Y_test, Y_pred): 
        # To compute R-squared score+
        r2 = r2_score(Y_test, Y_pred)
        st.metric('R-squared score: ', round(r2, 4))
        # To compute root mean squared error
        st.metric('RMSE: ', round((MSE(Y_test,Y_pred)**(0.5)), 4))
        # To compute adjusted R-squared error 
        # r_adj = 1 - ((1-r2)*((Y_test.shape[0])-1))/(Y_test.shape[0]-X_test.shape[1]-1)
        # print('R-squared adjusted:',r_adj)
    
    def prediction_plot(self,Y_test, Y_pred):
        # To display the 2D-plot for the actual vs predicted values
        df=pd.DataFrame({'y_test':Y_test,'y_pred':Y_pred})
        fig = plt.figure(figsize=(10, 4))
        sns.scatterplot(x='y_test',y='y_pred',data=df,label='Real Prediction')
        sns.lineplot(x='y_test',y='y_test',data=df,color='red',alpha=0.5,label='Optimal Prediction')
        plt.title('y_test vs y_pred')
        plt.legend()
        st.pyplot(fig)

    def gridSearchCV(self,reg,X_train, y_train):
        # Find the best parameters for the model using Grid search optimisation
        parameters = {
       'max_depth': [70, 80, 90, 100],
       'min_samples_split':[2,5,10],
       'min_samples_leaf':[1,2,4],
       'max_features': ["auto", "sqrt", "log2"],
        }
        gridforest = GridSearchCV(reg, parameters, cv = 3, n_jobs = -1, verbose = 1)
        gridforest.fit(X_train, y_train)
        best = gridforest.best_params_
        #Storing the best parameter values to pass as paramters in the function 'model'
        a=best['max_depth']
        b=best['min_samples_split']
        c=best['min_samples_leaf']
        d=best['max_features']
        return(a,b,c,d)
       

# Read data and converting it to dataframe
# df=pd.read_csv("Y:/MAIT/Object Oriented Programming - Wolf/Data for the Software Development Project/Regression - Dataset/Regression/SynchronousMachine.csv")
# df.rename(columns = {'I_y':'Load Current', 'PF':'Power Factor',
#                               'e_PF':'Power Factor Error','d_if':'Excitation Current Change','I_f':'Excitation Current'}, inplace = True)


# # To initialize the features and target 
# X=df.iloc[:,:-1]
# Y= df.iloc[:,-1]

# # object for the class Regressor
# obj = Regressor(X,Y)

# # calling the methods using object 'obj'
# obj.model(900,0.3)
# obj.result(obj.Y_test,obj.Y_pred)
# obj.prediction_plot(obj.Y_test,obj.Y_pred)



























# df.iloc[:,[3,4]].plot(use_index=True, figsize=(32,8), title='Samples vs Excitation Current Change & Excitation Current')
# df.iloc[:,[2,4]].plot(use_index=True, figsize=(32,8), title='Samples vs Power Factor Error & Excitation Current')
# df.iloc[:,[0,4]].plot(use_index=True, figsize=(32,8), title='Samples vs Load Current & Excitation Current')
# df.iloc[:,[1,4]].plot(use_index=True, figsize=(32,8), title='Samples vs Power Factor & Excitation Current')

# sns.scatterplot(x='Load Current',y='Excitation Current',data=df)
# plt.show()

# sns.scatterplot(x='Power Factor',y='Excitation Current',data=df)
# plt.show()





# # Find the best parameters for the model
# parameters = {
#     'max_depth': [70, 80, 90, 100],
#     'n_estimators': [900, 1000, 1100]
# }
# gridforest = GridSearchCV(model, parameters, cv = 3, n_jobs = -1, verbose = 1)
# gridforest.fit(X_train, y_train)
# a= gridforest.best_params_
# print(a)



