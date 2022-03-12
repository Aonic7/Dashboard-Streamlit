import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from typing import NamedTuple
from numpy.core.fromnumeric import shape
import streamlit as st #streamlit backend


class NN_Regressor:

    y_pred:                     float # resulting output
    y_actual:                   float # expected output
    length:                     int   # length of y_test
    mean_squared_error:         float # mean square error
    Train_score:                float # Model Score (R^2) on the Training data  
    X_test:                     float
    test_score:                 float # Model Score (R^2) on the Testing data
    model:                      MLPRegressor # 
    Error_message:              str
    flag:                       bool

    #coeff:                     tuple

    dependant_var_index =0
    #NN_Inputs = Regressor_Inputs
    # = Regressor_Outputs
    flag=False
    Error_message='No Error occurred in the processing'
    
    #Constructor
    #External inputs are the Data frame object, the Named Tuple of NN_Inputs and the index of the dependant variable in the Data frame
    def __init__(self,df,NN_Inputs,dependant_var_index):
        self.df=df
        self.NN_Inputs=NN_Inputs
        
        self.k=dependant_var_index
        self.handle()
        #After initial Handling of the data we check to see if the user wants to normalize the X values prior to training the model
        if self.NN_Inputs.Normalize:
            self.preprocess()
        elif self.flag!=True:
            #if Normalization is not checked, the X values fed into the regressor are the same as the output from the initial handle method
            self.x_n=self.X
    
    def preprocess(self):
        #Simple Normalization method for X Data
        scaler=preprocessing.MinMaxScaler()
        self.x_n = scaler.fit_transform(self.X)

    #Data handling method, creates the X and Y arrays that go into the Train_test_split method
    #Called when the object is instantiated within the constructor 
    def handle(self):
        
        try:   
            self.internal_data =self.df.drop(self.df.iloc[:,[self.k]],axis=1)
            nxm=shape(self.internal_data)
            n=nxm[0]
            m=nxm[1]
      
            self.X=np.ndarray(shape=(n,m),dtype=float, order='F')
            

            for l in range(0,m):  
            
                for i in range(0,n):
                
                    self.X[i,l]=self.internal_data.iloc[i,l]
            
            Y=np.ndarray(shape=(n,1), dtype=float, order='F')
            Y[:,0]=self.df.iloc[:,self.k]

            self.y=np.ravel(Y)
        except Exception as e:
            self.Error_message='Error in Handling Method: ' + str(e)
            self.flag=True
        
       
    #Method that creates the MLP Regressor and returns the Named Tuple of to be used in other methods  
    def Regressor(self):
        
        if (self.flag) !=True:
            try:
                X_train, self.X_test, y_train, self.y_actual= train_test_split(self.x_n,self.y,test_size=self.NN_Inputs.test_size,shuffle=False, random_state=40)
                self.model = MLPRegressor(hidden_layer_sizes = self.NN_Inputs.hidden_layers, 
                                    activation = self.NN_Inputs.activation_fun, solver = self.NN_Inputs.solver_fun, 
                                    learning_rate = 'adaptive', max_iter = self.NN_Inputs.Max_iterations, random_state = 109,shuffle=True )


                self.model.fit(X_train,y_train)
                self.Train_score= self.model.score(X_train,y_train)
                self.test_score= self.model.score(self.X_test,self.y_actual)
                #self.coeff=self.model.coefs_

                
                self.y_pred = self.model.predict(self.X_test)

                y_pred = np.ndarray.tolist(self.y_pred)
                self.length = len(y_pred)
                
                #Mean squared error and accuracy
                self.mean_squared_error = mean_squared_error(self.y_actual, 
                self.y_pred)
                
            except Exception as e:
                self.Error_message= 'Error in Regressor Creation: ' + str(e)
                self.flag=True
                st.warning(self.Error_message)

                self.Train_score= 'Refer To error in Regressor Creation'
                self.test_score= 'Refer To error in Regressor Creation'
                #self.coeff=self.model.coefs_

                self.y_actual='Refer To error in Regressor Creation'
                self.y_pred = 'Refer To error in Regressor Creation'

                self.y_pred = 'Refer To error in Regressor Creation'
                self.length = 'Refer To error in Regressor Creation'
                
                #Mean squared error and accuracy
                self.mean_squared_error = 'Refer To error in Regressor Creation'
        else:
            st.warning(self.Error_message)

            self.Train_score= 'Refer To error in Handling Method'
            self.test_score= 'Refer To error in Handling Method'
            #self.coeff=self.model.coefs_

            self.y_actual='Refer To error in Handling Method'
            self.y_pred = 'Refer To error in Handling Method'

            self.y_pred = 'Refer To error in Handling Method'
            self.length = 'Refer To error in Handling Method'
            
            #Mean squared error and accuracy
            self.mean_squared_error = 'Refer To error in Handling Method'


        return self
        
    def printing(self):
        #Printing the chosen metrics
        if (self.flag) != True:
            self.Error_message= ' No Error Occurred during processing of the code'
        
        try:
            

            # print('Error Message           ', self.Error_message)
            # print('expected output:        ', self.y_actual)
            # print('Predicted Output:       ', self.y_pred)
            # print('Model Train_score on the Training Data: ',  self.Train_score)
            # print('Model Train_score on the Testing Data:  ',  self.test_score)
            # print('Mean Square error:      ',  self.mean_squared_error)
            # print('length of output array: ',  self.length)
            st.warning(self.Error_message)
            cc1, cc2 = st.columns(2)
            with cc1:
                # st.metric('Expected output:        ', self.NN_Outputs.y_actual)
                # st.write('Predicted Output:       ', self.NN_Outputs.y_pred)
                st.metric('Model score on the Training Data:',  round(self.Train_score, 8))
                st.metric('Model score on the Testing Data:',  round(self.test_score, 8))
                st.write('R-squared score (aka coefficient of determination) measures the variation that is explained by a regression model.')
                st.metric('Length of output array: ',  self.length)
            with cc2:
                #st.write('Classification Report: ')
                st.metric('RMSE: ',  round((self.mean_squared_error)**(0.5),8))
        except Exception as e:
            self.Error_message = 'Error while printing outputs: ' +str(e)
            self.flag=True
            st.warning(self.Error_message)
        #print('coeff: ',.coeff )

    def plotting(self):
        #Plotting a 2D plot of the Regression output with external input i being the field of which we want to plot as x-axis
        if self.flag != True:
            try:
                # fig = plt.figure()
                # ax1 = fig.add_subplot(111)

                # ax1.scatter(self.X_test[:,i],self.y_pred, s=10, c='r', marker="o", label='Predicted')
                # ax1.scatter(self.X_test[:,i],self.y_actual, s=10, c='b', marker="o", label='actual')
                # ax1.set_xlabel (self.df.columns[i])
                # ax1.set_ylabel  (self.df.columns[self.k])   
                # ax1.legend()
                # plt.show()
                df=pd.DataFrame({'y_test':self.y_actual,'y_pred':self.y_pred})
                fig = plt.figure(figsize=(10, 4))
                sn.scatterplot(x='y_test',y='y_pred',data=df,label='Real Prediction')
                sn.lineplot(x='y_test',y='y_test',data=df,color='red',alpha=0.5,label='Optimal Prediction')
                plt.title('Expected vs Predicted Output')
                plt.legend()
                #plt.show()
                st.pyplot(fig)
            except Exception as e:
                self.Error_message='Error in Plotting Method: ' + str(e)
                self.flag=True
                st.warning(self.Error_message)

# #Using the Class
# data = pd.read_csv("D:\MAIT\OOP\Datasets\Regression\synchronous_machine.csv",delimiter=';',decimal=',')
# #print(data.head)
# activation_fun1 = ("identity", "logistic", "tanh", "relu")
# solver_fun1 = ("lbfgs", "sgd", "adam")

# hidden_layers3=(600,300,100)

class Regressor_Inputs(NamedTuple):
        
        test_size:              float  # test size percentage
        activation_fun:         tuple  # activation function selection
        hidden_layers:          tuple  # size of hidden layer and number of neurons in each layer
        solver_fun:             tuple  # solver function
        Max_iterations:         int    # number of iterations
        Normalize:              bool   # flag to normalize X data or not

#Creating a tuple that will work with the method defined inside the class
# Inputs= Regressor_Inputs(0.2,activation_fun1[3],hidden_layers3,solver_fun1[0],500,False)
# #print(Inputs)

# Regressor1=NN_Regressor(data,Inputs,4)

# #print(Regressor1.NN_Inputs)
# #Calling the Regression method that would create the Neural Network
# Regressor1.Regressor()

# #input the index of the field that you want to occupy the x axis 
# Regressor1.plotting()
# #Printing the metrics chosen for the Regressor
# Regressor1.printing()


#print(NN_Regressor.Error_message)


#     y_pred:                     float # resulting output
#     y_actual:                   float # expected output
#     length:                     int   # length of y_test
#     mean_squared_error:         float # mean square error
#     Train_score:                float # Model Score (R^2) on the Training data  
#     X_test:                     float
#     test_score:                 float # Model Score (R^2) on the Testing data
#     model:                      MLPRegressor # 
#     Error_message:              str
#     flag:                       bool

#     #coeff:                     tuple

#     dependant_var_index =0
#     #NN_Inputs = Regressor_Inputs
#     # = Regressor_Outputs
#     flag=False
#     Error_message='No Error occurred in the processing'
    
#     #Constructor
#     #External inputs are the Data frame object, the Named Tuple of NN_Inputs and the index of the dependant variable in the Data frame
#     def __init__(self,df,NN_Inputs,dependant_var_index):
#         self.df=df
#         self.NN_Inputs=NN_Inputs
        
#         self.k=dependant_var_index
#         self.handle()
#         #After initial Handling of the data we check to see if the user wants to normalize the X values prior to training the model
#         if self.NN_Inputs.Normalize:
#             self.preprocess()
#         elif self.flag!=True:
#             #if Normalization is not checked, the X values fed into the regressor are the same as the output from the initial handle method
#             self.x_n=self.X
    
#     def preprocess(self):
#         #Simple Normalization method for X Data
#         scaler=preprocessing.MinMaxScaler()
#         self.x_n = scaler.fit_transform(self.X)

#     #Data handling method, creates the X and Y arrays that go into the Train_test_split method
#     #Called when the object is instantiated within the constructor 
#     def handle(self):
#         try:   
#             self.internal_data =self.df.drop(self.df.iloc[:,[self.k]],axis=1)
#             nxm=shape(self.internal_data)
#             n=nxm[0]
#             m=nxm[1]
      
#             self.X=np.ndarray(shape=(n,m),dtype=float, order='F')
            

#             for l in range(0,m):  
            
#                 for i in range(0,n):
                
#                     self.X[i,l]=self.internal_data.iloc[i,l]
            
#             Y=np.ndarray(shape=(n,1), dtype=float, order='F')
#             Y[:,0]=self.df.iloc[:,self.k]

#             self.y=np.ravel(Y)
#         except Exception as e:
#             self.Error_message='Error in Handling Method: ' + str(e)
#             self.flag=True
        
       
#     #Method that creates the MLP Regressor and returns the Named Tuple of to be used in other methods  
#     def Regressor(self):
        
#         if (self.flag) !=True:
#             try:
#                 X_train, self.X_test, y_train, self.y_actual= train_test_split(self.x_n,self.y,test_size=self.NN_Inputs.test_size,shuffle=False, random_state=40)
#                 self.model = MLPRegressor(hidden_layer_sizes = self.NN_Inputs.hidden_layers, 
#                                     activation = self.NN_Inputs.activation_fun, solver = self.NN_Inputs.solver_fun, 
#                                     learning_rate = 'adaptive', max_iter = self.NN_Inputs.Max_iterations, random_state = 109,shuffle=True )


#                 self.model.fit(X_train,y_train)
#                 self.Train_score= self.model.score(X_train,y_train)
#                 self.test_score= self.model.score(self.X_test,self.y_actual)
#                 #self.coeff=self.model.coefs_

                
#                 self.y_pred = self.model.predict(self.X_test)

#                 y_pred = np.ndarray.tolist(self.y_pred)
#                 self.length = len(y_pred)
                
#                 #Mean squared error and accuracy
#                 self.mean_squared_error = mean_squared_error(self.y_actual, 
#                 self.y_pred)
                
#             except Exception as e:
#                 self.Error_message= 'Error in Regressor Creation: ' + str(e)
#                 self.flag=True
#                 self.Train_score= 'Refer To error in Regressor Creation'
#                 self.test_score= 'Refer To error in Regressor Creation'
#                 #self.coeff=self.model.coefs_

#                 self.y_actual='Refer To error in Regressor Creation'
#                 self.y_pred = 'Refer To error in Regressor Creation'

#                 self.y_pred = 'Refer To error in Regressor Creation'
#                 self.length = 'Refer To error in Regressor Creation'
                
#                 #Mean squared error and accuracy
#                 self.mean_squared_error = 'Refer To error in Regressor Creation'
#         else:
#             self.Train_score= 'Refer To error in Handling Method'
#             self.test_score= 'Refer To error in Handling Method'
#             #self.coeff=self.model.coefs_

#             self.y_actual='Refer To error in Handling Method'
#             self.y_pred = 'Refer To error in Handling Method'

#             self.y_pred = 'Refer To error in Handling Method'
#             self.length = 'Refer To error in Handling Method'
            
#             #Mean squared error and accuracy
#             self.mean_squared_error = 'Refer To error in Handling Method'


#         return self
        
#     def printing(self):
#         #Printing the chosen metrics
#         if (self.flag) != True:
#             self.Error_message= ' No Error Occurred during prcoessing of the code'
        
#         try:
            

#             # print('Error Message           ', self.Error_message)
#             # print('expected output:        ', self.y_actual)
#             # print('Predicted Output:       ', self.y_pred)
#             # print('Model Train_score on the Training Data: ',  self.Train_score)
#             # print('Model Train_score on the Testing Data:  ',  self.test_score)
#             # print('Mean Square error:      ',  self.mean_squared_error)
#             # print('length of output array: ',  self.length)
#             st.warning(self.Error_message)
#             cc1, cc2 = st.columns(2)
#             with cc1:
#                 # st.metric('Expected output:        ', self.NN_Outputs.y_actual)
#                 # st.write('Predicted Output:       ', self.NN_Outputs.y_pred)
#                 st.metric('Model score on the Training Data:',  round(self.Train_score, 8))
#                 st.metric('Model score on the Testing Data:',  round(self.test_score, 8))
#                 st.metric('Length of output array: ',  self.length)
#             with cc2:
#                 #st.write('Classification Report: ')
#                 st.metric('RMSE: ',  round((self.mean_squared_error)**(0.5),8))
#         except Exception as e:
#             self.Error_message = 'Error while printing outputs: ' +str(e)
#             self.flag=True
#             st.warning(self.Error_message)
#         #print('coeff: ',.coeff )

#     def plotting(self):
#         #Plotting a 2D plot of the Regression output with external input i being the field of which we want to plot as x-axis
#         if self.flag != True:
#             try:
#                 # fig = plt.figure()
#                 # ax1 = fig.add_subplot(111)

#                 # ax1.scatter(self.X_test[:,i],self.y_pred, s=10, c='r', marker="o", label='Predicted')
#                 # ax1.scatter(self.X_test[:,i],self.y_actual, s=10, c='b', marker="o", label='actual')
#                 # ax1.set_xlabel (self.df.columns[i])
#                 # ax1.set_ylabel  (self.df.columns[self.k])   
#                 # ax1.legend()
#                 # plt.show()
#                 df=pd.DataFrame({'y_test':self.y_actual,'y_pred':self.y_pred})
#                 fig = plt.figure(figsize=(10, 4))
#                 sn.scatterplot(x='y_test',y='y_pred',data=df,label='Real Prediction')
#                 sn.lineplot(x='y_pred',y='y_test',data=df,color='red',alpha=0.5,label='Optimal Prediction')
#                 plt.title('Expected vs Predicted Output')
#                 plt.legend()
#                 #plt.show()
#                 st.pyplot(fig)
#             except Exception as e:
#                 self.Error_message='Error in Plotting Method: ' + str(e)
#                 self.flag=True
#                 #print(e)


# #Using the Class
# # data = pd.read_csv("C:\\Users\\Beats\\Desktop\\Python_OOP\\synchronous machine.csv",delimiter=';',decimal=',')
# # #print(data.head)
# # activation_fun1 = ("identity", "logistic", "tanh", "relu")
# # solver_fun1 = ("lbfgs", "sgd", "adam")

# # hidden_layers3=(600,300,100)

# class Regressor_Inputs(NamedTuple):
        
#         test_size:              float  # test size percentage
#         activation_fun:         tuple  # activation function selection
#         hidden_layers:          tuple  # size of hidden layer and number of neurons in each layer
#         solver_fun:             tuple  # solver function
#         Max_iterations:         int    # number of iterations
#         Normalize:              bool   # flag to normalize X data or not

#Creating a tuple that will
# work with the method defined inside the class
# Inputs= Regressor_Inputs(0.2,activation_fun1[3],hidden_layers3,solver_fun1[0],500,False)
# #print(Inputs)

# Regressor1=NN_Regressor(data,Inputs,4)

# #print(Regressor1.NN_Inputs)
# #Calling the Regression method that would create the Neural Network
# Regressor1.Regressor()

# #input the index of the field that you want to occupy the x axis 
# Regressor1.plotting()
# #Printing the metrics chosen for the Regressor
# Regressor1.printing()
