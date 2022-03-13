from cProfile import label
from collections import namedtuple
from lib2to3.pgen2.pgen import DFAState
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn 
import seaborn as sn
from sklearn import model_selection
from sklearn import metrics,preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from typing import NamedTuple
from numpy.core.fromnumeric import shape, size
import streamlit as st #streamlit backend

class NN_Regressor:
    
    """Neural Network Regressor Class:

    This Class contains the methods used for Neural Network Regression using MLP Regressor from sklearn library

    Class input parameters:

    :param df: The input data frame
    :type df: Pandas DataFrame
    :param NN_Inputs: Tuple of parameters for the Regressor clarified by the user
    :type NN_Inputs: Named Tuple
    :param dependant_var_index: The index of the target column in the df for the Regression
    :type dependant_var_index: int

    Class Output Parameters:

    :param y_pred: The resulting output of the Regression test
    :type y_pred: float 
    :param y_actual: The expected output of the Regression test
    :type y_actual: float 
    :param length: The length of the output of the Regression test set
    :type length: int 
    :param mean_squared_error: The MSE of the y_pred with respect to the y_actual
    :type mean_squared_error: float 
    :param Train_score: Model Score (R^2) on the Training data
    :type Train_score: float 
    :param test_score: Model Score (R^2) on the Testing data  
    :type test_score: float
    :param model: The MLP Regressor model created using the specified inputs
    :type model: MLPRegressor
    :param Error_message: Error message if an exception was encountered during the processing of the code
    :type Error_message: str
    :param flag: internal flag for marking if an error occurred while processing a previous method
    :type flag: bool
    """
    

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
        """Class Constructor

        :param df: The input data frame
        :type df: Pandas DataFrame
        :param NN_Inputs: Tuple of parameters for the Regressor clarified by the user
        :type NN_Inputs: Named Tuple
        :param dependant_var_index: The index of the target column in the df for the Regression
        :type dependant_var_index: int
        """
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
        """Method Used to Normalize the X data if the user required

        This method is called when the class instance is created and the Normalize flag in the input NN_Inputs tuple is True.

        """
        #Simple Normalization method for X Data
        scaler=preprocessing.MinMaxScaler()
        self.x_n = scaler.fit_transform(self.X)

    #Data handling method, creates the X and Y arrays that go into the Train_test_split method
    #Called when the object is instantiated within the constructor 
    def handle(self):
        """Data Handling Method:

        This method takes the Target column index and splits the data frame "df" into X and Y numpy arrays so they are ready for being split into train and test sets.

        This method is called internally once the class instance is created and the X,Y output arrays are fed to the "Regressor" method.
        """
        
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
        """ 
        Regressor Creation Method:

        This method splits the data into train and test sets, then creates the MLP regressor based on the user inputs from NN_Inputs Named Tuple.
        
        It then fits the model and returns some metrics for the performance of the model on the test and train data sets.

        :return: Modified set of class parameters
        
        """
        
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
        """Printing Outputs:

        This method prints the chosen metrics to the user after the model is trained and fitted.
        
        The metrics are:
            1. Model R2 Score on the Training Data
            2. Model R2 Score on the Testing Data
            3. Length of the output array
            4. Root Mean Squared Error 
        """
        #Printing the chosen metrics
        if (self.flag) != True:
            self.Error_message= ' No Error Occurred during prcoessing of the code'
        
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
        """Plotting Method:

        This method plots the scatter plot of the predicted vs Expected output to visualize the quality of the regression
        """
        #Plotting a 2D plot of the predicted values vs the actual optimal prediction 
        if self.flag != True:
            try:
               
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

#Using the Class
# data = pd.read_csv("D:\MAIT\OOP\Datasets\Regression\synchronous_machine.csv",delimiter=';',decimal=',')
# #print(data.head)
# activation_fun1 = ("identity", "logistic", "tanh", "relu")
# solver_fun1 = ("lbfgs", "sgd", "adam")

# hidden_layers3=(600,300,100)

class Regressor_Inputs(NamedTuple):
    """
    This class is used to parse inputs from the user into this Named Tuple structure for easy use inside the NN_Regressor class.

    Below is a description of the Named Tuple Elements:

    """
    test_size:              float  ;""" Test size percentage"""
    activation_fun:         tuple  ;""" Activation function selection"""
    hidden_layers:          tuple  ;""" Size of hidden layer and number of neurons in each layer"""
    solver_fun:             tuple  ;""" Solver function"""
    Max_iterations:         int    ;""" Number of Maximum iterations"""
    Normalize:              bool   ;""" Flag to normalize X data or not"""  

#Creating a tuple that will work with the method defined inside the class
# Inputs= Regressor_Inputs(0.2,activation_fun1[3],hidden_layers3,solver_fun1[0],500,False)
# # #print(Inputs)

# Regressor1=NN_Regressor(data,Inputs,4)

# #print(Regressor1.NN_Inputs)
# #Calling the Regression method that would create the Neural Network
# Regressor1.Regressor()

# #input the index of the field that you want to occupy the x axis 
# Regressor1.plotting()
# #Printing the metrics chosen for the Regressor
# Regressor1.printing()


#print(NN_Regressor.Error_message)