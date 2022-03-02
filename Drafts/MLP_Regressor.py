from cProfile import label
from collections import namedtuple
from lib2to3.pgen2.pgen import DFAState
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn 
import seaborn as sn
from sklearn import model_selection
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from typing import NamedTuple
from numpy.core.fromnumeric import shape, size

class NN_Regressor:
    
    class Regressor_Outputs(NamedTuple):
   
        y_pred:                     float # resulting output
        y_actual:                   float # expected output
        length:                     int   # length of y_test
        test_mean_squared_error:    float # mean square error
        Train_score:                float
        X_test:                     float
        test_score:                 float
        model:                      MLPRegressor
        Error_message:              str
        flag:                       bool
        #coeff:                     tuple

    dependant_var_index =0
    #NN_Inputs = Regressor_Inputs
    NN_Outputs = Regressor_Outputs
    
    #Constructor
    #External inputs are the Data frame object, the Named Tuple of NN_Inputs and the index of the dependant variable in the Data frame
    def __init__(self,df,NN_Inputs,dependant_var_index):
        self.df=df
        self.NN_Inputs=NN_Inputs
        
        self.k=dependant_var_index
        self.handle()

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
            self.NN_Outputs.Error_message='Error in Handling Method: ' + str(e)
            self.NN_Outputs.flag=True
        
       
    #Method that creates the MLP Regressor and returns the Named Tuple of NN_Outputs to be used in other methods  
    def Regressor(self):
        
        if (self.NN_Outputs.flag) !=True:
            try:
                X_train, self.NN_Outputs.X_test, y_train, self.NN_Outputs.y_actual= train_test_split(self.X,self.y,test_size=self.NN_Inputs.test_size,shuffle=False, random_state=40)
                self.NN_Outputs.model = MLPRegressor(hidden_layer_sizes = self.NN_Inputs.hidden_layers, 
                                    activation = self.NN_Inputs.activation_fun, solver = self.NN_Inputs.solver_fun, 
                                    learning_rate = 'adaptive', max_iter = self.NN_Inputs.Max_iterations, random_state = 109,shuffle=True )


                self.NN_Outputs.model.fit(X_train,y_train)
                self.NN_Outputs.Train_score= self.NN_Outputs.model.score(X_train,y_train)
                self.NN_Outputs.test_score= self.NN_Outputs.model.score(self.NN_Outputs.X_test,self.NN_Outputs.y_actual)
                #self.NN_Outputs.coeff=self.NN_Outputs.model.coefs_

                
                self.NN_Outputs.y_pred = self.NN_Outputs.model.predict(self.NN_Outputs.X_test)

                self.NN_Outputs.y_pred = np.ndarray.tolist(self.NN_Outputs.y_pred)
                self.NN_Outputs.length = len(self.NN_Outputs.y_pred)
                
                #Mean squared error and accuracy
                self.NN_Outputs.test_mean_squared_error = mean_squared_error(self.NN_Outputs.y_actual, 
                self.NN_Outputs.y_pred)
            except Exception as e:
                self.NN_Outputs.Error_message= 'Error in Regressor Creation: ' + str(e)
                self.NN_Outputs.flag=True
        else:
            self.NN_Outputs.Train_score= 'Refer To error in Handling Method'
            self.NN_Outputs.test_score= 'Refer To error in Handling Method'
            #self.NN_Outputs.coeff=self.NN_Outputs.model.coefs_

            self.NN_Outputs.y_actual='Refer To error in Handling Method'
            self.NN_Outputs.y_pred = 'Refer To error in Handling Method'

            self.NN_Outputs.y_pred = 'Refer To error in Handling Method'
            self.NN_Outputs.length = 'Refer To error in Handling Method'
            
            #Mean squared error and accuracy
            self.NN_Outputs.test_mean_squared_error = 'Refer To error in Handling Method'


        return self.NN_Outputs
        
    def printing(self):
        #Printing the chosen metrics
        if (self.NN_Outputs.flag) != True:
            self.NN_Outputs.Error_message= ' No Error Occurred during prcoessing of the code'
        


        print('Error Message           ', self.NN_Outputs.Error_message)
        print('expected output:        ', self.NN_Outputs.y_actual)
        print('Predicted Output:       ', self.NN_Outputs.y_pred)
        print('Model Train_score on the Training Data: ',  self.NN_Outputs.Train_score)
        print('Model Train_score on the Testing Data:  ',  self.NN_Outputs.test_score)
        print('Mean Square error:      ',  self.NN_Outputs.test_mean_squared_error)
        print('length of output array: ',  self.NN_Outputs.length)
        #print('coeff: ', NN_Outputs.coeff )

    def plotting(self,i):
        #Plotting a 2D plot of the Regression output with external input i being the field of which we want to plot as x-axis
        if self.NN_Outputs.flag != True:
            try:
                fig = plt.figure()
                ax1 = fig.add_subplot(111)

                ax1.scatter(self.NN_Outputs.X_test[:,i],self.NN_Outputs.y_pred, s=10, c='r', marker="o", label='Predicted')
                ax1.scatter(self.NN_Outputs.X_test[:,i],self.NN_Outputs.y_actual, s=10, c='b', marker="o", label='actual')
                ax1.set_xlabel (self.df.columns[i])
                ax1.set_ylabel  (self.df.columns[self.k])   
                ax1.legend()
                plt.show()
            except Exception as e:
                self.NN_Outputs.Error_message='Error in Plotting Method: ' + str(e)
                self.NN_Outputs.flag=True
                #print(e)

#Using the Class
data = pd.read_csv("D:\MAIT\OOP\Datasets\Regression\synchronous_machine.csv",delimiter=';',decimal=',')
#print(data.head)
activation_fun1 = ("identity", "logistic", "tanh", "relu")
solver_fun1 = ("lbfgs", "sgd", "adam")

hidden_layers3=(600,300,100)

class Regressor_Inputs(NamedTuple):
        
        test_size:              float  # test size percentage
        activation_fun:         tuple  # activation function selection
        hidden_layers:          tuple  # size of hidden layer and number of neurons in each layer
        solver_fun:             tuple  # solver function
        Max_iterations:         int    # number of iterations

#Creating a tuple that will work with the method defined inside the class
Inputs= Regressor_Inputs(0.2,activation_fun1[3],hidden_layers3,solver_fun1[0],500)
#print(Inputs)

Regressor1=NN_Regressor(data,Inputs,4)

#print(Regressor1.NN_Inputs)
#Calling the Regression method that would create the Neural Network
Regressor1.Regressor()

#input the index of the field that you want to occupy the x axis 
Regressor1.plotting(3)
#Printing the metrics chosen for the Regressor
Regressor1.printing()


#print(NN_Regressor.NN_Outputs.Error_message)