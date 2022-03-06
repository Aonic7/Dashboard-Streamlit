import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from sklearn import metrics,preprocessing
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN
from typing import NamedTuple
from numpy.core.fromnumeric import shape
import streamlit as st #streamlit backend
import collections


class NN_Classifier:
        
    
    
    y_pred:                     int # resulting output
    y_actual:                   int # expected output
    length:                     int   # length of y_test
    model:                      MLPClassifier #Outputting the Classifier to use outside the class
    Train_score:                float #Model score on Training data
    X_test:                     float #Testing samples
    test_score:                 float #Model score on the Testing data
    Report:                     pd.DataFrame #Complete Report of the classifier performance on the testing data
    Report_dic:                 dict
    Error_message:              str #Error message to be sent to the user if any issues occur
    flag:                       bool #Flag to signal an Error occurred in a previous method


    dependant_var_index =0
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
            self.Class_len=len(collections.Counter(self.y))
        except Exception as e:
            self.Error_message='Error in Handling Method: ' + str(e)
            self.flag=True

    #Data handling method, Used by the Regressor to re-handle the data after resampling and shuffling is done
    def handle2(self,df):
        internal_data =df.drop(df.iloc[:,[self.k]],axis=1)
        nxm=shape(internal_data)
        n=nxm[0]
        m=nxm[1]
    
        
        X=np.ndarray(shape=(n,m),dtype=float, order='F')
        

        for l in range(0,m):  
        
            for i in range(0,n):
            
                X[i,l]=internal_data.iloc[i,l]
        
        Y=np.ndarray(shape=(n,1), dtype=float, order='F')
        Y[:,0]=df.iloc[:,self.k]

        y_t=np.ravel(Y)
        return X, y_t

    #Method that creates the MLP Classifier and returns the Named Tuple of to be used in other methods  
    def Classify(self):
        if (self.flag) !=True:
            try:
                    
                X_train, self.X_test, y_train, self.y_actual= train_test_split(self.x_n,self.y,test_size=self.NN_Inputs.test_size,shuffle=True, random_state=109)
                self.model = MLPClassifier(hidden_layer_sizes = self.NN_Inputs.hidden_layers, 
                                    activation = self.NN_Inputs.activation_fun, solver = self.NN_Inputs.solver_fun, 
                                    learning_rate = 'adaptive', max_iter = self.NN_Inputs.Max_iterations, random_state = 109,shuffle=True,batch_size=15,alpha=0.0005 )
                
                #Re-sampling method used to handle imbalanced data 
                if self.Class_len >2:
                    method = SMOTEENN(random_state=109,sampling_strategy='minority')
                else:
                    method = SMOTEENN(random_state=109,sampling_strategy=0.48)    
                X_res, y_res = method.fit_resample(X_train, y_train)
                df=pd.DataFrame(X_res)
                #print(df)
                new_df=pd.concat([df,pd.DataFrame(y_res,columns=[self.k])],axis=1)
                #print(new_df.head)
                df1=new_df.sample(frac=1,random_state=1).reset_index(drop=True)
                #After Resampling and shuffling, we feed the df to the handle method to generate X,y arrays used to train the model
                x2,y2 = self.handle2(df1)
                #print(shape(x2))
                #print(shape(y2))
                

                
                self.model.fit(x2, y2)

                self.Train_score= self.model.score(X_train,y_train)
                self.test_score= self.model.score(self.X_test,self.y_actual)
                self.y_pred = self.model.predict(self.X_test)

                y_pred_int = np.ndarray.tolist(self.y_pred)
                self.length = len(y_pred_int)
                #target_names = ['class 0', 'class 1']
                self.Report_dic=classification_report(self.y_actual, self.y_pred, output_dict=True)#, target_names=target_names)
                self.Report=pd.DataFrame.from_dict(self.Report_dic)#, target_names=target_names)

                # st.write(self.y_actual)
            except Exception as e:
                self.Error_message= 'Error in Classifier Creation: ' + str(e)
                self.flag=True
                self.Train_score= 'Refer To error in Classifier Creation'
                self.test_score= 'Refer To error in Classifier Creation'
                #self.coeff=self.model.coefs_

                self.y_actual='Refer To error in Classifier Creation'
                self.y_pred = 'Refer To error in Classifier Creation'

                self.y_pred = 'Refer To error in Classifier Creation'
                self.length = 'Refer To error in Classifier Creation'
                
                #Mean squared error and accuracy
                self.Report_dic ={'1': ['Refer To error in Handling Method 1'] }
                self.Report=pd.DataFrame.from_dict(self.Report_dic)#, target_names=target_names)

        else:
            self.Train_score= 'Refer To error in Handling Method'
            self.test_score= 'Refer To error in Handling Method'
            #self.coeff=self.model.coefs_

            self.y_actual='Refer To error in Handling Method'
            self.y_pred = 'Refer To error in Handling Method'

            self.y_pred = 'Refer To error in Handling Method'
            self.length = 'Refer To error in Handling Method'
            
            #Mean squared error and accuracy
            self.Report_dic ={'1': ['Refer To error in Handling Method 1'] }
            self.Report=pd.DataFrame.from_dict(self.Report_dic)#, target_names=target_names)
    


        return self
    
    def printing(self):
        if (self.flag) != True:
            self.Error_message= ' No Error Occurred during processing of the code'
        
        try:
            st.warning(self.Error_message)
            cc1, cc2 = st.columns(2)
            with cc1:
                # st.metric('Expected output:        ', self.NN_Outputs.y_actual)
                # st.write('Predicted Output:       ', self.NN_Outputs.y_pred)
                st.metric('Model score on the Training Data:',  round(self.Train_score, 4))
                st.metric('Model score on the Testing Data:',  round(self.test_score, 4))
                st.metric('Length of output array: ',  self.length)
            with cc2:
                st.write('Classification Report: ')
                st.dataframe(self.Report)
                st.write("")
        except Exception as e:
            self.Error_message = 'Error while printing outputs: ' +str(e)
            self.flag=True
            st.warning(self.Error_message)

        
    def Conf(self):
        
        fig = plt.figure(figsize=(10, 4))
        ax=fig.add_subplot(111)
        if (self.flag) !=True:

            conf_matrix=metrics.confusion_matrix(self.y_actual, self.y_pred)
            df_conf=pd.DataFrame(conf_matrix,range(self.Class_len),range(self.Class_len))
            sn.set(font_scale=1.4)
            sn.heatmap(df_conf, annot=True, annot_kws={"size":16},ax=ax)
            ax.set_xlabel('Predicted labels')
            ax.set_ylabel('True labels')
            key=list(self.Report_dic)
            labels=key[0:(self.Class_len)]
            #print(labels)
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels)

            st.pyplot(fig)
            #plt.show()

        else:
            st.write('Error occurred in previous methods, Refer to Error Message Warning')


class classifier_inputs(NamedTuple):

    
    test_size:              float  # test size percentage
    activation_fun:         tuple  # activation function selection
    hidden_layers:          tuple  # size of hidden layer and number of neurons in each layer
    solver_fun:             tuple  # solver function
    Max_iterations:         int    # number of iterations
    Normalize:              bool   # flag to normalize X data or not