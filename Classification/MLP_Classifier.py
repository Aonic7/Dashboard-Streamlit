import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from sklearn import metrics,preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN
from typing import NamedTuple
from numpy.core.fromnumeric import shape
import streamlit as st #streamlit backend

from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils.multiclass import unique_labels
from tabulate import tabulate

def classification_report(y_true, y_pred, labels=None, target_names=None,
                          sample_weight=None, digits=4, tablfmt='pipe'):
    """  Better format for sklearn's classification report
    Based on tabulate package
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values.
    y_pred : 1d array-like, or label indicator array / sparse matrix
        Estimated targets as returned by a classifier.
    labels : array, shape = [n_labels]
        Optional list of label indices to include in the report.
    target_names : list of strings
        Optional display names matching the labels (same order).
    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.
    digits : int
        Number of digits for formatting output floating point values
    Returns
    -------
    report : string
        Text summary of the precision, recall, F1 score for each class.
        The reported averages are a prevalence-weighted macro-average across
        classes (equivalent to :func:`precision_recall_fscore_support` with
        ``average='weighted'``).
        Note that in binary classification, recall of the positive class
        is also known as "sensitivity"; recall of the negative class is
        "specificity".
    Examples
    --------
    >>> from sklearn.metrics import classification_report
    >>> y_true = [0, 1, 2, 2, 2]
    >>> y_pred = [0, 0, 2, 2, 1]
    >>> target_names = ['class 0', 'class 1', 'class 2']
    >>> print(classification_report(y_true, y_pred, target_names=target_names))
                 precision    recall  f1-score   support
    """
    floatfmt = '.{:}f'.format(digits)
    if labels is None:
        labels = unique_labels(y_true, y_pred)
    else:
        labels = np.asarray(labels)

    if target_names is not None and len(labels) != len(target_names):
        print(
            "labels size, {0}, does not match size of target_names, {1}"
            .format(len(labels), len(target_names))
        )

    last_line_heading = 'avg / total'

    if target_names is None:
        target_names = [u'%s' % l for l in labels]

    headers = ["precision", "recall", "f1-score", "support"]

    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred,
                                                  labels=labels,
                                                  average=None,
                                                  sample_weight=sample_weight)

    rows = zip(target_names, p, r, f1, s)
    tbl_rows = []
    for row in rows:
        tbl_rows.append(row)

    # compute averages
    last_row = (last_line_heading,
                np.average(p, weights=s),
                np.average(r, weights=s),
                np.average(f1, weights=s),
                np.sum(s))
    tbl_rows.append(last_row)
    return tabulate(tbl_rows, headers=headers,
                    tablefmt=tablfmt, floatfmt=floatfmt)

class NN_Classifier:
        
    class classifier_outputs(NamedTuple):
    
        y_pred:                     int # resulting output
        y_actual:                   int # expected output
        length:                     int   # length of y_test
        model:                      MLPClassifier #Outputting the Classifier to use outside the class
        Train_score:                float #Model score on Training data
        X_test:                     float #Testing samples
        test_score:                 float #Model score on the Testing data
        Report:                     dict #Comlete Report of the classifier performance on the testing data
        Error_message:              str #Error message to be sent to the user if any issues occur
        flag:                       bool #Flag to signal an Error occurred in a previous method


    dependant_var_index =0
    
    NN_Outputs = classifier_outputs
    #Constructor
    #External inputs are the Data frame object, the Named Tuple of NN_Inputs and the index of the dependant variable in the Data frame
    def __init__(self,df,NN_Inputs,dependant_var_index):
        self.df=df
        self.NN_Inputs=NN_Inputs
        self.NN_Outputs.flag = False
        self.NN_Outputs.Error_message = 'No errors found.'
        
        self.k=dependant_var_index
        self.handle()
        #After initial Handling of the data we check to see if the user wants to normalize the X values prior to training the model
        if self.NN_Inputs.Normalize:
            self.preprocess()
        elif self.NN_Outputs.flag!=True:
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
            self.NN_Outputs.Error_message='Error in Handling Method: ' + str(e)
            self.NN_Outputs.flag=True

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

    #Method that creates the MLP Classifier and returns the Named Tuple of NN_Outputs to be used in other methods  
    def Classify(self):
        if (self.NN_Outputs.flag) !=True:
            try:
                    
                X_train, self.NN_Outputs.X_test, y_train, self.NN_Outputs.y_actual= train_test_split(self.x_n,self.y,test_size=self.NN_Inputs.test_size,shuffle=True, random_state=109)
                self.NN_Outputs.model = MLPClassifier(hidden_layer_sizes = self.NN_Inputs.hidden_layers, 
                                    activation = self.NN_Inputs.activation_fun, solver = self.NN_Inputs.solver_fun, 
                                    learning_rate = 'adaptive', max_iter = self.NN_Inputs.Max_iterations, random_state = 109,shuffle=True,batch_size=15,alpha=0.0005 )
                
                #Re-sampling method used to handle imbalanced data 
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
                

                
                self.NN_Outputs.model.fit(x2, y2)

                self.NN_Outputs.Train_score= self.NN_Outputs.model.score(X_train,y_train)
                self.NN_Outputs.test_score= self.NN_Outputs.model.score(self.NN_Outputs.X_test,self.NN_Outputs.y_actual)
                self.NN_Outputs.y_pred = self.NN_Outputs.model.predict(self.NN_Outputs.X_test)

                self.NN_Outputs.y_pred = np.ndarray.tolist(self.NN_Outputs.y_pred)
                self.NN_Outputs.length = len(self.NN_Outputs.y_pred)
                #target_names = ['class 0', 'class 1']
                self.NN_Outputs.Report=classification_report(self.NN_Outputs.y_actual, self.NN_Outputs.y_pred)#, target_names=target_names)
                # st.write(self.NN_Outputs.y_actual)
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
            self.NN_Outputs.Report = 'Refer To error in Handling Method 1'  
    


        return self.NN_Outputs
    
    def printing(self):
        
        if (self.NN_Outputs.flag) != True:
            self.NN_Outputs.Error_message= ' No Error Occurred during processing of the code'
        

        st.warning(self.NN_Outputs.Error_message)
        cc1, cc2 = st.columns(2)
        with cc1:
            # st.metric('Expected output:        ', self.NN_Outputs.y_actual)
            # st.write('Predicted Output:       ', self.NN_Outputs.y_pred)
            st.metric('Model Train_score on the Training Data:',  round(self.NN_Outputs.Train_score, 4))
            st.metric('Model Train_score on the Testing Data:',  round(self.NN_Outputs.test_score, 4))
            st.metric('Length of output array: ',  self.NN_Outputs.length)
        with cc2:
            st.write('Classification Report: ')
            st.write(self.NN_Outputs.Report)
            st.write("")

        
    def Conf(self):
        fig = plt.figure()
        if (self.NN_Outputs.flag) !=True:
            fig = plt.figure(figsize=(10, 4))
            conf_matrix=metrics.confusion_matrix(self.NN_Outputs.y_actual, self.NN_Outputs.y_pred)
            df_conf=pd.DataFrame(conf_matrix,range(2),range(2))
            sn.set(font_scale=1.4)
            sn.heatmap(df_conf, annot=True, annot_kws={"size":16})
            st.pyplot(fig)
        else:
            st.write('Error occurred in previous methods, Refer to Error Message Field')


class classifier_inputs(NamedTuple):

    
    test_size:              float  # test size percentage
    activation_fun:         tuple  # activation function selection
    hidden_layers:          tuple  # size of hidden layer and number of neurons in each layer
    solver_fun:             tuple  # solver function
    Max_iterations:         int    # number of iterations
    Normalize:              bool   # flag to normalize X data or not

# data2 = pd.read_csv("D:\MAIT\OOP\Datasets/transfusion.csv",',')
# data = pd.read_csv("D:\\TH Koeln\\Wolf\\Project\\Data\\Classification.data", ',')

# activation_fun1 = ("identity", "logistic", "tanh", "relu")
# solver_fun1 = ("lbfgs", "sgd", "adam")
# hidden_layers2=(20,5)
# inputs=classifier_inputs(0.2,activation_fun1[1],hidden_layers2,solver_fun1[2],500,False)

# Classifier = NN_Classifier(data,inputs,4)
# Classifier.Classify()
# Classifier.printing()
#Classifier.Conf()
