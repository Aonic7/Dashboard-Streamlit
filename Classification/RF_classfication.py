# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import streamlit as st

# Class variables: Y_test, Y_pred 

# Methods:
#          model() -- Splits the data into train and test dataset, fits the training data to the model and uses classifier to get the prediction for testing data
#          accuracy() -- To get the accuracy of the model
#          report() -- To get the classification report which includes the precision, recall and F1-score values for the model

# User inputs:
#          estimator -- a parameter in the RandomForestClassifier which denotes the number of trees (Range : 10 to 1000 )
#          test_size -- proportion of the original dataset to be included in the test split  (Range: 0% to 100 % )

# Outputs:
#          Accuracy  
#          F1 Score 
#          Classification Report
#          Confusion Matrix

class Sample:

    model_train = None
    model_test = None
    Y_test=None
    Y_pred=None
    X_train=None
    Y_train=None
    acctrain = None
    acctest = None
    Class_len=None
    Report_dic=dict
    Error_message=None
    flag=None

    # instance attribute
    def __init__(self,X,Y):
        self.X = X
        self.Y = Y

    # random= Sample('estimator','test_size')

    def model(self,estimator, test_size, criteria, depth, minimum_leaf, min_split):
        try:

            self.estimator=estimator 
            self.test_size=test_size
            self.criteria = criteria
            self.depth = depth
            self.min_split = min_split
            self.minimum_leaf = minimum_leaf
            # Split the data into training set and testing set
            X_train, X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=test_size, random_state = 123,stratify=self.Y)

            #create a model
            classifier=RandomForestClassifier(n_estimators=estimator)   
            b = self.RandomizedSearchoptim()
            rf_classifier = RandomizedSearchCV(estimator= classifier ,param_distributions = b, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
            #fitting training data to the model
            rf_classifier.fit(X_train,self.Y_train)
            self.model_train = rf_classifier.predict(X_train)

            self.Y_pred=rf_classifier.predict(X_test)
        except Exception as e:
            self.Error_message = 'Error while creating model: ' +str(e)
            self.flag=True
            st.warning(self.Error_message)
        #return(self.Y_pred)
    def RandomizedSearchoptim(self):
            try:
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
            except Exception as e:
                self.Error_message = 'Error while creating model: ' +str(e)
                self.flag=True
                st.warning(self.Error_message)
                random_grid=[]
            return random_grid

    def accuracy(self,k,l):
        #To calculate the accuracy
        if self.flag != True:
            try:
                acc=accuracy_score(self.Y_test,self.Y_pred)
               # st.write("Pass2")

                acctrain = accuracy_score(self.Y_train,self.model_train)
                #st.write("Pass2")

                print("Train accuracy of the model: ",acctrain)
                print("Test accuracy of the model: ",acc)
                st.metric("Accuracy of the model on the Training data: ", round(acctrain, 4))
                st.metric("Accuracy of the model on the Test data: ", round(acc, 4))
            except Exception as e:
                self.Error_message = 'Error while calculating accuracy: ' +str(e)
                self.flag=True
                st.warning(self.Error_message)
        else:
            st.write('Error occurred in previous methods, Refer to Error Message Warning')


    def report(self,Y_test,Y_pred):
        #To display the confusion matrix
        if self.flag != True:
            try:
                import collections
                self.Report_dic=classification_report(self.Y_test,self.Y_pred,output_dict=True)
                self.y=np.ravel(self.Y)
                self.Class_len=len(collections.Counter(self.y))
                fig = plt.figure(figsize=(10, 4))
                ax=fig.add_subplot(111)
                conf_matrix=confusion_matrix(self.Y_test, self.Y_pred)
                df_conf=pd.DataFrame(conf_matrix,range(self.Class_len),range(self.Class_len))
                sns.set(font_scale=1.4)
                sns.heatmap(df_conf, annot=True, annot_kws={"size":16},ax=ax)
                ax.set_xlabel('Predicted labels')
                ax.set_ylabel('True labels')
                key=list(self.Report_dic)
                labels=key[0:(self.Class_len)]
                print(labels)
                ax.set_xticklabels(labels)
                ax.set_yticklabels(labels)

                plt.show()
                st.pyplot(fig)
                        #To st.write the Precision, Recall and F1-score
                st.write("Classification report: ")
                #st.write(classification_report(Y_test,Y_pred))
                #self.Report_dic=classification_report(Y_test, Y_pred, output_dict=True)
                self.Report=pd.DataFrame.from_dict(self.Report_dic)

                st.dataframe(self.Report)
                st.write("Precision:  Precision is the ratio of correctly predicted positive observations to the total predicted positive observations. ")
                st.write('Recall (Sensitivity) - Recall is the ratio of correctly predicted positive observations to the all observations in actual class')
                st.write('The F1 score represents the balance of accuracy and recall. F1 Score is the weighted average of Precision and Recall. Good for Unbalanced dataset.')

            except Exception as e:
                self.Error_message = 'Error while Printing output: ' +str(e)
                self.flag=True
                st.warning(self.Error_message) 
        else:
            st.write('Error occurred in previous methods, Refer to Error Message Warning')







    # Y_test=None
    # Y_pred=None

    # # instance attribute
    # def __init__(self,X,Y):
    #     self.X = X
    #     self.Y = Y

    # # random= Sample('estimator','test_size')

    # def model(self,estimator, test_size, criteria, depth, minimum_leaf, min_split): 
    #     self.estimator=estimator 
    #     self.test_size=test_size
    #     self.criteria = criteria
    #     self.depth = depth
    #     self.min_split = min_split
    #     self.minimum_leaf = minimum_leaf
    #     # Split the data into training set and testing set
    #     X_train, X_test, Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=test_size, random_state = 123,stratify=self.Y)
            
    #     #create a model
    #     classifier=RandomForestClassifier(n_estimators=estimator,criterion = self.criteria, max_depth = self.depth, min_samples_leaf = self.minimum_leaf, min_samples_split = self.min_split)

    #     #fitting training data to the model
    #     classifier.fit(X_train,Y_train)

    #     self.Y_pred=classifier.predict(X_test)    
         
    #     return(self.Y_pred)

    # def accuracy(self,Y_test,Y_pred):
    #     #To calculate the accuracy
    #     acc=accuracy_score(Y_test,Y_pred)
    #     st.metric("Accuracy of the model: ", round(acc, 4))

    # def report(self,Y_test,Y_pred):
    #     #To display the confusion matrix
    #     cm=confusion_matrix(Y_test,Y_pred)
    #     st.write(cm)
        
    #     #To st.write the Precision, Recall and F1-score
    #     st.write("Classification report: ")
    #     #st.write(classification_report(Y_test,Y_pred))
    #     self.Report_dic=classification_report(Y_test, Y_pred, output_dict=True)
    #     self.Report=pd.DataFrame.from_dict(self.Report_dic)
    #     st.write(self.Report)
        
    #    #To st.write the F1-score for binary target
    #     f1= f1_score(Y_test, Y_pred)
    #     st.metric("F1-score: ", round(f1, 4))