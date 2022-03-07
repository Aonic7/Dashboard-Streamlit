# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

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

    Y_test=None
    Y_pred=None

    # instance attribute
    def __init__(self,X,Y):
        self.X = X
        self.Y = Y

    # random= Sample('estimator','test_size')

    def model(self,estimator, test_size, criteria, depth, minimum_leaf, min_split):
        self.estimator=estimator 
        self.test_size=test_size
        self.criteria = criteria
        self.depth = depth
        self.min_split = min_split
        self.minimum_leaf = minimum_leaf
        # Split the data into training set and testing set
        X_train, X_test, Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=test_size, random_state = 123,stratify=self.Y)
        
        #create a model
        classifier=RandomForestClassifier(n_estimators=estimator,criterion = self.criteria, max_depth = self.depth, min_samples_leaf = self.minimum_leaf, min_samples_split = self.min_split)

        #fitting training data to the model
        classifier.fit(X_train,Y_train)

        self.Y_pred=classifier.predict(X_test)
        return(self.Y_pred)

    def accuracy(self,Y_test,Y_pred):
        #To calculate the accuracy
        acc=accuracy_score(Y_test,Y_pred)
        st.metric("Accuracy of the model: ", round(acc, 4))

    def report(self,Y_test,Y_pred):
        #To display the confusion matrix
        cm=confusion_matrix(Y_test,Y_pred)
        st.write(cm)
        
        #To st.write the Precision, Recall and F1-score
        st.write("Classification report: ")
        #st.write(classification_report(Y_test,Y_pred))
        self.Report_dic=classification_report(Y_test, Y_pred, output_dict=True)
        self.Report=pd.DataFrame.from_dict(self.Report_dic)
        st.write(self.Report)
        
       #To st.write the F1-score for binary target
        f1= f1_score(Y_test, Y_pred)
        st.metric("F1-score: ", round(f1, 4))