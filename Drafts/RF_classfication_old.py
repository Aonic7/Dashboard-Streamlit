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

# Class variables: Y_test, Y_pred

# Methods:
#          model() -- Splits the data into train and test dataset, fits the training data to the model and uses classifier to get the prediction for testing data
#          accuracy() -- To get the accuracy of the model
#          report() -- To get the classification report which includes the precision, recall and F1-score values for the model

# User inputs:
#          estimator -- a paramter in the RandomForestClassifier which denotes the number of trees (Range : 10 to 1000 )
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

    def model(self,estimator, test_size):
        self.estimator=estimator 
        self.test_size=test_size
        # Split the data into training set and testing set
        X_train, X_test, Y_train, self.Y_test = train_test_split(X, Y, test_size=test_size, random_state = 123,stratify=Y)
        
        #create a model
        classifier=RandomForestClassifier(n_estimators=estimator,criterion = 'gini', max_depth = 5, min_samples_leaf = 3,min_samples_split = 2)

        #fitting training data to the model
        classifier.fit(X_train,Y_train)

        self.Y_pred=classifier.predict(X_test)
        return(self.Y_pred)

    def accuracy(self,Y_test,Y_pred):
        #To calculate the accuracy
        acc=accuracy_score(Y_test,Y_pred)
        print("Accuracy of the model: ",acc) 

    def report(self,Y_test,Y_pred):
        #To display the confusion matrix
        cm=confusion_matrix(Y_test,Y_pred)
        print(cm)
        
        #To print the Precision, Recall and F1-score
        print("Classification report: ")
        print(classification_report(Y_test,Y_pred))
        
       #To print the F1-score for binary target
        f1= f1_score(Y_test, Y_pred)
        print("F1- score: ",f1)

# Reading csv data and converting it to dataframe
Dataframe = pd.read_csv("C:/Users/HP/Desktop/OOPS/trans.csv")
Dataframe.head()
Dataframe.info()

# Renaming column names
Dataframe = Dataframe.rename(columns = {'Recency (months)': 'Recency', 'Frequency (times)': 'Frequency',
                          'Monetary (c.c. blood)': 'Monetary value', 'Time (months)': 'Time',
                          'whether he/she donated blood in March 2007':'Target'})
# To view top 10 rows 
Dataframe.head(10)

# To visualize the nature of data 
count = Dataframe['Target'].value_counts()
print(count)

# Feature Scaling
# Initialize a scaler and apply it to the feature
scaler = MinMaxScaler()
features = ['Recency', 'Frequency', 'Monetary value', 'Time']
scaled_Dataframe = pd.DataFrame(data = Dataframe)
scaled_Dataframe[features] = scaler.fit_transform(Dataframe[features])
scaled_Dataframe.head(10)

# Initialize the features and class/Target
processed_Dataframe = scaled_Dataframe[['Recency', 'Frequency', 'Monetary value', 'Time', 'Target']]
X = processed_Dataframe.iloc[:,:-1]  # contains the features
Y = processed_Dataframe.iloc[:, -1]  # contains the target column 

# object for the class 'Sample'
obj = Sample(X, Y)

obj.model(500,0.35)
obj.report(obj.Y_test,obj.Y_pred)
obj.accuracy(obj.Y_test,obj.Y_pred)