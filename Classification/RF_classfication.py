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

    """Class Sample for Random forest Classification

    This Class contains the methods used for Random forest Classifier

    Class input parameters:

    :param df: The input data frame
    :type df: Pandas DataFrame
    :param estimator: Number of decision trees to be specified by user
    :type estimator: Integer
    :param test_size: User Input - Proportion of test data specified by user in which dataset is to be splitted.
    :type test_size: float

    Class Output Parameters:

    :param Y_pred: The resulting output of the Regression test
    :type Y_pred: int array 
    :param Y_test: The expected output of the Regression test
    :type Y_test: int array  
    :param acctrain: Model accuracy on the Training data
    :type acctrain: float 
    :param acctest: Model accuracy on the Testing data  
    :type acctest: float
    :param Error_message: Error message if an exception was encountered during the processing of the code
    :type Error_message: str
    :param flag: internal flag for marking if an error occurred while processing a previous method
    :type flag: bool
    """

    model_train = None
    model_test = None
    Y_test=None
    Y_pred=None
    X_train=None
    Y_train=None
    acctrain = None
    acctest = None
    Class_len=None
    Report_dic:dict
    Error_message=None
    flag=None

    # instance attribute
    def __init__(self,X,Y):

        """Class Constructor

        :param X: Features to train the model
        :type X: array
        :param Y: Target variable that is to be classified
        :type Y: array
        
        """
        self.X = X
        self.Y = Y

    # random= Sample('estimator','test_size')

    def model(self,estimator, test_size):
        """ Model Method :
         
            This method splits the data into train and test sets, then creates a model based on the user input n_estimator and test_size. 
            
            It calls model 'RandomizedSearchoptim' that returns the best parameters on which the model can be fitted.
            
            It then fits the model based on the best parameter obtained after Randomized search cross validation and test it on the test dataset, then returns the predicted value 'Y_pred' 

            

            :param estimator: User Input - Number of decision trees for random forest classifier
            :type estimator: Integer
            :param test_size: User Input - Proportion of test data specified by user in which dataset is to be splitted.
            :type test_size: float

        """
        try:

            self.estimator=estimator 
            self.test_size=test_size
            # self.criteria = criteria
            # self.depth = depth
            # self.min_split = min_split
            # self.minimum_leaf = minimum_leaf
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
            """ RandomizedSearchoptim Method : 

                This method returns the best parameters using Randomized search cross validation method on which the model is to be fitted.

                Parameters:
                    max_features:      Number of features to consider at every split
                    max_depth :        Maximum number of levels in tree
                    min_samples_split: Minimum number of samples required to split a node
                    in_samples_leaf:   Minimum number of samples required at each leaf node
                    bootstrap:         Method of selecting samples for training each tree

                :return: Best parameters 

            """
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

    def accuracy(self):
        """ accuracy Method :

            Classification accuracy is a measure that indicates a classification model's performance by dividing the number of correct predictions by the total number of predictions. 

            This method returns the accuracy for the training and testing dataset. 
        """
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


    def report(self):
        """ report Method :

                This method prints the Confusion Matrix and classification report for the performance of the whole model.

                Confusion matrix - A confusion matrix is a summary of prediction results on a classification problem.The number of correct and incorrect predictions are summarized with count values and broken down by each class.
                
                Classification Report - The classification report visualizer displays the precision, recall, F1, and support scores for the model. 
                     Precision:  Precision is the ratio of correctly predicted positive observations to the total predicted positive observations.
                     
                     Recall (Sensitivity) - Recall is the ratio of correctly predicted positive observations to the all observations in actual class
                     
                     The F1 score represents the balance of accuracy and recall. F1 Score is the weighted average of Precision and Recall. F1 score is good parameter in analyzing performance of model on an unbalanced dataset.

        """
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

        #st.write(cm)
        
        

    