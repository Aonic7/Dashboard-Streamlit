import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from sklearn.svm import NuSVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import streamlit as st

class Regression:

    ############### Load Pandas Dataframe ###############

    def __init__(self, dataframe):
        """Constructor method

        :param dataframe: Give input as Pandas Dataframe
        :type dataframe: pd.DataFrame
        :raises Exception: When no Pandas Dataframe is given into the function
        """

        if isinstance(dataframe, pd.DataFrame):
            self.dataframe = dataframe
            self.dataframe_shape = dataframe.shape
            print("Regression-Object with", str(self.dataframe_shape[0]), "samples and", str(self.dataframe_shape[1]),
                  "columns was created")
        else:
            raise Exception("No pandas dataframe was given!")

    def __str__(self):
        pass

    def __del__(self):
        pass

    ############### Drop Columns for Regression ###############

    def dropColumns(self, label_drop):
        """Deletes a specified column within the dataframe.

        :param label_drop: Target Columns to drop
        :type label_drop: str
        """

        dropped_dataframe = self.dataframe.drop(label_drop, axis=1)

        print("Column: ", label_drop, " is deleted.")
        return dropped_dataframe

    ############### Split Columns ###############

    def split_columns(self, label_target, new_label_1, new_label_2, seperator=" "):
        """Splits a specified column into two new columns.

        :param label_target: Target column to split
        :type label_target: str
        :param new_label_1: Name of the first new column
        :type new_label_1: str
        :param new_label_2: Name of the second new column
        :type new_label_2: str
        :param seperator: Operator to seperate the target column, defaults to " "
        :type seperator: str, optional
        """

        if (label_target in self.dataframe.columns):
            print("Target column is within the Dataframe.")
            try:
                new_column = self.dataframe[label_target].str.split(seperator, n=1, expand=True)
                new_column_1 = new_column[0]
                new_column_2 = new_column[1]

                self.dataframe[new_label_1] = new_column_1
                self.dataframe[new_label_2] = new_column_2

                self.dataframe = self.dataframe.drop(label_target, 1)

                print("Regression-Object was changed to ", str(self.dataframe_shape[0]), "samples and",
                      str(self.dataframe_shape[1]), "columns.")

            except:
                print(
                    "The Target Column couldnt be seperated. Check wether the selected seperator and given data is correct.")
        else:
            print("The Target Column is not within the Dataframe. Check your Input.")

    ############### Dividing Data into Training and Test ###############

    def split_train_test(self, label_target, testsize=0.3, random_state=1, deleting_na=False, scaling=False,
                         deleting_duplicates=False):
        """Sets a target column and splits the given dataframe into test and training data.

        :param label_target: Sets the target column for the regression
        :type label_target: str
        :param testsize: Represents the proportion of the dataset to include in the test split, defaults to 0.3
        :type testsize: float, optional
        :param random_state: Controls the shuffling applied to the data before applying the split, defaults to 1
        :type random_state: int, optional
        :param deleting_na: Remove missing values., defaults to False
        :type deleting_na: bool, optional
        :param scaling: Standardize features by removing the mean and scaling to unit variance, defaults to False
        :type scaling: bool, optional
        :param deleting_duplicates: Deletes duplicates, defaults to False
        :type deleting_duplicates: bool, optional
        """

        if (label_target in self.dataframe.columns.values):
            print("The target label is set as: ", label_target)
            self.label_target = label_target

        else:
            print("Error: No valid label name!")
            label_target = self.dataframe.columns.values[len(self.dataframe.columns.values) - 1]
            self.label_target = label_target
            print("As default the last column of dataframe is placed as target label: ", label_target)


        if deleting_na:
            self.dataframe = self.dataframe.dropna(how='all')
            st.write("Data has been preprocessed!")

        if scaling:
            scaler = StandardScaler()
            self.dataframe = pd.DataFrame(scaler.fit_transform(self.dataframe), columns=self.dataframe.columns)
            # dataframe = scaler.transform(dataframe)
            st.write("Data has been rescalled!")

        if deleting_duplicates:
            self.dataframe.drop_duplicates(keep='first', inplace=True)
            st.write("Duplicates have been deleted!")

        self.x = self.dataframe.drop(label_target, axis=1)
        self.y = self.dataframe[[label_target]]
        self.y = np.ravel(self.y)

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=testsize,
                                                                                random_state=random_state)

        print("The given data is seperated into test and training data for the given Target Label")
        print("Train data size for x:", self.x_train.shape)
        print("Train data size for y:", self.y_train.shape)

        self.data_splitted = True

        #except:
            #st.write("Something went wrong. Check your Inputs.")

    ############### Create the Regression Model ###############

    def build_regression(self, regression_name, **args):
        """Builds a specified Regression Model with given Training Data.

        :param regression_name: Name of the chosen Regression Model :type regression_name: str = 'Support Vector Machine Regression ','Elastic Net Regression ','Ridge Regression ','Linear Regression ', 'Stochastic Gradient Descent Regression '
        :raises Exception: _description_
        """

        if self.data_splitted:
            if regression_name == "Support Vector Machine Regression ":
                self.regression = self._build_svm_regression(**args)
            elif regression_name == "Elastic Net Regression ":
                self.regression = self._build_elastic_net_regression(**args)
            elif regression_name == "Ridge Regression ":
                self.regression = self._build_ridge_regression(**args)
            elif regression_name == "Linear Regression ":
                self.regression = self._build_linear_regression(**args)
            elif regression_name == "Stochastic Gradient Descent Regression ":
                self.regression = self._build_sgd_regression(**args)
            else:
                raise Exception(
                    "Regression was not found! Avialable options are Support Vector Machine Regression (SVM), Polynomial Regression and Ridge Regression")
            self.y_pred = self.regression.predict(self.x_test)
            params = [str(regression_name),
                      round(mean_squared_error(self.y_test, self.y_pred), 4),
                      round(r2_score(self.y_test, self.y_pred),4)]
            return self.regression, params
        else:
            print(
                "The data has to be splitted in train and test data befor a regression can be build (use the split_train_test command).")

    def _build_svm_regression(self, kernel='poly', degree=3, svmNumber=0.5, maxIterations=-1):
        """**Nu Support Vector Regression**

        :param kernel: Specifies the kernel type to be used in the algorithm, defaults to 'poly'
        :type kernel: str = 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed', str, optional
        :param degree: Degree of the polynomial kernel function, defaults to 3
        :type degree: int, optional
        :param svmNumber: An upper bound on the fraction of training errors and a lower bound of the fraction of support vectors, should be in the interval (0, 1), defaults to 0.5
        :type svmNumber: float, optional
        :param maxIterations: Hard limit on iterations within solver, or -1 for no limit, defaults to -1
        :type maxIterations: int, optional
        """
        # kernel {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’} or callable, default=’rbf’
        # degree: int, default = 3
        # svmNumber: float, default=0.5
        # max_iter: int, default=-1

        svm = NuSVR(kernel=kernel, degree=degree, nu=svmNumber, max_iter=maxIterations)
        return svm.fit(self.x_train, self.y_train)

    def _build_ridge_regression(self, max_iter=15000, solver='auto'):
        """Ridge Regression or Tikhonov regularization.

        :param max_iter: Maximum number of iterations for conjugate gradient solver, defaults to 15000
        :type max_iter: int, optional
        :param solver: Solver to use in the computational routines:, defaults to 'auto'
        :type solver: str = 'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs', optional
        """
        # max_iter: int, default=None
        # solver{‘auto’, ‘svd’, ‘cholesky’, ‘lsqr’, ‘sparse_cg’, ‘sag’, ‘saga’, ‘lbfgs’}, default=’auto’

        clf = Ridge(max_iter=max_iter, solver=solver)
        return clf.fit(self.x_train, self.y_train)

    def _build_elastic_net_regression(self):
        """Gradient Boosting for regression.


        """

        eln = GradientBoostingRegressor()
        return eln.fit(self.x_train, self.y_train)

    def _build_linear_regression(self):
        """Ordinary least squares Linear Regression
        """

        reg = LinearRegression()
        return reg.fit(self.x_train, self.y_train)

    def _build_sgd_regression(self, max_iter=1000):
        """Linear model fitted by minimizing a regularized empirical loss with SGD.

        :param max_iter: The maximum number of passes over the training data, defaults to 1000
        :type max_iter: int, optional
        """

        sgd = SGDRegressor(max_iter=max_iter)
        return sgd.fit(self.x_train, self.y_train)

    ############### Predict User Inputs ###############

    def regression_function(self, user_input):
        """Feeds User Inputs into the Regression Model and outputs the Prediction

        :param user_input: Give input as Pandas Dataframe
        :type user_input: pd.DataFrame
        """

        if isinstance(user_input, pd.DataFrame):
            user_input = user_input
            print("User Input is given as Pandas Dataframe!")
            if len(user_input.columns) == (len(self.dataframe.columns) - 1):
                print("Number of Input variables fits the function!")
                result = round(self.regression.predict(user_input)[0], 2)
                print('The prediction for the target label "', self.label_target, '" is ', result, '.')
                return result
        else:
            print(
                "User Input is not correct. Check wether the Input is converted as a Pandas Dataframe and the length of the input frame is correct."
                "The active dataframe allows ", (len(self.dataframe.columns) - 1),
                " input variables. The predicted colum is ", self.label_target)

    ############### Plot Data in 2D Graph ###############

    def plot_correlation(self, label_target_1, label_target_2):
        """Plots the correlation of two Target Labels, defined by the User, within a Scatter Plot.

        :param label_target_1: First column name to plot
        :type label_target_1: str
        :param label_target_2: Second column name to plot
        :type label_target_2: str
        """

        if label_target_1 in self.dataframe.columns:

            if label_target_2 in self.dataframe.columns:

                try:
                    plt.scatter(self.dataframe[label_target_1], self.dataframe[label_target_2])
                    plt.xlabel(label_target_1)
                    plt.ylabel(label_target_2)
                    plt.show
                except:
                    print("Something went wrong. Check your Inputs.")

            else:
                print("The second selected column is not within the Dataframe")

        else:
            print("The first selected column is not within the Dataframe")

    ############### Plot Data in 2D Graph with regression ###############

    def plot_regression_1(self):
        """Plots the deviation between the prediction and the acutal test data of the Target Label.
        """

        try:
            fig = plt.figure()
            plt.scatter(self.y_test, self.regression.predict(self.x_test))

            point1 = [min(self.y_test), min(self.y_test)]
            point2 = [max(self.y_test), max(self.y_test)]
            x_values = [point1[0], point2[0]]
            y_values = [point1[1], point2[1]]
            plt.plot(x_values, y_values, 'bo', linestyle="--", color='red')

            plt.xlabel("y_test")
            plt.ylabel("y_pred")
            return fig

        except:
            print("Something went wrong. Check your Inputs.")

    def plot_regression_2(self, label_second):
        """Plots the deviation between the prediction and the acutal test data from a choosen column.

        :param label_second: Set dimension / column to plot
        :type label_second: str
        """

        if label_second in self.dataframe.columns:
            print("First Target Column is within the Dataframe.")

        try:

            # show prediction with test data
            plt.scatter(self.x_test[label_second], self.y_test)
            plt.scatter(self.x_test[label_second], self.regression.predict(self.x_test), color='red', alpha=0.3)

            plt.xlabel("x")
            plt.ylabel("y")
            plt.show()


        except:
            print("Something went wrong. Check your Inputs.")

        else:
            print("The first selected column is not within the Dataframe")

    ############### Heatmap for correlation ###############

    def plot_heatmap_correlation(self, figsize=(5, 4)):
        """Plots a heatmap to show correlation between different columns of the dataframe.

        :param figsize: Size of the figure, defaults to (5,4)
        :type figsize: tuple, optional
        """

        corrmat = self.dataframe.corr()

        mask = np.zeros_like(corrmat)
        mask[np.triu_indices_from(mask)] = True

        f, ax = plt.subplots(figsize=figsize)
        ax = sns.heatmap(corrmat, mask=mask, cmap="YlGnBu", linewidths=0.1, fmt=".2f", annot=True, vmin=-1, vmax=1, ax=ax)
        st.pyplot(f)

    ############### Main Effects Plot ##############
    def MainEffectsPlot(self):

        self.dataframe

        # normData = (self.dataframe-np.min(self.dataframe))/(np.max(self.dataframe)-np.min(self.dataframe))

        meanList = []
        fig = plt.figure()
        for a in list(self.dataframe.columns):
            if a == self.label_target:
                pass
            else:
                meanList.append(np.mean(self.dataframe[a]))

        ii = 0

        for i in range(len(list(self.dataframe.columns))):

            if self.dataframe.columns[i] == self.label_target:

                pass
            else:

                yy = []
                xx = []

                PredRange = np.linspace(min(self.dataframe[self.dataframe.columns[i]]), max(self.dataframe[self.dataframe.columns[i]]), 20)

                for e in PredRange:
                    meanList[ii] = e
                    xx.append(e)
                    yy.append(self.regression_function(pd.DataFrame([meanList])))

                plt.plot(xx, yy, label=self.dataframe.columns[i])

                ii += 1

        plt.xlabel('input')
        plt.ylabel(self.label_target)
        plt.legend(loc='upper left')
        return fig 

    ############### Check Active Regression Type ###############

    def get_regression_type(self):
        """Returns the Regression Type of the created model.

        :return: Regression Type
        :rtype: str
        """

        return self.regression

    ############### Get Pandas Dataframe Description ###############

    def get_dataframe_description(self):
        """Returns a brief overiew with basic information of the dataframe.

        :return: Dataframe Overview
        :rtype: table
        """
        return self.dataframe.describe()

    ############### Get Pandas Dataframe Description ###############

    def get_dataframe_head(self):
        """Return the head of the dataframe.

        :return: Dataframe Head
        :rtype: table
        """

        return self.dataframe.head()
