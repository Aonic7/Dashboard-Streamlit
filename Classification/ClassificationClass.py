import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st

class Classification:
    """
    This Class is used to build one of three
    possible classifiers (KNN,LR,SVM) with a givin dataset.
    """

    def __init__(self, dataframe, column_names=[]):
        """
        Initialization of an instance of the Classification class

        :param pandas.core.frame.DataFrame dataframe: The dataframe which provides both training and test data
        :param list,optional column_names: Column names for the dataframe
        :return: None
        """
        self.data_splitted = False
        # Checks if the input is an instance of a pandas Dataframe
        if isinstance(dataframe, pd.DataFrame):
            self.data = dataframe
            data_shape = self.data.shape
            st.write("Classification-Object with", str(data_shape[0]), "samples and", str(data_shape[1]),
                     "columns was created")
        else:
            raise Exception("No pandas dataframe was given!")
        # Checks if Column names were given
        if column_names and len(column_names) == data_shape[1]:
            self.data.columns = column_names
            # Try Catch if not enough column names were given
        elif not column_names:
            # st.write("No column names were given")
            pass
        else:
            raise Exception("Number of column names dont match with number of columns!")

    def __del__(self):
        pass

    def split_train_test(self, y_column_name, test_size=0.2, random_state=0, upsample=False, scaling=False,
                         deleting_na=False):
        """
        The first of the three important classes methods.
        This function splits the input data frame into training and test data
        and provides three optional steps for data preparation:

            1. data which are not available are deleted
            2. data is upsampled (see: _upsample_dataset)
            3. data is scaled to a value between 0 and 1

        :param str y_column_name: Name of the target column
        :param float,optional test_size: Percentage of the number of test data, default=0.2
        :param int,optional random_state: Value for the seed, default=0
        :param boolean,optional upsample: If True the data gets upsampled, default = False
        :param boolean,optional scaling: If True the data gets scaled, default = False
        :param boolean,optional deleting_na: If True all na data gets deleted, default = False
        :return: None
        """
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        # Checks if the given target column name is part of the dataframe
        if y_column_name in self.data.columns:

            # Delets not available data if deleting_na equals true
            if deleting_na:
                self.data = self.data.dropna(how='all')
                #st.write("Data has been preprocessed!")

            self.data_splitted = True
            self._x_data = self.data.copy()
            # Splits the Dataframe in a test and train portion
            self._x_train, self._x_test = train_test_split(self._x_data,
                                                           test_size=test_size,
                                                           random_state=random_state)

            # Upsamples the Dataset if upsample equals true
            if upsample:
                self._x_train = self._upsample_dataset(y_column_name)

            #seperating both x_test and x_train from the selected target column (y_train/y_test)
            self._y_train = self._x_train.pop(y_column_name)
            self._y_test = self._x_test.pop(y_column_name)

            self._y_train.value_counts().plot(kind="bar",
                                              title="Distribution of classification values in the train data set",
                                              colormap="gray")
            #st.write("Train data shape:\t" + str(self._x_train.shape) + "\ttrain label shape:\t" + str(
                #self._y_train.shape[0]))
            #st.write(
                #"Test data shape:\t" + str(self._x_test.shape) + " \ttest label shape:\t" + str(self._y_test.shape[0]))

            # scaling both the x_train and x_test dataset
            if scaling:
                scaler = StandardScaler()
                self._x_train = scaler.fit_transform(self._x_train)
                self._x_test = scaler.transform(self._x_test)
                #st.write("Data has been rescalled!")



        elif y_column_name not in self.data.columns:
            raise Exception("Given column name was not found in the dataset")

    def _upsample_dataset(self, y_column_name):
        """
        Function for upsampling data

        :param string y_column_name: Name of the target column of the dataset
        :return pandas.core.frame.DataFrame dataframe: The upscaled dataframe
        """

        from sklearn.utils import resample
        # counting the number of 0 and 1 values for the y column in the dataset
        count = self._x_train[y_column_name].value_counts(sort=True)
        # creating a list with every 0 and every 1 for the y column value in the dataset
        mask_min = self._x_train[y_column_name].values == count.keys()[-1]
        mask_max = self._x_train[y_column_name].values == count.keys()[0]
        # Locating and storing both types of datapoints in a specific list
        x_train_min = self._x_train.loc[mask_min]
        x_train_max = self._x_train.loc[mask_max]

        # upsample the datapoints with the minimal number to the maximal number
        x_train_min_upsampled = resample(x_train_min,
                                         replace=True,
                                         n_samples=x_train_max.shape[0],
                                         random_state=0)

        '''
        x_train_max_downsampled = resample(x_train_max, 
                                           replace=True,
                                           n_samples=x_train_min.shape[0],
                                           random_state=0)

        downsampled_x_train = pd.concat([x_train_max_downsampled, x_train_min])
        '''

        # concatenating both the upsampled minimal datapoints with the maximal datapoints
        upsampled_x_train = pd.concat([x_train_max, x_train_min_upsampled])

        return upsampled_x_train

    def build_classifier(self, classifier_name, *args):
        """
        The second important class method.
        This function builds the heart of the class - the classifier.
        It can be distinguished between three different types of classifiers
        by usage of the classifier_name argument.

        :param str classifier_name: Name of the wished classifier.
                                    Three options:
                                    KNN = K-Nearest Neighbor,
                                    SVM = Support Vector Machine,
                                    LR = Logistic Regression
        :param args: Argument depending on the chosen classifier.
                    For KNN: the value of k (int),
                    for SVM: the desired kernel_fun (string = linear, ploy, rbf, sigmoid)
                    for LR: the desired Solver (string = liblinear, newton-cg, lbfgs, sag, saga)
        :return: None
        """
        from sklearn import metrics

        if self.data_splitted:
            if classifier_name == "KNN":  # K nearest neighbor
                self.classifier = self._build_knn_classifier(*args)
            elif classifier_name == "SVM":  # Support Vector Machine
                self.classifier = self._build_SVM_classifier(*args)
            elif classifier_name == "LR":  # Logistic Regression
                self.classifier = self._build_LogisticRegression_classifier(*args)
            else:
                raise Exception(
                    "Classifier was not found! Avialable options are KNN (K-Nearest Neighbor), "
                    "SVM (Support Vector Machine), LR (Logistic Regression)")
            self._y_pred = self.classifier.predict(self._x_test)

            return self.classifier , round(metrics.accuracy_score(self._y_test, self._y_pred), 2)
        else:
            st.write(
                "The data has to be splitted in train and test data befor a "
                "classifier can be build (use the split_train_test command)")

    def _build_knn_classifier(self, k=5):
        """
        Builds the K-Nearest Neighbor Classifier with the given dataframe.

        :param int k: (optional, default=5) Number of k-Values
        :return: sklearn.neighbors.KNeighborsClassifier
        """
        from sklearn.neighbors import KNeighborsClassifier

        classifier = KNeighborsClassifier(n_neighbors=k)
        return classifier.fit(self._x_train, self._y_train)

    def _build_SVM_classifier(self, kernel_fun="linear"):
        """
        Builds the Support Vector Machine Classifier with the given dataframe.

        :param string kernel_fun: (optional, default=linear) Name of the Kernel function: linear, ploy, rbf, sigmoid.
        :return: sklearn.svm.SVC
        """
        from sklearn import svm

        classifier = svm.SVC(kernel=kernel_fun)
        return classifier.fit(self._x_train, self._y_train)

    def _build_LogisticRegression_classifier(self, solver="liblinear"):
        """
        Builds the Logistic Regression Classifier with the given dataframe.

        :param string solver: (optional, default=liblinear) Name of the Solver: liblinear, newton-cg, lbfgs, sag, saga.
        :return: sklearn.linear_model._logistic.LogisticRegression
        """
        from sklearn.linear_model import LogisticRegression

        classifier = LogisticRegression(solver=solver)
        return classifier.fit(self._x_train, self._y_train)

    def show_classifier_accuracy(self):
        """
        The third of the three important
        Shows the Accuracy of the classifier by giving the Accuracy for each class and by showing a correlation matrix.

        :return: None
        """
        from sklearn import metrics
        cnf_matrix = metrics.confusion_matrix(self._y_test, self._y_pred)

        from sklearn.metrics import classification_report

        # creating the classification report
        report_dic = classification_report(self._y_test, self._y_pred,output_dict=True)  # , target_names=target_names)
        report = pd.DataFrame.from_dict(report_dic)  # , target_names=target_names)

        # plot the confusion matrix
        class_names = [0, 1]
        fig, ax = plt.subplots()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names)
        plt.yticks(tick_marks, class_names)
        # st.pyplot(fig)

        # create heatmap
        fig1 = plt.figure(figsize=(4,3))
        sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
        ax.xaxis.set_label_position("top")
        plt.tight_layout()
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        return fig1, cnf_matrix, report


    def drop_dataframe_column(self, column_name):
        """
        This function drops a given column from the dataset.
        (Not used in Streamlit, but helpful for other applications)

        :param str column_name: Name of the cloumn to drop from the dataset
        :return: None
        """
        # Checks if the given column name is part of the dataset
        if column_name in self.data.columns:
            self.data.drop([column_name], axis=1, inplace=True)
            st.write("Column", str(column_name), "was dropped, new dataframe:")
            st.write(self.data.head())
        else:
            raise Exception("Given column name was not found in the dataset")

    def show_correlation(self, figsize=(5, 4)):
        """
        Shows the correlation of the given dataset.
        (Not used in Streamlit, but helpful for other applications)
        :param list figsize: x and y size of the plotted figure (optional, default=(5,4))
        :return: None
        """
        corrmat = self.data.corr()
        # creates a mask for not displaying mirrored data
        mask = np.zeros_like(corrmat)
        mask[np.triu_indices_from(mask)] = True

        f, ax = plt.subplots(figsize=figsize)
        ax = sns.heatmap(corrmat, mask=mask, cmap="YlGnBu", linewidths=0.1, fmt=".2f", annot=True, vmin=-1, vmax=1)
        return

    def get_classifer(self):
        """
        Returns the Classifier which was build in the build_classifier method.
        (Not used in Streamlit, but helpful for other applications)
        :return: sklearn.classifier
        """
        return self.classifier

    ###################################### SOME BASIC DATAFRAME FUNCTIONS
    def show_dataframe_head(self):
        """
        Returns the Head of the internal dataframe.
        (Not used in Streamlit, but helpful for other applications)
        :return: pd.Dataframe.head()
        """
        return self.data.head()

    def describe_dataframe(self):
        """
        Describes the internal dataframe.
        (Not used in Streamlit, but helpful for other applications)
        :return: pd.Dataframe.describe()
        """
        return self.data.describe()

    def show_unique_values(self):
        """
        Shows the unique values of each column of the internal dataset
        (Not used in Streamlit, but helpful for other applications)
        :return: None
        """
        for column_name in self.data.columns:
            st.write(column_name)
            st.write(sorted(self.data[column_name].unique()))
            st.write()