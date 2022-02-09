import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

# removeOutlier - function based on mean and standard deviation; to use it, you need to define 3 components:
# df - Dataset (synchronous_machine)
# columnName - the name of the column we want to filter ('If')
# n - standard deviation coefficient (multiplier) 
def removeOutlier (df, columnName, n):
    mean = df[columnName].mean() #find the mean for the column
    std = df[columnName].std()  #find standard deviation for column
    print(f'Mean = {mean}, std = {std}') #we can print these 2 values
    print(f'Dataframe size = {df.size}') # also we can print initial size of the dataset
    fromVal = mean - n * std  # find the min value of the filtering boundary (by default n=2)
    toVal = mean + n * std # find the max value of the filtering boundary (by default n=2)
    print(f'Valid values from {fromVal} to {toVal}') # we can print the filtering boundaries
    filtered = df[(df[columnName] >= fromVal) & (df[columnName] <= toVal)] #apply the filtering formula to the column
    return filtered #return the filtered dataset

syncMachine = pd.read_csv("D:\MAIT21\OOP\Data\Regression\synchronous_machine.csv", delimiter=';', decimal=',')
syncMachine = removeOutlier(syncMachine, 'If', 2)
print(f'Dataframe size after filtering = {syncMachine.size}') # also we can print the size of the dataset after filtering
syncMachine.If.hist() # we can plot the histogram after filtering
plt.show()




