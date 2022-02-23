# removeOutlier_q - function based on the removal of data that lie below the lower quantile (by defolt - 25%) 
# and above the upper quantile (by defolt - 75%)
#to use it, you need to define 4 components:
# df - Dataset (synchronous_machine)
# columnName - the name of the column we want to filter ('If')
# n1 - lower quantile
# n2 - upper quantile
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
def removeOutlier_q (df, columnName, n1, n2):
    print(f'Dataframe size = {df.size}') # we can print initial size of the dataset
    lower_quantile, upper_quantile = df[columnName].quantile([n1, n2]) #quantiles are generally expressed as a fraction (from 0 to 1)
    filtered = df[(df[columnName] > lower_quantile) & (df[columnName] < upper_quantile)]
    return filtered #return the filtered dataset

syncMachine = pd.read_csv("D:\MAIT21\OOP\Data\Regression\synchronous_machine.csv", delimiter=';', decimal=',')
syncMachine = removeOutlier_q(syncMachine,'If', 0.25, 0.75)
print(f'Dataframe size after filtering = {syncMachine.size}') # also we can print the size of the dataset after filtering
syncMachine.If.hist()
plt.show()


