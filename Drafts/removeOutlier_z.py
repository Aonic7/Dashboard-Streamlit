import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import scipy.stats as stats

# removeOutlier_z - function based on Z-score;The z-score tells us how many standard deviations away a value is from the mean. 
#to use it, you need to define 3 components:
# df - Dataset (synchronous_machine)
# columnName - the name of the column we want to filter ('If')
# n - standard deviation coefficient
def removeOutlier_z (df, columnName, n):
    z = np.abs(stats.zscore(df[columnName])) #find the Z-score for the column
    print(f'Z-score for each value in column = {z}') #we can print Z-score values
    print(f'Dataframe size = {df.size}') # also we can print initial size of the dataset
    filtered = df[(z < n)] #apply the filtering formula to the column
    return filtered #return the filtered dataset

syncMachine = pd.read_csv("D:\MAIT21\OOP\Data\Regression\synchronous_machine.csv", delimiter=';', decimal=',')
syncMachine = removeOutlier_z(syncMachine,'If', 1.1)
print(f'Dataframe size after filtering = {syncMachine.size}') # also we can print the size of the dataset after filtering
syncMachine.If.hist()
plt.show()