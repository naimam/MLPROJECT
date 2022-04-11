# Import libraries
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import nan_euclidean_distances

# Import datasets
missingdata1 = pd.read_csv('input/MissingData1.txt', sep='\s+', header=None, na_values='1.00000000000000e+99')
missingdata2 = pd.read_csv('input/MissingData2.txt', sep='\s+', header=None, na_values='1.00000000000000e+99')
missingdata3 = pd.read_csv('input/MissingData3.txt', sep='\s+', header=None, na_values='1.00000000000000e+99')
missing_data = [missingdata1, missingdata2, missingdata3]

# KNN Imputation
def KNNImpute(data, k):
    result = data.copy() # copy the data
    indexes_nan = np.argwhere(np.isnan(data)) # find the indexes of the missing values
    dist_matrix = nan_euclidean_distances(data) # calculate the distance matrix for the data
    for index in indexes_nan: # iterate over the indexes of the missing values
        row = index[0] 
        col = index[1] 
        sorted_rows = dist_matrix[row].argsort() # sorts the distance matrix for the row
        n_values = [] 
        for i in sorted_rows: # iterate over the sorted rows
            if(np.isnan(data[i,col])): # if the value is empty then skip it
                continue
            n_values.append(data[i,col]) # else add the value to the list
            if(len(n_values)==k): # if the length of the list is equal to k then break
                break
        new_value = np.mean(n_values) # calculate the mean of the k closest values
        result[row][col] = new_value # replace the missing value with the mean
    return(result)

if __name__ == "__main__":
    for i in range(len(missing_data)):
        k = int(np.sqrt(missing_data[i].shape[0]))  # k = square root of the number of rows
        output = KNNImpute(missing_data[i].values, k)
        output = pd.DataFrame(output)
        output.to_csv('output/MohamedMissingResult'+str(i+1)+'.txt', sep='\t', index=False, header=False)

