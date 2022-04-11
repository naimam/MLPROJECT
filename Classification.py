import numpy as np
import pandas as pd
import sklearn
from sklearn import impute
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import sys
np.set_printoptions(threshold=sys.maxsize)

# Import datasets
traindata1 = pd.read_csv('input/TrainData1.txt', sep='\s+', header=None, na_values='1.00000000000000e+99')
traindata2 = pd.read_csv('input/TrainData2.txt', sep='\s+', header=None, na_values='1.00000000000000e+99')
traindata3 = pd.read_csv('input/TrainData3.txt', sep='\s+', header=None, na_values='1.00000000000000e+99')
traindata4 = pd.read_csv('input/TrainData4.txt', sep='\s+', header=None, na_values='1.00000000000000e+99')
traindata5 = pd.read_csv('input/TrainData5.txt', sep='\s+', header=None, na_values='1.00000000000000e+99')
train_data = [traindata1, traindata2, traindata3, traindata4, traindata5]

testdata1 = pd.read_csv('input/TestData1.txt', sep='\s+', header=None, na_values='1.00000000000000e+99')
testdata2 = pd.read_csv('input/TestData2.txt', sep='\s+', header=None, na_values='1.00000000000000e+99')
testdata3 = pd.read_csv('input/TestData3.txt', delimiter = ",", header=None,na_values=1000000000)
testdata4 = pd.read_csv('input/TestData4.txt', sep='\s+', header=None, na_values='1.00000000000000e+99')
testdata5 = pd.read_csv('input/TestData5.txt', sep='\s+', header=None, na_values='1.00000000000000e+99')
test_data = [testdata1, testdata2, testdata3, testdata4, testdata5]

trainlabel1 = pd.read_csv('input/TrainLabel1.txt', sep='\t', header=None)
trainlabel2 = pd.read_csv('input/TrainLabel2.txt', sep='\t', header=None)
trainlabel3 = pd.read_csv('input/TrainLabel3.txt', sep='\t', header=None)
trainlabel4 = pd.read_csv('input/TrainLabel4.txt', sep='\t', header=None)
trainlabel5 = pd.read_csv('input/TrainLabel5.txt', sep='\t', header=None)
trainlabel1 = trainlabel1.values.ravel()
trainlabel2 = trainlabel2.values.ravel()
trainlabel3 = trainlabel3.values.ravel()
trainlabel4 = trainlabel4.values.ravel()
trainlabel5 = trainlabel5.values.ravel()
train_label = [trainlabel1, trainlabel2, trainlabel3, trainlabel4, trainlabel5]

# KNN Imputation to fill missing values
for i in range(len(train_data)):
    k = int(np.sqrt(train_data[i].shape[0])/2)  # k = square root of the number of rows
    imputer = KNNImputer(n_neighbors=k)
    train_data[i] = imputer.fit_transform(train_data[i])
    test_data[i] = imputer.transform(test_data[i])

# Oversample/Downsample  dataset 1,3,5 the data to avoid biased predictions
# dataset 1:
oversample_1 = RandomOverSampler(sampling_strategy={1: 108, 2: 60, 4: 60, 3: 55, 5: 50})
train_data[0], train_label[0] = oversample_1.fit_resample(train_data[0], train_label[0])

# dataset 3:
oversample_3 = RandomOverSampler(sampling_strategy={1: 1235, 8: 1200, 6: 1200, 7: 1200, 9: 1200, 4: 1200, 2: 1200, 5: 1200, 3: 1200})
train_data[2], train_label[2] = oversample_3.fit_resample(train_data[2], train_label[2])

#dataset 5:
oversample_5 = RandomOverSampler(sampling_strategy={5: 471, 6: 447, 7: 400, 4: 300, 8: 200, 3: 200})
train_data[4], train_label[4] = oversample_5.fit_resample(train_data[4], train_label[4])

# Split the data into training and testing sets
X_train = []
X_test = []
y_train = []
y_test = []

def split_data(data, label):
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.3, random_state = 2)
    return X_train, X_test, y_train, y_test

for i in range(len(train_data)):
    X_train_i, X_test_i, y_train_i, y_test_i = split_data(train_data[i], train_label[i])
    X_train.append(X_train_i)
    X_test.append(X_test_i)
    y_train.append(y_train_i)
    y_test.append(y_test_i)
 
# Standardize the data
sc = StandardScaler()
for i in range(len(train_data)):
    X_train[i] = sc.fit_transform(X_train[i])
    X_test[i] = sc.transform(X_test[i])
    test_data[i] = sc.transform(test_data[i])
    
# train the model
test_labels = []
# dataset 1:
clf = MLPClassifier(hidden_layer_sizes=(50, 32),activation="relu",random_state=1,max_iter=2000).fit(X_train[0], y_train[0])
train_pred = clf.predict(X_test[0])
print("Dataset 1: " + str(accuracy_score(y_test[0], train_pred)))
test_labels.append(clf.predict(test_data[0]))

# dataset 2:
clf = MLPClassifier(hidden_layer_sizes=(75,64,32),activation="relu",random_state=1,max_iter=2000).fit(X_train[1], y_train[1])
train_pred = clf.predict(X_test[1])
print("Dataset 2: " + str(accuracy_score(y_test[1], train_pred)))
test_labels.append(clf.predict(test_data[1]))

# dataset 3:
clf = MLPClassifier(hidden_layer_sizes=(500,300,150,75,32),activation="relu",random_state=1,max_iter=3000).fit(X_train[2], y_train[2])
train_pred = clf.predict(X_test[2])
print("Dataset 3: " + str(accuracy_score(y_test[2], train_pred)))
test_labels.append(clf.predict(test_data[2]))

# dataset 4:
clf = MLPClassifier(hidden_layer_sizes=(500,150,75,32),activation="relu",random_state=1,max_iter=3000).fit(X_train[3], y_train[3])
train_pred = clf.predict(X_test[3])
print("Dataset 4: " + str(accuracy_score(y_test[3], train_pred)))
test_labels.append(clf.predict(test_data[3]))

# dataset 5:
clf = MLPClassifier(hidden_layer_sizes=(500,128,64,32),activation="relu",random_state=1,max_iter=3000).fit(X_train[4], y_train[4])
train_pred = clf.predict(X_test[4])
print("Dataset 5: " + str(accuracy_score(y_test[4], train_pred)))
test_labels.append(clf.predict(test_data[4]))

if __name__ == "__main__":
    #export the test labels to file
    for i in range(len(test_labels)):
        test_labels[i] = test_labels[i].astype(int)
        test_labels[i] = pd.DataFrame(test_labels[i])
        test_labels[i].to_csv('MohamedClassification'+str(i+1)+'.txt', sep='\t', index=False, header=False)



