# import python core modules

# import 3rd party pkgs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

###################################
# loading data and preprocessing
###################################

# loading data
from sklearn.datasets import load_breast_cancer
cancer_data = load_breast_cancer()

# description of dataset
# print(cancer_data.DESCR)

# keys in disctionary
# print(cancer_data.keys())
# output: dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])

# unique feature names (30 column name)
# print(cancer_data.feature_names)

# look into data
# data for each patients (30 column about 500 rows)
# print(cancer_data.data)

# convering to pandas dataframes
df = pd.DataFrame(data=cancer_data.data, columns=cancer_data.feature_names)

# first five rows with the heading of dataframes
# print(df.head())

# description of dataframe, statictics
# print(df.describe())


# add 'target' column to the dataframe
df['target'] = cancer_data.target

# unique values of a specific column
# print(df['target'].value_counts())


# spliting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    cancer_data.data,
    cancer_data.target,
    stratify=cancer_data.target,
    shuffle=True,
    random_state=144)

# train and test shapes
# print(X_train.shape)
# output: (426, 30)

# print(X_test.shape)
# output: (143, 30)

# print(y_train.shape)
# output: (426,)

# print(y_test.shape)
# output:(143,)


###################################
# applying KNN
###################################

# initializeing model
knn = KNeighborsClassifier(n_neighbors=3)

# training our model
knn.fit(X_train, y_train)

# making prediction output is 1s and 0s
# print("Test set predictions: {}".format(knn.predict(X_test)))

# test accuracy
# print("Test set accuracy: {:.2f}".format(knn.score(X_test, y_test)))
# output: 0.92


###################################
# improve performance (tuning)
###################################
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_minmax_scaled = scaler.transform(X_train)
X_test_minmax_scaled = scaler.transform(X_test)


# test performace after tuning
knn.fit(X_train_minmax_scaled, y_train)
# print("Test set Accuracy {:.2f}".format(
#     knn.score(X_test_minmax_scaled, y_test)))
# output: Test set Accuracy 0.97

# print("Train set Accuracy {:.2f}".format(
#     knn.score(X_train_minmax_scaled, y_train)))
# output: Train set Accuracy 0.98

# the accuracy boosted from 0.92 to 0.97


###################################
# Testing with defferent values of k
###################################
train_accuracy = []
test_accuracy = []
neighbors = range(1, 11)

for n in neighbors:
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(X_train_minmax_scaled, y_train)
    train_accuracy.append(knn.score(X_train_minmax_scaled, y_train))
    test_accuracy.append(knn.score(X_test_minmax_scaled, y_test))

plt.plot(neighbors, train_accuracy, label="train accuracy")
plt.plot(neighbors, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("neighbors")
plt.legend()
plt.show()

# based on the ploted results, k=5 and k=3 give us the best result
# so the final result
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_minmax_scaled, y_train)

print(knn.score(X_train_minmax_scaled, y_train))
# output: 0.9835680751173709

print(knn.score(X_test_minmax_scaled, y_test))
# output: 0.965034965034965
