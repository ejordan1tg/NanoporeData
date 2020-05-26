##import sys
##import scipy
import numpy
##import matplotlib
import sklearn

import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import preprocessing

desired_width = 320
pandas.set_option('display.width', desired_width)
numpy.set_printoptions(linewidth=desired_width)
pandas.set_option('display.max_columns', 10)

location = r"C:\Users\Elijah\Documents\NanoporeData\1me_nome.csv"
names = ['Total Dwell Time', 'Level 3% Blockage', 'Level 3 Dwell Time', 'Level 5% Blockage', 'Level 5 Dwell Time', 'Label']
dataset = pandas.read_csv(location, names = names)
print(dataset)
print('\n')
print('splitting out the validation dataset')
print('\n')
array = dataset.values
print('\n')
print('here is the array of all of the values ,meaning everything but the names that were added from to the values')
print(array)
print('\n')
values = array[:, 0:5]
print('here are the values of the dataset')
print('\n')
print(values)
print('\n')
labels = array[:, 5]
print('here are the labels of the dataset')
print('\n')
print(labels)
print('\n')
print('training and validation')
print('\n')
validation_size = 0
seed = 7

values_scaled = preprocessing.scale(values)

values_train, values_validation, labels_train, labels_validation = model_selection.train_test_split(values_scaled, labels,
    test_size = validation_size, random_state=seed)

print('printing values_train, values_validation, names_train, names_validation')
print('\n')
print(values_train, values_validation, labels_train, labels_validation)
print('\n')

scoring = 'accuracy'


print('spot check and evaluating each model in turn')
print('\n')
models = []
models.append(('SVM', SVC()))


results = []
names1 = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, values_scaled, labels_train, cv= kfold, scoring=scoring)
    results.append(cv_results)
    names1.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
print(msg)




