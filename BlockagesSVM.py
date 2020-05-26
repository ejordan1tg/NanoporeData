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
from sklearn.svm import SVC
from sklearn import preprocessing

desired_width = 320
pandas.set_option('display.width', desired_width)
numpy.set_printoptions(linewidth=desired_width)
pandas.set_option('display.max_columns', 10)

location = r"C:\Users\Elijah\Documents\NanoporeData\MethDataBlockagesOnly.csv"
names = ['Level 3% Blockage', 'Level 5% Blockage','Label']
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
values = array[:, 0:2]
print('here are the values of the dataset')
print('\n')
print(values)
print('\n')
labels = array[:, 2]
print('here are the labels of the dataset')
print('\n')
print(labels)
print('\n')
print('training and validation')
print('\n')



validation_size = 0
seed = 7
scoring = 'accuracy'

values_scaled = preprocessing.normalize(values)
values_train, values_validation, labels_train, labels_validation = model_selection.train_test_split(values_scaled, labels,
    test_size = validation_size, random_state=seed)

models=[]
models.append(('SVM', SVC()))
results = []

for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, values_train, labels_train, cv= kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)