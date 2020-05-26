
from sklearn.svm import SVC
import pandas
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier

location = r"C:\Users\Elijah\Documents\NanoporeData\Levels_2.3.4.4extra 1me_nome.csv"
print(location)
names = ['Total Dwell Time', 'Level 2% Blockage', 'Level 2 Dwell Time', 'Level 3% Blockage', 'Level 3 Dwell Time',
         'Level 4% Blockage', 'Level 4 Dwell Time',
         'Level 4 Extra % Blockage', 'Level 4 Extra Dwell Time',
         'Label']
dataset = pandas.read_csv(location, names = names)
print(dataset)
print('\n')

unknown = r"C:\Users\Elijah\Documents\NanoporeData\UnknownGroup2_2.3.4.4extra.csv"
datasetUK= pandas.read_csv(unknown, names = names)
#datasetUK = datasetUK.reset_index()
print(datasetUK)

array = (dataset.values)

values = array[:, 0:9]
#print("printing values" + '\n')
#print(values)

labels = array[:, 9]
#print("printing labels" + '\n')
#print(labels)


validation_size = 0
seed = 7


##values_scaled = preprocessing.scale(values)
print("values_scaled")
#print(values_scaled)
print('\n')
print("mean:")
#print(values_scaled.mean(axis=0))
print('\n')
print("variance")
#print(values_scaled.std(axis=0))

values_train, values_validation, labels_train, labels_validation = model_selection.train_test_split(values, labels,
    test_size = validation_size, random_state=seed)

##fit final model

model = DecisionTreeClassifier()
print("hello")
model.fit(values_train, labels_train)
print("hello2")


##creating new instances where we do not know the answer

arrayUK = (datasetUK.values)

valuesUK = arrayUK[:, 0:9]
#print("printing valuesUK" + '\n')
#print(valuesUK)
print('\n')

labelsUK = arrayUK[:, 9]
#print("printing labelsUK" + '\n')
#print(labelsUK)
print('\n')

##values_scaledUK = preprocessing.scale(valuesUK)

#valuesnew, _ = valuesUK, labelsUK
#print("values_new, _")
#print(valuesnew, _)
#print('\n')

valuesnew = valuesUK

##making a prediction
labelsnew = model.predict(valuesnew)
print("labelsnew" + '\n')
print(labelsnew)
print('\n')


##show the inputs and predicted outputs
count = 0
print("valuesnew length")
print(valuesnew)
print(len(valuesnew))
print('\n')
print("labelsnew length")
print(len(labelsnew))

for i in range(len(valuesnew)):
    print('\n')
    print("X=%s, Predicted=%s" % (valuesnew[i], labelsnew[i]))
    if labelsnew[i] == "nome":
        count += 1
print("count:")
print(count)
print("valuesnew length")
print(len(valuesnew))
print("labelsnew")
print(labelsnew)




