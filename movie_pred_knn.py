import math
import csv
import random

filename = "IMDB-Movie-Data.csv"
fields = []
data = []

with open(filename,'r', encoding="utf8") as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader)
    for row in csvreader:
        for ind in range(len(row)):
             try:
                 row[ind] = float(row[ind])
             except:
                 pass
        data.append(row[6:33])

data = data[:-1]

test_value_global = []


def find_distance(x, y):
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)]))


def find_order(train_value):
    return find_distance(train_value, test_value_global)


def find_class_simple(test_value, training_data,k):
    global test_value_global
    test_value_global = test_value[:-1]
    training_data_copy = sorted(training_data,key=find_order)[:k]
    classes = {}
    for training_values in training_data_copy:
        if training_values[-1] in classes:
            classes[training_values[-1]]+=1
        else:
            classes[training_values[-1]]=1
    return max(classes,key=lambda x:classes[x])


def find_class(test_value, training_data, k):
    global test_value_global
    test_value_global = test_value[:-1]
    training_data_copy = sorted(training_data, key=find_order)[:k]
    classes = {}
    for training_values in training_data_copy:
        try:
            if training_values[-1] in classes:
                classes[training_values[-1]] += 1 * (1 / find_distance(test_value_global, training_values))
            else:
                classes[training_values[-1]] = 1 * (1 / find_distance(test_value_global, training_values))
        except:
            pass
    return max(classes, key=lambda x: classes[x])


random.shuffle(data)
training_data_length = int(0.7 * len(data))
test_data_length = len(data) - training_data_length
training_data = data[:training_data_length]
test_data = data[training_data_length:]
k = int(input("enter the k value"))
correct = 0
test_actual = []
test_pred = []

for test_values in test_data:
    class_found_knn = find_class_simple(test_values, training_data, k)
    test_actual.append(test_values[-1])
    test_pred.append(class_found_knn)
    if class_found_knn == test_values[-1]:
        #print("correctly Predicted")
        correct += 1
    else:
        #print("wrongly Predicted")
        pass
accuracy = correct / len(test_data) * 100


TP, FP, TN, FN = 0, 0, 0, 0

for ind in range(len(test_actual)):
    if test_actual[ind] == 1.0 and test_pred[ind] == 1.0:
        TP+=1
    elif test_actual[ind] == 1.0 and test_pred[ind] == 0.0:
        FN+=1
    elif test_actual[ind] == 0.0 and test_pred[ind] == 1.0:
        FP+=1
    else:
        TN+=1
print('\nFrom Simple KNN: ')
print('TP: ',TP,'FP: ', FP,'TN: ', TN,'FN: ', FN)

precision = TP/(TP+FP)*100
recall = TP/(TP+FN)*100
TPR = precision
FPR = FP/(TN+FP)

print("Accuracy is ", accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('True Positive rate:', TPR)
print('False positive rate:', FPR)


correct = 0
for test_values in test_data:
    class_found_knn = find_class(test_values, training_data, k)
    test_actual.append(test_values[-1])
    test_pred.append(class_found_knn)
    if class_found_knn == test_values[-1]:
        #print("correctly Predicted")
        correct += 1
    else:
        #print("wrongly Predicted")
        pass
accuracy = correct / len(test_data) * 100

TP, FP, TN, FN = 0, 0, 0, 0

for ind in range(len(test_actual)):
    if test_actual[ind] == 1.0 and test_pred[ind] == 1.0:
        TP+=1
    elif test_actual[ind] == 1.0 and test_pred[ind] == 0.0:
        FN+=1
    elif test_actual[ind] == 0.0 and test_pred[ind] == 1.0:
        FP+=1
    else:
        TN+=1
print()
print('From Weighted KNN: ')
print('TP: ',TP,'FP: ', FP,'TN: ', TN,'FN: ', FN)

precision = TP/(TP+FP)*100
recall = TP/(TP+FN)*100
TPR = precision
FPR = FP/(TN+FP)

print("Accuracy is ", accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('True Positive rate:', TPR)
print('False positive rate:', FPR)
