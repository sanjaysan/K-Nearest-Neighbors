from __future__ import division
import pandas as pd
import numpy as np
from scipy.spatial import distance as ed
import scipy.io.arff as arff
import collections
import sys
# import time

count = -1
mean_absolute_error = 0
class_labels = []
class_values = []
nbr_instances_correctly_classified = 0
incorrectly_classified_counts_for_each_k = []
mean_absolute_error_for_each_k = []
row = -1

def process_arff_file(file_name):
    loaded_arff = arff.loadarff(open(file_name, "rb"))
    data, metadata = loaded_arff
    features = metadata.names()

    data = pd.DataFrame(data)
    return data, features, metadata

def get_mean(train_data, new_train_data, test_data, row_to_predict, k, metadata, regression = True):
        global row, count, mean_absolute_error, nbr_instances_correctly_classified
        distances = None
        if not new_train_data.empty:
            row += 1
            train_data_leave_one_out = new_train_data.drop(row)
            # Calculating Eucledian distance
            distances = train_data_leave_one_out.apply(lambda train_row : ed.euclidean(train_row, row_to_predict), axis = 1)
        else:
            train_data_without_class_label = train_data.drop(train_data.columns[-1], 1)
            distances = train_data_without_class_label.apply(lambda train_row : ed.euclidean(train_row, row_to_predict), axis = 1)
        # Getting the indexes of the k nearest neighbours
        indexes = distances.to_frame().sort_index(by=[0, 0]).head(k).index

        if regression is True:
            # Finding the mean of the responses of the k nearest neighbours
            mean_error = np.mean(train_data.ix[indexes]['response'])
            count = count + 1
            if not new_train_data.empty:
                test_row_value = train_data.ix[count]['response']
            else:
                test_row_value = test_data.ix[count]['response']
            mean_absolute_error += abs(mean_error - test_row_value)
            if new_train_data.empty:
                print "Predicted value : {0:f}\tActual value : {1:f}".format(mean_error, test_row_value)
        else:
            # Getting the class labels of the k nearest neighbours and converting
            # the type of the result from pandas.core.series.series to list
            class_list = train_data.ix[indexes]['class'].tolist()

            # Finding the number of occurrences of each word
            occurrences = collections.Counter(class_list)
            class_labels_with_count = occurrences.most_common()
            class_labels.append(class_labels_with_count[0][0])
            max_class_count = class_labels_with_count[0][1]
            i = 1
            max_count = 1
            for i in range(1, len(class_labels_with_count)):
                if class_labels_with_count[i][1] == max_class_count:
                    max_count += 1
                    # Add the class label to list
                    class_labels.append(class_labels_with_count[i][0])
                else:
                    break
            predicted_test_row_class = None
            if (len(class_labels) == 1):
                predicted_test_row_class = class_labels[0]
            else:
                dict = {}
                for i in range(0, len(metadata['class'][1])):
                    dict[metadata['class'][1][i]] = i
                j = 0
                min = float('inf')
                for label in class_labels:
                    if label in dict:
                        class_order_in_metadata = dict[label]
                        if class_order_in_metadata < min:
                            min = class_order_in_metadata
                            predicted_test_row_class = label
            count = count + 1
            if not new_train_data.empty:
                actual_test_row_class = train_data.ix[count]['class']
            else:
                actual_test_row_class = test_data.ix[count]['class']
            if predicted_test_row_class == actual_test_row_class:
                nbr_instances_correctly_classified += 1
            if new_train_data.empty:
                print "Predicted class : {0}\tActual class : {1}".format(predicted_test_row_class, actual_test_row_class)
            del class_labels[:]

def predict_train_set(train_data, test_data, k, metadata, regression = True):

    new_train_data = train_data.drop(train_data.columns[-1], 1)
    new_train_data.apply(
        lambda train_row_to_predict: get_mean(train_data, new_train_data, test_data, train_row_to_predict, k, metadata, regression),axis=1)
    if regression is True:
        mean_absolute_error_for_each_k.append((mean_absolute_error, k))
        print "Mean absolute error for k =", k, ": {0:.16f}".format(mean_absolute_error / (count + 1)).rstrip(".0")
    else:
        nbr_instances_incorrectly_classified = (count - nbr_instances_correctly_classified) + 1
        incorrectly_classified_counts_for_each_k.append((nbr_instances_incorrectly_classified, k))
        print "Number of incorrectly classified instances for k =", k, ":", nbr_instances_incorrectly_classified


def predict_test_set(train_data, test_data, k, metadata = None, regression = True):
    # Removing the last column from test set
    test_data_without_class_label = test_data.drop(test_data.columns[-1], 1)
    print "Best k value :",k
    test_data_without_class_label.apply(lambda test_row_to_predict : get_mean(train_data, pd.DataFrame(), test_data, test_row_to_predict, k, metadata, regression), axis = 1)
    if regression is True:
        print "Mean absolute error :","{0:.16f}".format(mean_absolute_error / (count + 1)).rstrip(".0")
        print "Total number of instances :",count + 1
    else:
        print "Number of correctly classified instances :",nbr_instances_correctly_classified
        print "Total number of instances :",count + 1
        print "Accuracy :", "{0:.16f}".format(nbr_instances_correctly_classified / (count + 1)).rstrip(".0")


if __name__ == '__main__':
    # start = time.time()
    train_data, features, metadata = process_arff_file(sys.argv[1])
    test_data, _, _ = process_arff_file(sys.argv[2])
    # "/home/sanjay/MLProjects/KNN/hw2/wine_test.arff"
    kvals = [int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])]

    for k in kvals:
        row, count, nbr_instances_correctly_classified, mean_absolute_error = -1, -1, 0, 0
        if features[-1] == 'response':
            predict_train_set(train_data, test_data, k, metadata)
        else:
            class_values = metadata['class'][1]
            predict_train_set(train_data, test_data, k, metadata, False)
    # min_incorrectly_classified = float("inf")
    # Storing in the form (incorrectly_classified_count, k)
    min_mean_absolute_error = float("inf")
    min_incorrectly_classified = float("inf")
    if features[-1] == 'response':
        min_mean_absolute_error = min(mean_absolute_error_for_each_k)
    else:
        min_incorrectly_classified = min(incorrectly_classified_counts_for_each_k)
    # Getting best k value for classification
    if features[-1] == 'response':
        best_k_value = min_mean_absolute_error[1]
    else:
        best_k_value = min_incorrectly_classified[1]

    #Predicting the test set with the best k value
    row, count, nbr_instances_correctly_classified, mean_absolute_error = -1, -1, 0, 0
    if features[-1] == 'response':
        predict_test_set(train_data, test_data, best_k_value, metadata)
    else:
        class_values = metadata['class'][1]
        predict_test_set(train_data, test_data, best_k_value, metadata, False)

    # print("--- %s seconds ---" % (time.time() - start))