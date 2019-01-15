import math
import random

import numpy as np
import hw3_utils as utils


# question 1
def euclidian_distance(x_list, y_list):
    dist = 0
    for x, y in zip(x_list, y_list):
        dist += (x-y)**2
    return np.sqrt(dist)


# question 2

# The sorting key
def sortByDistance(value):
    return value[2] # Sorting by the 3rd element


class knn_classifier(utils.abstract_classifier):

    def __init__(self, k_factor, data_list):
        self.k_factor = k_factor
        self.data_list = data_list

    def classify(self, features):
        '''
        classify a new set of features
        :param features: the list of features to classify
        :return: a tagging of the given features (1 or 0)
        '''
        # For each of the examples, get the ED from features to it using the function and store the results
        data_as_matrix = []
        for i in range(len(self.data_list)):
            entry = [self.data_list[i][0], self.data_list[i][1], euclidian_distance(features, self.data_list[i][0])]
            data_as_matrix.append(entry)

        # Sort the results
        data_as_matrix.sort(key = sortByDistance, reverse = True)

        zeros = 0
        ones = 0
        # Go over the K closest examples and count the number of 1's and 0's
        for j in range(self.k_factor):
            if (data_as_matrix[j])[1] == 0:
                zeros += 1
            else:
                ones += 1

        # Return the number that has the highest vote among the K
        if zeros > ones:
            return 0
        else:
            return 1

class knn_factory(utils.abstract_classifier_factory):

    def __init__(self, k_factor):
        self.k_factor = k_factor

    def train(self, data, labels):
        '''
        train a classifier
        :param data: a list of lists that represents the features that the classifier will be trained with
        :param labels: a list that represents the labels that the classifier will be trained with
        :return: knn_classifier object
        '''
        # Gets raw data and produces a classifier
        data_list = []
        # Get the data and the labels and create a list where each of its entries is a list with
        # two elements, the vector and the label
        for i in range(len(data)):
            entry = [data[i], labels[i]]
            data_list.append(entry)
        # Return this list, which is the classifier
        classifier = knn_classifier(self.k_factor, data_list)

        return classifier


# question 3,1

def split_crosscheck_groups(dataset, num_folds):
    '''
    :param dataset: a list of examples
    :param num_folds: number of groups to divide to
    '''

    # divide the group into 0's and 1's

    list_of_zeros = []
    list_of_ones = []

    for i in range(len(dataset[0])):
        value_tuple = (dataset[0][i], dataset[1][i])

        if value_tuple[1] == 0:
            list_of_zeros.append(value_tuple)
        else:
            list_of_ones.append(value_tuple)

    # shuffle each group

    shuffled_list_zeros = random.sample(list_of_zeros, k=len(list_of_zeros))
    shuffled_list_ones = random.sample(list_of_ones, k=len(list_of_ones))

    # for num_folds: take len(0/1) / num_folds into each group (only data[0] and data[1])
    # TODO: fix indices
    len_of_zeros = len(shuffled_list_zeros) // num_folds
    len_of_ones = len(shuffled_list_ones) // num_folds

    zeros_list_index = 0
    ones_list_index = 0


    # For each group - add the elements and write to file
    for i in range(num_folds):

        file_name = 'ecg_fold_<' + str(i+1) +'>.data'
        with open(file_name, 'w') as file:

            while zeros_list_index < len_of_zeros * (i+1):
                patient_data = "".join(str(shuffled_list_zeros[zeros_list_index][0]).splitlines())
                file.write("%s\n" % patient_data)
                patient_label = "".join(str(shuffled_list_zeros[zeros_list_index][1]).splitlines())
                file.write("%s\n" % patient_label)

                zeros_list_index += 1


            while ones_list_index < len_of_ones * (i+1):
                patient_data = "".join(str(shuffled_list_ones[ones_list_index][0]).splitlines())
                file.write("%s\n" % patient_data)
                patient_label = "".join(str(shuffled_list_ones[ones_list_index][1]).splitlines())
                file.write("%s\n" % patient_label)

                ones_list_index += 1

def load_k_fold_data(index):
    '''
    :param index: the index of the subgroup
    :return a tuple of train at index i and label at index i
    '''
    file_name = 'ecg_fold_<' + str(index) +'>.data'
    with open(file_name) as file:
        lines = file.read().splitlines()
    # lines = [a.strip() for a in file_content]

    patients = []
    labels = []
    for i in range(0, len(lines)-1, 2):
        patients.append(lines[i])
        labels.append(lines[i+1])

    return (patients, labels)

# question 4
def evaluate(classifier_factory, k):
    '''
    :param classifier_factory: a classifier factory object
    :param k: number of folds
    :return the means of the accuracy and the error
    '''
    # load the training sets
    # load the test set

    accuracy = []
    error = []

    # go over each of the files that were created (0 to k-1):
    for i in range(k):

        # Choose the i group as eval
        file_name = 'ecg_fold_<' + str(i+1) + '>.data'
        with open(file_name) as file:
            lines = file.read().splitlines()

            patients = []
            labels = []
            for i in range(0, len(lines) - 1, 2):
                patients.append(lines[i])
                labels.append(lines[i + 1])

        # put the elements of this group in eval_list and all of the others combined in test_list

        # calculate the accuracy and the errors

    # Return the average of both accuracy of errors




    return math.mean(accuracy), math.mean(error)





# question 3.2

data = utils.load_data()
split_crosscheck_groups(data, 2)