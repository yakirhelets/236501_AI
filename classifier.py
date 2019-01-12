import math
import random

import hw3_utils as utils


# question 1
def euclidian_distance(x_list, y_list):
    dist = 0
    for x, y in zip(x_list, y_list):
        dist += (x-y)**2
    return math.sqrt(dist)


# question 2

# The sorting key
def sortByDistance(value):
    return value[2] # Sorting by the 3rd element


class knn_classifier(utils.abstract_classifier):
    def classify(self, features):
        '''
        classify a new set of features
        :param features: the list of features to classify
        :return: a tagging of the given features (1 or 0)
        '''
        # Load the data and the labels with the appropriate function
        data = utils.load_data()
        # Pass the data and the labels to the factory to get a knn_classifier object (map)
        factory = knn_factory()
        knn_classifier = factory.train(data[0], data[1])
        # For each of the examples, get the ED from features to it using the function and store the resuts
        data_as_matrix = []
        for i in range(len(knn_classifier)):
            entry = [knn_classifier[i][0], knn_classifier[i][1], euclidian_distance(features, knn_classifier[i][0])]
            data_as_matrix.append(entry)

        # Sort the results
        data_as_matrix.sort(key = sortByDistance, reverse = True)

        # Go over the K closest examples and count the number of 1/0
        # for j in range(K):
        return (data_as_matrix[0])[2]
        # Return the number that has the highest vote among the K


class knn_factory(utils.abstract_classifier_factory):
    def train(self, data, labels):
        '''
        train a classifier
        :param data: a list of lists that represents the features that the classifier will be trained with
        :param labels: a list that represents the labels that the classifier will be trained with
        :return: knn_classifier object
        '''
        # Gets raw data and produces a classifier
        classifier = []
        # Get the data and the labels and create a list where each of its entries is a list with
        # two elements, the vector and the label
        for i in range(len(data)):
            entry = [data[i], labels[i]]
            classifier.append(entry)
        # Return this list, which is the classifier
        return classifier


# question 3,1

def create_data_as_list_of_tuples(dataset):

    # Gets raw data and produces a list
    patients = dataset[0]
    labels = dataset[1]
    result = []
    # Get the data and the labels and create a list where each of its entries is a list with
    # two elements, the vector and the label
    for i in range(len(patients)):
        entry = [patients[i], labels[i]]
        result.append(entry)
    return result

def split_crosscheck_groups(dataset, num_folds):
    '''
    :param dataset: a list of examples
    :param num_folds: number of groups to divide to
    '''

    data_as_list_of_tuples = create_data_as_list_of_tuples(dataset)

    num_of_entries_per_group = int(len(data_as_list_of_tuples) / num_folds)

    shuffled_list = random.sample(data_as_list_of_tuples, k=len(data_as_list_of_tuples))
    # For each group - add the elements and write to file
    for i in range(num_folds):
        file_name = 'ecg_fold_<' + str(i+1) +'>.data'
        with open(file_name, 'w') as file:
            for j in range(num_of_entries_per_group):
                index_of_next_element = (i * num_of_entries_per_group) + j
                patient_data = "".join(str(shuffled_list[index_of_next_element][0]).splitlines())
                patient_label = str(shuffled_list[index_of_next_element][1])
                file.write("%s\n" % patient_data)
                file.write("%s\n" % patient_label)


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


# question 3.2

data = utils.load_data()
split_crosscheck_groups(data, 2)