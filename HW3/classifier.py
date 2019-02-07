import pickle
import random
import numpy
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
import hw3_utils as utils


# question 1
def euclidian_distance(x_list, y_list):
    dist = 0
    for x, y in zip(x_list, y_list):
        dist += (x-y)**2
    return numpy.sqrt(dist)


# question 2

# The sorting key
def sortByDistance(value):
    return value[2] # Sorting by the 3rd element


class knn_classifier(utils.abstract_classifier):

    def __init__(self, k_factor, data, labels):
        self.k_factor = k_factor
        self.data = data
        self.labels = labels

    def classify(self, features):
        '''
        classify a new set of features
        :param features: the list of features to classify
        :return: a tagging of the given features (1 or 0)
        '''
        # For each of the examples, get the ED from features to it using the function and store the results
        data_as_matrix = []
        for i in range(len(self.data)):
            entry = [self.data[i], self.labels[i], euclidian_distance(features, self.data[i])]
            data_as_matrix.append(entry)

        # Sort the results
        data_as_matrix.sort(key = sortByDistance, reverse = False)

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
        # No training is occuring here because knn's training is being performed in the classification part
        result_knn_classifier = knn_classifier(self.k_factor, data, labels)

        return result_knn_classifier


# question 3.1

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
    # TODO: fix indices according to FB?
    len_of_zeros = len(shuffled_list_zeros) // num_folds
    len_of_ones = len(shuffled_list_ones) // num_folds

    zeros_list_index = 0
    ones_list_index = 0

    # For each group - add the elements and write to file
    for i in range(num_folds):

        file_name = 'ecg_fold_' + str(i+1) +'.data'

        list_for_numfold_i = []

        while zeros_list_index < len_of_zeros * (i + 1):
            list_for_numfold_i.append(shuffled_list_zeros[zeros_list_index])
            zeros_list_index += 1

        while ones_list_index < len_of_ones * (i + 1):
            list_for_numfold_i.append(shuffled_list_ones[ones_list_index])
            ones_list_index += 1

        # list_for_numfold_i has all of the values combined - 0's and 1's (the ratio is preserved)
        with open(file_name, 'wb') as file:
            pickle.dump(list_for_numfold_i, file)


def load_k_fold_data(index):
    '''
    :param index: the index of the subgroup
    :return a tuple of train at index i and label at index i
    '''

    file_name = 'ecg_fold_' + str(index) +'.data'

    with open(file_name, 'rb') as file:
        return pickle.load(file)

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

    # go over each of the files that were created (0 to k-1):
    for i in range(1, k+1):

        eval_group_patients = []
        eval_group_labels = []

        test_groups_patients = []
        test_groups_labels = []

        # Choose the i group as eval
        # put the elements of this group in eval_list

        data_from_load_for_eval = load_k_fold_data(i) # A list of tuples for each i index - 0 is the data and 1 is the label
        for j in range(0, len(data_from_load_for_eval) - 1):
            eval_group_patients.append(data_from_load_for_eval[j][0])
            eval_group_labels.append(data_from_load_for_eval[j][1])

        # put all of the other elements of all of the combined groups in tests_group
        for l in range(1, k+1):
            if l is not i:

                data_from_load_for_test = load_k_fold_data(l)  # A list of tuples for each i index - 0 is the data and 1 is the label
                for m in range(0, len(data_from_load_for_test) - 1):
                    test_groups_patients.append(data_from_load_for_test[m][0])
                    test_groups_labels.append(data_from_load_for_test[m][1])


        # calculate the accuracy and the errors and add to the accuracy and error lists
        curr_classifier = classifier_factory.train(test_groups_patients, test_groups_labels)

        accuracy_counter = 0

        for i in range(len(eval_group_patients)):
            result = curr_classifier.classify(eval_group_patients[i])
            result_from_eval_group = 1 if eval_group_labels[i] == 1 else 0
            if result == result_from_eval_group:
                accuracy_counter += 1

        n = len(eval_group_patients)
        curr_accuracy = accuracy_counter / n
        accuracy.append(curr_accuracy)


    # Return the average of both accuracy of errors
    average_accuracy = numpy.mean(accuracy)
    return average_accuracy, 1-average_accuracy


class id3_factory(utils.abstract_classifier_factory):

    def train(self, data, labels):
        '''
        train a classifier
        :param data: a list of lists that represents the features that the classifier will be trained with
        :param labels: a list that represents the labels that the classifier will be trained with
        :return: id3_classifier object
        '''

        result_id3_classifier = id3_classifier()

        # Here with ID3, the training is being performed prior to the classification
        result_id3_classifier.classifier.fit(data, labels)

        return result_id3_classifier

class id3_classifier(utils.abstract_classifier):

    def __init__(self):
        self.classifier = DecisionTreeClassifier(criterion="entropy")

    def classify(self, features):
        '''
        classify a new set of features
        :param features: the list of features to classify
        :return: a tagging of the given features (1 or 0)
        '''

        result = self.classifier.predict(features.reshape(1, -1))
        if result:
            return 1
        else:
            return 0

class perceptron_factory(utils.abstract_classifier_factory):

    def train(self, data, labels):
        '''
        train a classifier
        :param data: a list of lists that represents the features that the classifier will be trained with
        :param labels: a list that represents the labels that the classifier will be trained with
        :return: id3_classifier object
        '''

        result_perceptron_classifier = perceptron_classifier()

        # Here with ID3, the training is being performed prior to the classification
        result_perceptron_classifier.classifier.fit(data, labels)

        return result_perceptron_classifier

class perceptron_classifier(utils.abstract_classifier):

    def __init__(self):
        self.classifier = Perceptron(max_iter=5, tol=None)

    def classify(self, features):
        '''
        classify a new set of features
        :param features: the list of features to classify
        :return: a tagging of the given features (1 or 0)
        '''

        result = self.classifier.predict(features.reshape(1, -1))
        if result:
            return 1
        else:
            return 0