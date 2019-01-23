import pickle
import random
import numpy
import matplotlib.pyplot as plt
import hw3_utils as utils
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from classifier import split_crosscheck_groups, knn_factory, evaluate




class part_c_classifiers():

    # def __init__(self, k_factor, data, labels):
    #     self.a = k_factor

    # TODO: Change names of functions later
    # TODO: Trained on everything. Is that ok?
    def classifier_one(self):
        patients, labels, test = utils.load_data()
        split_crosscheck_groups([patients, labels], 10)
        k_list = [1,3,5,7,13]
        best_k = 0
        best_accuracy_value = 0

        for k in k_list:
            knn_f = knn_factory(k)
            accuracy, error = evaluate(knn_f, 10)
            if accuracy > best_accuracy_value:
                best_k = k
                best_accuracy_value = accuracy

        # Returns a classifier
        knn_f = knn_factory(best_k)
        return knn_f.train(patients, labels)


    # TODO: Change names of functions later
    # TODO: Trained on everything. Is that ok?
    def classifier_two(self):

        num_folds_values_list = [2,3,5,8,10]
        k_list = [1,3,5,7,13]
        patients, labels, test = utils.load_data()

        best_k = 0
        best_average_accuracy_value = 0

        for k in k_list:

            accuracy_list_for_k = []

            for nf in num_folds_values_list:
                split_crosscheck_groups([patients, labels], nf)
                knn_f = knn_factory(k)
                accuracy, error = evaluate(knn_f, nf)
                accuracy_list_for_k.append(accuracy)

            average_accuracy_for_k = numpy.mean(accuracy_list_for_k)

            if average_accuracy_for_k > best_average_accuracy_value:
                best_k = k
                best_average_accuracy_value = average_accuracy_for_k

        # Returns a classifier
        knn_f = knn_factory(best_k)
        return knn_f.train(patients, labels)


    # TODO: Change names of functions later
    # TODO: Trained on everything. Is that ok?
    def classifier_three(self):

        k_list = [1, 3, 5, 7, 13]
        patients, labels, test = utils.load_data()

        committee_classifier = knn_committee_classifier(k_list, patients, labels)
        return committee_classifier




class knn_committee_classifier(utils.abstract_classifier):

    def __init__(self, k_list, data, labels):
        self.k_factor = k_list
        self.data = data
        self.labels = labels

    def classify(self, features):
        '''
        classify a new set of features
        :param features: the list of features to classify
        :return: a tagging of the given features (1 or 0)
        '''

        true_counter = 0
        false_counte = 0

        for k in self.k_list:
            knn_f = knn_factory(k)
            knn_classifier = knn_f.train(self.data, self.labels)
            result = knn_classifier.classify(features)
            if result:
                true_counter += 1
            else:
                false_counte += 1

        if true_counter > false_counte:
            return True
        else:
            return False


    # def train_classifier_with_all_data(self):
    #     patients, labels, test = utils.load_data()

