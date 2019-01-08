# task 1
import numpy as np
import hw3_utils as utils

def euclidian_distance(x_list, y_list):
    dist = 0
    for x, y in zip(x_list, y_list):
        dist += (x-y)**2
    return np.sqrt(dist)

# TEST FOR TASK 1
# list1 = [1,2,3,4,5,6,7]
# list2 = [7,6,5,4,3,2,1]
#
# print(euclidian_distance(list1,list2))


# task 2
class knn_classifier(utils.abstract_classifier):
    def classify(self, features):
        '''
        classify a new set of features
        :param features: the list of feature to classify
        :return: a tagging of the given features (1 or 0)
        '''
        # TODO: implement


class knn_factory(utils.abstract_classifier_factory):
    def train(self, data, labels):
        '''
        train a classifier
        :param data: a list of lists that represents the features that the classifier will be trained with
        :param labels: a list that represents  the labels that the classifier will be trained with
        :return: knn_classifier object
        '''
        # TODO: implement
