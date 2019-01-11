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
#The sorting key
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