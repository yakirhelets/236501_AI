import random
import numpy
import matplotlib.pyplot as plt

import classifier
import hw3_utils as utils
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from classifier import split_crosscheck_groups, evaluate
from sklearn.ensemble import RandomForestClassifier



# TODO: Change names of functions later
# class classifier_one(utils.abstract_classifier):
#
#     def __init__(self, data, labels):
#         self.data = data
#         self.labels = labels
#
#     def classify(self, features):
#         split_crosscheck_groups([self.data, self.labels], 10)
#         k_list = [1,3,5,7,13]
#         best_k = 0
#         best_accuracy_value = 0
#
#         for k in k_list:
#             knn_f = knn_factory(k)
#             accuracy, error = evaluate(knn_f, 10)
#             if accuracy > best_accuracy_value:
#                 best_k = k
#                 best_accuracy_value = accuracy
#
#         # Returns a classifier
#         knn_f = knn_factory(best_k)
#         result_classifier = knn_f.train(self.data, self.labels)
#         return result_classifier.classify(features)


# TODO: Change names of functions later
# class classifier_two(utils.abstract_classifier):
#
#     def __init__(self, data, labels):
#         self.data = data
#         self.labels = labels
#
#     def classify(self, features):
#
#         num_folds_values_list = [2,3,5,8,10]
#         k_list = [1,3,5,7,13]
#
#         best_k = 0
#         best_average_accuracy_value = 0
#
#         for k in k_list:
#
#             accuracy_list_for_k = []
#
#             for nf in num_folds_values_list:
#                 split_crosscheck_groups([self.data, self.labels], nf)
#                 knn_f = knn_factory(k)
#                 accuracy, error = evaluate(knn_f, nf)
#                 accuracy_list_for_k.append(accuracy)
#
#             average_accuracy_for_k = numpy.mean(accuracy_list_for_k)
#
#             if average_accuracy_for_k > best_average_accuracy_value:
#                 best_k = k
#                 best_average_accuracy_value = average_accuracy_for_k
#
#         # Returns a classifier
#         knn_f = knn_factory(best_k)
#         result_classifier = knn_f.train(self.data, self.labels)
#         return result_classifier.classify(features)



# TODO: Change names of functions later
# ------------------ KNN ENSEMBLE CLASSIFIER -------------------
# class classifier_three(utils.abstract_classifier):
#
#     def __init__(self, data, labels):
#         self.data = data
#         self.labels = labels
#
#     def classify(self, features):
#
#         k_list = [1, 3, 5, 7, 13]
#
#         true_counter = 0
#         false_counte = 0
#
#         for k in k_list:
#             knn_f = knn_factory(k)
#             knn_classifier = knn_f.train(self.data, self.labels)
#             result = knn_classifier.classify(features)
#             if result:
#                 true_counter += 1
#             else:
#                 false_counte += 1
#
#         if true_counter > false_counte:
#             return True
#         else:
#             return False
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
        self.classifier = Perceptron()

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
#
# # TODO: Change names of functions later
# # ------------------ RANDOM FOREST CLASSIFIER -------------------
# class classifier_four(utils.abstract_classifier):
#
#     def __init__(self, data, labels):
#         self.data = data
#         self.labels = labels
#
#     def classify(self, features):
#         classifier = RandomForestClassifier(n_estimators=10) # 10 random classifiers
#         result_classifier = classifier.fit(self.data, self.labels)
#         return result_classifier.classify(features)

class random_forest_factory(utils.abstract_classifier_factory):

    def train(self, data, labels):
        '''
        train a classifier
        :param data: a list of lists that represents the features that the classifier will be trained with
        :param labels: a list that represents the labels that the classifier will be trained with
        :return: id3_classifier object
        '''

        random_forest_clf = random_forest_classifier()
        result_random_forest_classifier = random_forest_clf.fit(self.data, self.labels)

        # Here with ID3, the training is being performed prior to the classification
        # result_id3_classifier.classifier.fit(data, labels)

        return result_random_forest_classifier

class random_forest_classifier(utils.abstract_classifier):

    def __init__(self):
        self.classifier = RandomForestClassifier(n_estimators=10) # 10 random classifiers

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

class knn_classifier(utils.abstract_classifier):

    def __init__(self, data, labels):
        self.k_factor = 1
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
            entry = [self.data[i], self.labels[i], classifier.euclidian_distance(features, self.data[i])]
            data_as_matrix.append(entry)

        # Sort the results
        data_as_matrix.sort(key = classifier.sortByDistance, reverse = False)

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

    def train(self, data, labels):
        '''
        train a classifier
        :param data: a list of lists that represents the features that the classifier will be trained with
        :param labels: a list that represents the labels that the classifier will be trained with
        :return: knn_classifier object
        '''
        # No training is occuring here because knn's training is being performed in the classification part
        result_knn_classifier = knn_classifier(1, data, labels)

        return result_knn_classifier

    # לקחת כמה מסווגים ada boost ועוד מסווגים ולממש להם factory וגם classifier ואת train ו-classify בשביל ה-evaluate
    # לצייר עץ ID3 - יש פונ׳ בילט אין - נותן אינטואיציה
    # Ada boost - לתכונות - יש סרטון ביוטיוב לראות להבין איך עובד, זה מותאם רק למסווג של adaboost
    # Ada boost helps understand data about the features

    # kbest = פונ׳ בגוגל - עם פרמטר 40 נגיד, הוא נותן לך את הדיוק עבור 40 התכונות הטובות ביותר.
    # kbest takes the best already

    # צריך לנרמל משהו גם
    # knn נרמול - in google - article