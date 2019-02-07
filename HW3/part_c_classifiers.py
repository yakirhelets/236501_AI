from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
import classifier
import hw3_utils as utils


# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
# --------------------------------------- COMBINED ------------------------------------
# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------

class combined_factory(utils.abstract_classifier_factory):

    def train(self, data, labels):
        '''
        train a classifier
        :param data: a list of lists that represents the features that the classifier will be trained with
        :param labels: a list that represents the labels that the classifier will be trained with
        :return: id3_classifier object
        '''

        result_knn_classifier = one_nn_classifier(1, data, labels)
        result_random_forest_clf = random_forest_classifier()
        result_gaussian_process_clf = gaussian_process_classifier()

        result_random_forest_clf.classifier.fit(data, labels)
        result_gaussian_process_clf.classifier.fit(data, labels)

        combined_clf = combined_classifier((result_knn_classifier, result_random_forest_clf, result_gaussian_process_clf))

        return combined_clf

    def to_string(self):
        return "Combined Classifier"

class combined_classifier(utils.abstract_classifier):

    def __init__(self, tuple_clfs):

        self.classifier1 = tuple_clfs[0]
        self.classifier2 = tuple_clfs[1]
        self.classifier3 = tuple_clfs[2]

    def classify(self, features):
        '''
        classify a new set of features
        :param features: the list of features to classify
        :return: a tagging of the given features (1 or 0)
        '''

        true_counter = 0
        false_counter = 0

        result1 = self.classifier1.classify(features)
        result2 = self.classifier2.classify(features)
        result3 = self.classifier3.classify(features)

        results = [result1, result2, result3]

        for r in results:
            if r:
                true_counter += 1
                if true_counter == 2:
                    return 1
            else:
                false_counter += 1
                if false_counter == 2:
                    return 0


# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
# ---------------------------------------KNN-------------------------------------------
# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------


class one_nn_factory(utils.abstract_classifier_factory):

    def train(self, data, labels):
        '''
        train a classifier
        :param data: a list of lists that represents the features that the classifier will be trained with
        :param labels: a list that represents the labels that the classifier will be trained with
        :return: knn_classifier object
        '''
        # No training is occuring here because knn's training is being performed in the classification part
        result_knn_classifier = one_nn_classifier(1, data, labels)

        return result_knn_classifier

    def to_string(self):
        return "1-NN Classifier"

class one_nn_classifier(utils.abstract_classifier):

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


# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
# ---------------------------------------PERCEPTRON------------------------------------
# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
class perceptron_factory(utils.abstract_classifier_factory):

    def train(self, data, labels):
        '''
        train a classifier
        :param data: a list of lists that represents the features that the classifier will be trained with
        :param labels: a list that represents the labels that the classifier will be trained with
        :return: id3_classifier object
        '''

        result_perceptron_classifier = perceptron_classifier()

        result_perceptron_classifier.classifier.fit(data, labels)

        return result_perceptron_classifier

    def to_string(self):
        return "Perceptron Classifier"

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


# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
# ---------------------------------------GAUSSIAN PROCESS------------------------------
# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------


class gaussian_process_factory(utils.abstract_classifier_factory):

    def train(self, data, labels):
        '''
        train a classifier
        :param data: a list of lists that represents the features that the classifier will be trained with
        :param labels: a list that represents the labels that the classifier will be trained with
        :return: knn_classifier object
        '''
        result_gaussian_process_clf = gaussian_process_classifier()

        result_gaussian_process_clf.classifier.fit(data, labels)

        return result_gaussian_process_clf

    def to_string(self):
        return "Gaussian Process Classifier"

class gaussian_process_classifier(utils.abstract_classifier):

    def __init__(self):
        self.classifier = GaussianProcessClassifier()

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

# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
# ---------------------------------------ID3-------------------------------------------
# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
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

    def to_string(self):
        return "ID3 Classifier"

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

# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
# ---------------------------------------RANDOM FOREST---------------------------------
# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------

class random_forest_factory(utils.abstract_classifier_factory):

    def train(self, data, labels):
        '''
        train a classifier
        :param data: a list of lists that represents the features that the classifier will be trained with
        :param labels: a list that represents the labels that the classifier will be trained with
        :return: id3_classifier object
        '''

        result_random_forest_clf = random_forest_classifier()
        result_random_forest_clf.classifier.fit(data, labels)

        return result_random_forest_clf

    def to_string(self):
        return "Random Forest Classifier"

class random_forest_classifier(utils.abstract_classifier):

    def __init__(self):
        self.classifier = RandomForestClassifier(n_estimators=10)  # 10 random classifiers

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
