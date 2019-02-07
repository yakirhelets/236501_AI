import hw3_utils as utils
import part_c_classifiers
from classifier import split_crosscheck_groups, evaluate

num_folds_num = 10
best_num_list = [1, 10, 30, 80, 100, 130, 187] # How many best features to take
classifier_factory_list = [part_c_classifiers.id3_factory, part_c_classifiers.perceptron_factory,
                    part_c_classifiers.random_forest_factory, part_c_classifiers.knn_factory]
patients, labels, test = utils.load_data()


#--------------- KNN ------------------

for c in classifier_factory_list:

    accuracy_list_for_classifier = []

    for v in best_num_list:
        newData = 0#something with v that uses patients
        newLabels = 0#something with v that uses labels
        split_crosscheck_groups((newData, newLabels), num_folds_num)
        clf = c() # Creating a classifier factory
        accuracy, error = evaluate(clf, num_folds_num)
        accuracy_list_for_classifier.append(accuracy)
        # CALL yakir's function to print a graph with accuracy_list_for_classifier

