import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.feature_selection import SelectKBest, f_classif
import hw3_utils as utils
import part_c_classifiers
from classifier import split_crosscheck_groups, evaluate

num_folds_num = 10
best_num_list = [1, 10, 30, 80, 100, 130, 187] # How many best features to take
classifier_factory_list = [part_c_classifiers.combined_factory,
                           part_c_classifiers.id3_factory,
                           part_c_classifiers.perceptron_factory,
                            part_c_classifiers.random_forest_factory,
                           part_c_classifiers.one_nn_factory,
                           part_c_classifiers.gaussian_process_factory]
patients, labels, test = utils.load_data()

#--------------- Function to present the graphs ------------------

def present_graphs(xlist, ylist):
    num_of_subsets = 7  # Change this according to the number of the subsets
    classfier_names = []
    i=1
    for tuple in ylist:
        classfier_names.append(tuple[0])
        plt.figure(i)
        for j in range(num_of_subsets):
            plt.plot(xlist, tuple[1])
        plt.xlabel('K Best')
        plt.ylabel('Accuracy')
        plt.title("Accuracy for different K-Best values for " + str(tuple[0]))
        plt.show()
        i+=1
    # Present all the result altogether
    i=0
    plt.figure(num_of_subsets)
    for tuple in ylist:
        plt.plot(xlist, tuple[1])
        plt.legend(classfier_names)
        i += 0
    plt.xlabel('K-Best')
    plt.ylabel('Accuracy')
    plt.title("Accuracy for different K-Best values for all classifiers")
    plt.show()


#--------------- Produce accuracy rates and graphs ------------------

clf_names_and_results = []

for c in classifier_factory_list:

    accuracy_list_for_classifier = []
    classifier_name = c().to_string()

    for v in best_num_list:
        selector = SelectKBest(score_func=f_classif, k=v)
        selector.fit(patients, labels)
        newData = selector.transform(patients)
        split_crosscheck_groups((newData, labels), num_folds_num)
        # Creating a classifier factory
        clf = c()
        accuracy, error = evaluate(clf, num_folds_num)
        accuracy_list_for_classifier.append(accuracy)

    clf_names_and_results.append((classifier_name, accuracy_list_for_classifier))

    # accuracy_list_for_all_classifiers.append(accuracy_list_for_classifier)

present_graphs(best_num_list, clf_names_and_results)


# --------------------- Code that produces the dot file for the ID3 tree presented in the report ---------------------
clf = tree.DecisionTreeClassifier()
clf = clf.fit(patients, labels)
tree.export_graphviz(clf, out_file='tree.dot')




