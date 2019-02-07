import matplotlib.pyplot as plt
import hw3_utils as utils
import part_c_classifiers
from classifier import id3_factory, perceptron_factory
from classifier import split_crosscheck_groups, knn_factory, evaluate
from sklearn.feature_selection import SelectKBest, f_classif



# question 3.2

patients, labels, test = utils.load_data()
split_crosscheck_groups([patients, labels], 2)


# question 5.1

k_list = [1,3,5,7,13]
accuracy_list = []

file_name = 'experiments6.csv'
with open(file_name, 'wb') as file:
    for k in k_list:
        knn_f = knn_factory(k)
        accuracy, error = evaluate(knn_f, 2)
        line = str(k) + "," + str(accuracy) + "," + str(error) + "\n"
        accuracy_list.append(accuracy)
        file.write(line.encode())

# question 5.2

plt.plot(k_list, accuracy_list)
plt.xlabel('K value')
plt.ylabel('Average accuracy')
plt.title('Part B, question 5.2')
plt.show()

# questions 7.1, 7.2

file_name = 'experiments12.csv'
with open(file_name, 'wb') as file:
    # ID3 RUN
    id3_f = id3_factory()
    accuracy, error = evaluate(id3_f, 2)
    line = "1" + "," + str(accuracy) + "," + str(error) + "\n"
    file.write(line.encode())

    # Perceptron RUN
    perceptron_f = perceptron_factory()
    accuracy, error = evaluate(perceptron_f, 2)
    line = "2" + "," + str(accuracy) + "," + str(error) + "\n"
    file.write(line.encode())



# part C submission classifier

patients, labels, test = utils.load_data()

# create the factory
one_nn_f = part_c_classifiers.one_nn_factory()
# reduce the features
selector = SelectKBest(score_func=f_classif, k=130)
selector.fit(patients, labels)
newData = selector.transform(patients)
# train the algorithm with the new features
one_nn_clf = one_nn_f.train(newData, labels)
# write prediction of 300 in test to file
results = []
for t in test:
    results.append(one_nn_clf.classify(t))
utils.write_prediction(results)

