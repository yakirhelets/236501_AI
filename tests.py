import hw3_utils as utils
import classifier


# TEST for question 1
list1 = [1,2,3,4,5,6,7]
list2 = [7,6,5,4,3,2,1]

print(classifier.euclidean_distance(list1, list2))


# TEST for question 3.2

data = utils.load_data()
classifier.split_crosscheck_groups(data, 2)
print(classifier.load_k_fold_data(1)[1][0])