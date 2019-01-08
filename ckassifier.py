# task 1
import numpy as np

def euclidian_distance(xList, yList) :
    dist = 0
    for x, y in zip(xList, yList):
        dist += (x-y)**2
    return np.sqrt(dist)

# TEST FOR TASK 1
# list1 = [1,2,3,4,5,6,7]
# list2 = [7,6,5,4,3,2,1]
#
# print(euclidian_distance(list1,list2))
