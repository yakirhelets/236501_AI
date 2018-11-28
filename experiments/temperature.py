import numpy as np
from matplotlib import pyplot as plt

X = np.array([400, 450, 900, 390, 550])

N = 5
num_of_experiments = 100
start_point = 0.01
end_point = 5.0

T = np.linspace(start_point, end_point, num_of_experiments, True, False, float)

alpha = min(X)
P = [[0 for x in range(X.size)] for y in range(T.size)]

print(P)

def calc_probability(T, xi, alpha, X):
    numerator = (xi / alpha)**(-1 / T)
    denominator = 0

    for x in X:
        denominator += (x / alpha)**(-1 / T)

    return float(numerator / denominator)


for i in range(len(X)):
    xi = X[i]
    for j in range(len(T)):
        P[j][i] = calc_probability(T[j], xi, alpha, X)


print(P)

for i in range(len(X)):
    plt.plot(T, P[:, i], label=str(X[i]))

plt.xlabel("T")
plt.ylabel("P")
plt.title("Probability as a function of the temperature")
plt.legend()
plt.grid()
plt.show()
exit()


###
P = [[0 for x in range(5)] for y in range(100)]
for i in range(5):
    P[:, i]

