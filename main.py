import math
import random
import numpy as np
import matplotlib.pyplot as plt
import csv

# Initializations
current_displacement = 0
current_acceleration = 0
# acceleration_dict holds the value of acceleration at different time steps.
acceleration_dict = {
    0: 0,
    5: 2,
    10: 8,
    20: -2,
    40: 5,
    45: 9,
    60: -3,
    85: 0
}

true_values = []

file = open("shipdata.csv")
csvreader = csv.reader(file)
for row in csvreader:
    true_values.append((float(row[1]), float(row[2]), float(row[3])))

true_values = np.array(true_values)
values = np.array(true_values)
values = values[::100]

# Gaussian/Normal noise
# 1st argument = mean
# 2nd argument = standard deviation
# 3rd argument = how many values
latNoise = np.random.normal(0, 0.03, len(values))
lonNoise = np.random.normal(0, 0.01, len(values))

values[:, 0] = latNoise + values[:, 0]
values[:, 1] = lonNoise + values[:, 1]


# initialization
x_k = np.asarray([55, 11])  # first estimate. elem 0 = lat, elem 1 = lon
Q = np.asarray([[0.001, 0.001], [0.001, 0.001]])  # Estimate error covariance
A = np.asarray([[1, 0], [0, 1]])  # Transition matrix.
# Displacement is updated with prev disp + curr vel while velocity is updated with prev vel
# (assuming we have no knowledge about the acceleration)
R = np.asarray([[0.4, 0.01], [0.04, 0.01]])  # Measurement error.
# This is higher than estimation error since we know our measurement contains a lot of noises.

H = np.asarray([[1, 0], [0, 1]])  # Observation matrix. We want every state from our state vector.
P = np.asarray([[0, 0], [0, 0]])  # Error matrix.

estimation = []

for k_loop in range(len(values)):
    # z_k is the measurement at every step
    z_k = np.asarray([values[k_loop][0], values[k_loop][1]])

    x_k = A.dot(x_k)  # predict estimate
    P = (A.dot(P)).dot(A.T) + Q  # predict error covariance

    K = (P.dot(H.T)).dot(np.linalg.inv((H.dot(P).dot(H.T)) + R))  # update Kalman Gain
    x_k = x_k + K.dot((z_k - H.dot(x_k)))  # update estimate

    P = (np.identity(2) - K.dot(H)).dot(P)  # update error covariance

    estimation.append((x_k[0], x_k[1]))  # append the estimations

fig, ax = plt.subplots()

estimation = np.array(estimation)

ax.scatter(true_values[:, 0], true_values[:, 1], c="red", s=0.5)

# ax.plot(true_values[:, 0], true_values[:, 1], 'bo', linestyle="--")


# plt.scatter(estimation[:, 0], estimation[:, 1], c="g", s=0.7)

plt.ylabel("Lon")
plt.xlabel("Lat")

fig2, ax = plt.subplots()
ax.scatter(values[:, 0], values[:, 1], c="green", s=1)
# ax.scatter(estimation[:, 0], estimation[:, 1], c="g", s=0.7)

plt.ylabel("Lon")
plt.xlabel("Lat")

fig3, ax = plt.subplots()
ax.scatter(estimation[:, 0], estimation[:, 1], c="b", s=0.7)
plt.ylabel("Lon")
plt.xlabel("Lat")

plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
