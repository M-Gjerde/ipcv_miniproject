from numpy import dot, tile, log, pi, exp
# Initializations
from numpy.linalg import inv, det
import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt
import csv
import cartopy.crs as crs
import cartopy.feature as feat

"""
Based on paper:
https://arxiv.org/ftp/arxiv/papers/1204/1204.0375.pdf
Code is also from there but adapted to our AIS data.
"""


def gauss_pdf(X, M, S):
    if M.shape[1] == 1:
        DX = X - tile(M, X.shape[1])
        E = 0.5 * np.sum(DX * (dot(inv(S), DX)), axis=0)
        E = E + 0.5 * M.shape[0] * log(2 * pi) + 0.5 * log(det(S))
        P = exp(-E)
    elif X.shape[1] == 1:
        DX = tile(X, M.shape[1]) - M
        E = 0.5 * np.sum(DX * (dot(inv(S), DX)), axis=0)
        E = E + 0.5 * M.shape[0] * log(2 * pi) + 0.5 * log(det(S))
        P = exp(-E)
    else:
        DX = X - M
        E = 0.5 * dot(DX.T, dot(inv(S), DX))
        E = E + 0.5 * M.shape[0] * log(2 * pi) + 0.5 * log(det(S))
        P = exp(-E)
    return (P[0], E[0])


def kf_predict(X, P, A, Q, B, U):
    X = dot(A, X)
    P = dot(A, dot(P, A.T)) + Q
    return (X, P)


def kf_update(X, P, Y, H, R):
    IS = R + dot(H, dot(P, H.T))
    K = dot(P, dot(H.T, np.linalg.inv(IS)))
    X = X + dot(K, (Y - dot(H, X)))
    P = P - dot(K, dot(IS, K.T))
    return X, P, K, IS


# time step of mobile movement
dt = 1
# Initialization of state matrices
X = np.array([[0.0], [0.0]])
P = np.diag((0.01, 0.00))
A = np.array([[1, 0], [0, 1]])
Q = np.eye(X.shape[0])
B = np.eye(X.shape[0])
U = np.zeros((X.shape[0], 1))

# Measurement matrices
Y = np.array([[X[0, 0] + abs(randn(1)[0])], [X[1, 0] + abs(randn(1)[0])]])
H = np.array([[1, 0], [0, 1]])
R = np.eye(Y.shape[0])

true_values = []
file = open("shipdata.csv")
csvreader = csv.reader(file)
for row in csvreader:
    true_values.append((float(row[1]), float(row[2]), float(row[3])))

true_values = np.array(true_values)
values = np.array(true_values)
values = values[:, 0:2]
nth = 10
values = values[::nth]

# Gaussian/Normal noise
# 1st argument = mean
# 2nd argument = standard deviation
# 3rd argument = how many values
latNoise = np.random.normal(0, 0.05, len(values))
lonNoise = np.random.normal(0, 0.02, len(values))
values[:, 0] = latNoise + values[:, 0]
values[:, 1] = lonNoise + values[:, 1]

predictions = []
measurements = []
# Number of iterations in Kalman Filter
# Applying the Kalman Filter
fig2 = plt.figure(2)

# ## Use 10 points to upate the initial state matrices.
for x in range(10):
    Y = np.array([[true_values[x, 0], true_values[x, 1]]])
    (X, P) = kf_predict(X, P, A, Q, B, U)
    (X, P, K, IS) = kf_update(X, P, Y, H, R)

# ## RUN KALMAN FILTER ON INPUT DATA
for i in np.arange(0, len(values)):
    Y = np.array([[values[i, 0], values[i, 1]]])
    (X, P) = kf_predict(X, P, A, Q, B, U)
    (X, P, K, IS) = kf_update(X, P, Y, H, R)
    # Y = np.array([[X[0, 0] + values[i, 0], X[1, 0] + values[i, 1]]])
    predictions.append((X[0, 0], X[0, 1]))
    measurements.append((Y[0, 0], Y[0, 1]))

# ### MATPLOTLIB COMMANDS
predictions = np.array(predictions)
measurements = np.array(measurements)

fig, axs = plt.subplots(3)

plt.xlabel('Latitude')
plt.ylabel('Longitude')

axs[0].plot(predictions[:, 0], predictions[:, 1], c="r", label="Predicted values")  # s=10, alpha=0.9, )

axs[0].set_xlim([54.5, 56.7])
axs[0].set_ylim([11, 18])

axs[0].legend()
axs[0].grid(True)

axs[1].plot(measurements[:, 0], measurements[:, 1], c="b", label="Measured values")

axs[1].set_xlim([54.5, 56.7])
axs[1].set_ylim([11, 18])

axs[1].legend()
axs[1].grid(True)


axs[2].scatter(true_values[:, 0], true_values[:, 1], c="b", s=10, alpha=1, label="True values")

axs[2].set_xlim([54.5, 56.7])
axs[2].set_ylim([11, 18])

axs[2].legend()
axs[2].grid(True)

# ## CALCULATE ERRORS BETWEEN PREDICTED AND MEASURED COMPARED TO TRUE VALUES

true_values = true_values[:, 0:2]
true_values = true_values[::nth]

predictError = (np.square(true_values[:,0:1] - predictions[:,0:1])).mean(axis=0)
measurementError = (np.square(true_values[:,0:1] - measurements[:,0:1])).mean(axis=0)

#predictError = true_values - predictions
#measurementError = true_values - measurements

fig2, axs2 = plt.subplots(1)
series = np.linspace(0, len(predictError), len(predictError))
axs2.set_ylabel("Latitude")
categories = ["MSE prediction", "MSE measurement"]
axs2.bar(categories, [predictError[0], measurementError[0]], color="red")  # s=10, alpha=0.9, )

print([predictError[0], measurementError[0]])
predictError = (np.square(true_values[:,1:2] - predictions[:,1:2])).mean(axis=0)
measurementError = (np.square(true_values[:,1:2] - measurements[:,1:2])).mean(axis=0)
print([predictError[0], measurementError[0]])

fig4, axs4 = plt.subplots(1)
series = np.linspace(0, len(predictError), len(predictError))
axs4.set_ylabel("Longitude")
categories = ["MSE prediction", "MSE measurement"]
axs4.bar(categories, [predictError[0], measurementError[0]], color="red")  # s=10, alpha=0.9, )

#axs2[0].set_ylim([-0.15, 0.15])

axs4.legend()


fig3 = plt.figure(figsize=(8, 5))
ax = plt.axes(projection=crs.Mollweide())

ax.set_global()
ax.add_feature(feat.LAND, zorder=100, edgecolor='k', alpha=0.8)
ax.coastlines()

ax.scatter(true_values[:, 1], true_values[:, 0], c="b",transform=crs.PlateCarree(), s=2, alpha=1, label="True values")

fig4 = plt.figure(figsize=(8, 5))
ax = plt.axes(projection=crs.Mollweide())

ax.set_global()
ax.add_feature(feat.LAND, zorder=100, edgecolor='k', alpha=0.8)
ax.coastlines()

ax.scatter(values[:, 1], values[:, 0], c="b",transform=crs.PlateCarree(), s=2, alpha=1, label="True values")


fig5 = plt.figure(figsize=(8, 5))
ax = plt.axes(projection=crs.Mollweide())

ax.set_global()
ax.add_feature(feat.LAND, zorder=100, edgecolor='k', alpha=0.8)
ax.coastlines()

ax.scatter(predictions[:, 1], predictions[:, 0], c="r",transform=crs.PlateCarree(), s=2, alpha=1, label="True values")

plt.show()
