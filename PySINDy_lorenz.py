'''Import libraries'''
import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy import arange

''' Generate training data'''
#Define equations for Lorenz system
def lorenz(x, t):
    return [
        10 * (x[1] - x[0]),
        x[0] * (28 - x[2]) - x[1],
        x[0] * x[1] - 8 / 3 * x[2],
    ]
#Intial conditions for x and t
dt = 0.001
t_train = np.arange(0, 100, dt)
x0_train = [-8, 8, 27]
#Collect time-series data from Lorenz system (X)
x_train = odeint(lorenz, x0_train, t_train)
#Compute derivates of data collected (X')
x_dot_train_measured = np.array(
    [lorenz(x_train[i], 0) for i in range(t_train.size)]
)

''' Fit the models and simulate'''
#Set parameters for SINDy object
poly_order = 5
threshold = 0.05
order = 2
#Define some noise levels to introduce into data
noise_levels = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
models = []
t_sim = np.arange(0, 20, dt)
x_sim = []
for eps in noise_levels:
    #Create SINDy object
    model = ps.SINDy(
        #Define numerical differentiation method to compute X' from X
        differentiation_method =ps.FiniteDifference (order=order),
        #Provides a set of sparse regression solvers for determining  Ξ (contains sparse vectors)
        optimizer=ps.STLSQ(threshold=threshold),
        #Constructs set of library functions and handles formation of Θ(X)
        feature_library=ps.PolynomialLibrary(degree=poly_order),
    )
    #Fit SINDy model to training data generated earlier
    model.fit(
        x_train,
        t=dt,
        x_dot=x_dot_train_measured
        + np.random.normal(scale=eps, size=x_train.shape),
        quiet=True,
    )
    models.append(model)
    x_sim.append(model.simulate(x_train[0], t_sim))

''' Plot results'''
fig = plt.figure(figsize=(15, 4))
#Full simulation of system
ax = fig.add_subplot(131, projection="3d")
ax.plot(
    x_train[: t_sim.size, 0],
    x_train[: t_sim.size, 1],
    x_train[: t_sim.size, 2],
)
plt.title("full simulation")
ax.set(xlabel="$x$", ylabel="$y$", zlabel="$z$")
#Identified system (noise level = 0.01)
model_idx = 2
ax = fig.add_subplot(132, projection="3d")
ax.plot(x_sim[model_idx][:, 0], x_sim[model_idx][:, 1], x_sim[model_idx][:, 2])
plt.title(f"identified system, $\eta$={noise_levels[model_idx]:.2f}")
ax.set(xlabel="$x$", ylabel="$y$", zlabel="$z$")
#Identified system (noise level = 1)
model_idx = 4
ax = fig.add_subplot(133, projection="3d")
ax.plot(x_sim[model_idx][:, 0], x_sim[model_idx][:, 1], x_sim[model_idx][:, 2])
plt.title(f"identified system, $\eta$={noise_levels[model_idx]:.2f}")
ax.set(xlabel="$x$", ylabel="$y$", zlabel="$z$")

fig = plt.figure(figsize=(12, 5))
#2D plot for x and y (noise level= 0.01)
model_idx = 2
ax = fig.add_subplot(221)
ax.plot(t_sim, x_train[: t_sim.size, 0], "r",label='true')
ax.plot(t_sim, x_sim[model_idx][:, 0], "k--",label='model')
ax.legend()
plt.title(f"$\eta$={noise_levels[model_idx]:.2f}")
plt.ylabel("x")

ax = fig.add_subplot(223)
ax.plot(t_sim, x_train[: t_sim.size, 1], "r", label='true')
ax.plot(t_sim, x_sim[model_idx][:, 1], "k--", label='model')
ax.legend()
plt.xlabel("time")
plt.ylabel("y")
#2D plot for x and y (noise level= 1)
model_idx = 4
ax = fig.add_subplot(222)
ax.plot(t_sim, x_train[: t_sim.size, 0], "r",label='true')
ax.plot(t_sim, x_sim[model_idx][:, 0], "k--", label='model')
ax.legend()
plt.title(f"$\eta$={noise_levels[model_idx]:.2f}")
plt.ylabel("x")

ax = fig.add_subplot(224)
ax.plot(t_sim, x_train[: t_sim.size, 1], "r", label='true')
ax.plot(t_sim, x_sim[model_idx][:, 1], "k--", label='model')
ax.legend()
plt.xlabel("time")
plt.ylabel("y")

plt.show()

'''Assess results on a test trajectory'''
#New initial conditions for x and t
t_test = np.arange(0, 15, dt)
x0_test = np.array([8, 7, 15])
#Generate test data
x_test = odeint(lorenz, x0_test, t_test)
# Compare SINDy-predicted derivatives with finite difference derivatives
print('Model score: %f' % model.score(x_test, t=dt))
# Predict derivatives using the learned model
x_dot_test_predicted = model.predict(x_test)
# Compute derivatives with a finite difference method in model
x_dot_test_computed = model.differentiate(x_test, t=dt)
#Plot results
fig, axs = plt.subplots(x_test.shape[1], 1, sharex=True, figsize=(7, 9))
for i in range(x_test.shape[1]):
    axs[i].plot(t_test, x_dot_test_computed[:, i],
                'k', label='model derivative')
    axs[i].plot(t_test, x_dot_test_predicted[:, i],
                'r--', label='model prediction')
    axs[i].legend()
    axs[i].set(xlabel='t', ylabel='$\dot x_{}$'.format(i))
plt.show()

# Evolve the new initial conditions in time with the SINDy model
x_test_sim = model.simulate(x0_test, t_test)
#Plot results
fig, axs = plt.subplots(x_test.shape[1], 1, sharex=True, figsize=(7, 9))
for i in range(x_test.shape[1]):
    axs[i].plot(t_test, x_test[:, i], 'k', label='true simulation')
    axs[i].plot(t_test, x_test_sim[:, i], 'r--', label='model simulation')
    axs[i].legend()
    axs[i].set(xlabel='t', ylabel='$x_{}$'.format(i))

fig = plt.figure(figsize=(10, 4.5))
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(x_test[:, 0], x_test[:, 1], x_test[:, 2], 'k')
ax1.set(xlabel='$x_0$', ylabel='$x_1$',
        zlabel='$x_2$', title='true simulation')

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot(x_test_sim[:, 0], x_test_sim[:, 1], x_test_sim[:, 2], 'r--')
ax2.set(xlabel='$x_0$', ylabel='$x_1$',
        zlabel='$x_2$', title='model simulation')

plt.show()
