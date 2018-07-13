# Import libraries

from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import time
pi = np.pi
from brian2 import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA # "Support vector classifier"
model = LDA()

# Create some data

from coding import single_response_frequency





def plot_decision_function(model, ax=None, color='black'):
    """Plot the decision function for a 2D LDA"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors=color,
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    
    ax.set_xlim(xlim)

N=20 # number of neurons
n_stim = 100 # number of stimuli
stimuli = np.linspace(0, 2*pi, n_stim) # create 100 stimuli from 0-2pi
n_trials = 200

responses = np.empty([N, n_trials, n_stim])

for n in range(N):
    for i in range(n_trials):
        responses[n, i] = single_response_frequency(stimuli, neuronid=n)

# average responses for plotting if we wish
avg_responses = responses.mean(axis=1)

# preferred orientations of each neuron
preferred_orientations = stimuli[np.argmax(avg_responses, axis=1)] % pi

# Without correlations

st1 = 2.  # orientation of the two stimuli
st2 = st1 +1
n0, n1 = 0, 1  # choose two neurons
npoints = 100
stims1 = st1*np.ones(npoints)
resp0_1 = single_response_frequency(stims1, n0)
resp1_1 = single_response_frequency(stims1, n1)
plt.figure(1)
plt.scatter(resp0_1, resp1_1, alpha=0.4,
        label="Stimulus orientation {} rad".format(st1))
X_1 = np.column_stack((resp0_1, resp1_1))
Y_1 = np.zeros((len(resp0_1)))
stims2 = st2*np.ones(npoints)
resp0_2 = single_response_frequency(stims2, n0)
resp1_2 = single_response_frequency(stims2, n1)
plt.scatter(resp0_2, resp1_2, alpha=0.4,
        label="Stimulus orientation {} rad".format(st2))
X_2 = np.column_stack((resp0_2, resp1_2))
Y_2 = np.ones((len(resp0_1)))
X = np.concatenate((X_1, X_2))
Y = np.concatenate((Y_1, Y_2))
#print(X)
plt.ylim([0, 100])
plt.xlim([0, 100])
plt.xlabel("Response of neuron {}".format(n0))
plt.ylabel("Response of neuron {}".format(n1))
plt.legend()

plt.gca().set_aspect("equal")

model.fit(X, Y)
plot_decision_function(model)
# Now add trivial noise correlations

noise_intensity = 10
angle = pi/8
for i in range(17):
	model1 = LDA();
	noise = np.random.normal(0, noise_intensity, size=(2, npoints))
	plt.figure(2)
	resp0toS1 = single_response_frequency(stims1, n0) + noise[0]*np.sin(i * angle)
	resp1toS1 = single_response_frequency(stims1, n1) + noise[0]*np.cos(i * angle)
	plt.scatter(resp0toS1, resp1toS1, alpha=0.4,
	            label="Stimulus orientation {} rad".format(i*angle))

	resp0toS2 = single_response_frequency(stims2, n0) + noise[1]*np.sin(i * angle)
	resp1toS2 = single_response_frequency(stims2, n1) + noise[1]*np.cos(i * angle)
	plt.scatter(resp0toS2, resp1toS2, alpha=0.4,
	            label="Stimulus orientation {} rad".format(i*angle))
	X_n1 = np.column_stack((resp0toS1, resp1toS1))
	Y_n1 = np.zeros((len(resp0toS1)))
	X_n2 = np.column_stack((resp0toS2, resp1toS2))
	Y_n2 = np.ones((len(resp0toS2)))
	Xn = np.concatenate((X_n1, X_n2))
	Yn = np.concatenate((Y_n1, Y_n2))
	model1.fit(Xn, Yn);

	plt.ylim([0, 120])
	plt.xlim([0, 120])
	plt.xlabel("Response of neuron {}".format(n0))
	plt.ylabel("Response of neuron {}".format(n1))
	plt.legend()
	plt.gca().set_aspect("equal")
	plot_decision_function(model, color = 'red');
	plot_decision_function(model1);
	#print(model1.score(Xn,Yn))
	#print(model.score(Xn,Yn))
	plt.draw()
	plt.pause(2)
	plt.clf()


