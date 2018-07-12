# Import libraries

from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
pi = np.pi
from brian2 import *

#Create some data

from coding import single_response_frequency
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
st2 = st1 + 1.
n0, n1 = 0, 1  # choose two neurons
npoints = 100

stims1 = st1*np.ones(npoints)
resp0 = single_response_frequency(stims1, n0)
resp1 = single_response_frequency(stims1, n1)
plt.scatter(resp0, resp1, alpha=0.4,
            label="Stimulus orientation {} rad".format(st1))

stims2 = st2*np.ones(npoints)
resp0 = single_response_frequency(stims2, n0)
resp1 = single_response_frequency(stims2, n1)
plt.scatter(resp0, resp1, alpha=0.4,
            label="Stimulus orientation {} rad".format(st2))

plt.ylim([0, 100])
plt.xlim([0, 100])
plt.xlabel("Response of neuron {}".format(n0))
plt.ylabel("Response of neuron {}".format(n1))
plt.legend()

plt.gca().set_aspect("equal")
show()

# Now add trivial noise correlations

noise_intensity = 10
angle = pi/4

noise = np.random.normal(0, noise_intensity, size=(2, npoints))

resp0toS1 = single_response_frequency(stims1, n0) + noise[0]*np.sin(angle)
resp1toS1 = single_response_frequency(stims1, n1) + noise[0]*np.cos(angle)
plt.scatter(resp0toS2, resp1toS2, alpha=0.4,
            label="Stimulus orientation {} rad".format(st1))

resp0toS2 = single_response_frequency(stims2, n0) + noise[1]*np.sin(angle)
resp1toS2 = single_response_frequency(stims2, n1) + noise[1]*np.cos(angle)
plt.scatter(resp0toS2, resp1toS2, alpha=0.4,
            label="Stimulus orientation {} rad".format(st2))

plt.ylim([0, 120])
plt.xlim([0, 120])
plt.xlabel("Response of neuron {}".format(n0))
plt.ylabel("Response of neuron {}".format(n1))
plt.legend();

plt.gca().set_aspect("equal")
show()
