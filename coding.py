import numpy as np

print("The coding module was run.")
pi = np.pi
N = 20

np.random.seed(42)
tuning_angles = np.linspace(0, pi, N+1)[:-1]
np.random.shuffle(tuning_angles)

max_rate = 80.
min_rate = 30.


def tuning_function(angle, centre):
    if np.size(angle) == 1:
        angle = np.asarray([angle])
    if np.size(centre) == 1:
        centre = np.asarray([centre])
    diff = angle-centre[:, np.newaxis]
    #distances = np.min([np.abs(diff), pi-np.abs(diff)], axis=0)
    distances = 2*diff
    return np.squeeze(.5 + .5*np.cos(distances))  # cosine


def single_response_frequency(stimulus, neuronid, tuning_angles=tuning_angles):
    tuned = tuning_function(stimulus, tuning_angles[neuronid])
    rates = min_rate + tuned*(max_rate-min_rate)
    nspikes = np.random.poisson(rates)
    return nspikes


def generate_pop_response(stimulus, tuning_angles=tuning_angles):
    n = len(tuning_angles)
    nspikes = [single_response_frequency(stimulus, i, tuning_angles)
               for i in range(n)]
    totspikes = np.sum(nspikes)

    ids = []
    for i in range(n):  # who cares
        for _ in range(nspikes[i]):
            ids.append(i)
    np.random.shuffle(ids)
    times = np.random.random(totspikes)
    times = np.sort(times)
    return times, np.asarray(ids)


secret_stimulus_value = np.random.random()*pi
response_to_secret_stimulus = generate_pop_response(secret_stimulus_value)
