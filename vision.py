N = 10000     # number of neurons
S = 5000     # number of sample points (eval_points)
K = 3        # number of gabors per sample point
width = 75   # width (and height) of the patch
SV = 300    # number of singular values to keep
input_gain = 1 # scaling factor to apply to input (to speed up integration)

import platform
if platform.system() == 'Windows':
    use_multi = False
else:
    use_multi = True

import gabor
import numpy as np
import scipy.sparse.linalg
import time
start = time.time()


def now():
    return '%7.3f' % (time.time() - start)
print now(), 'generating encoders'
if use_multi:
    encoders = gabor.make_gabors_multi(N, width)
else:
    encoders = gabor.make_gabors(N, width)

print now(), 'generating samples'
samples = np.zeros((S, width * width))
for i in range(K):
    if use_multi:
        samples += gabor.make_gabors_multi(S, width)
    else:
        samples += gabor.make_gabors(S, width)


print now(), 'computing SVD'
U, S, V = scipy.sparse.linalg.svds(encoders.T, k=SV)
basis = U
print 'SV ratio:', np.min(S) / np.max(S)


def compress(original):
    return np.dot(original, basis)

def uncompress(compressed):
    return np.dot(basis, compressed.T).T


print now(), 'defining model'
import nengo
neuron_type = nengo.LIF()   # default is nengo.LIF()

stim_image = gabor.make_gabors(K, width)
stim_image = np.sum(stim_image, axis=0)
stim_image /= np.linalg.norm(stim_image)

import nengo
model = nengo.Network()
with model:
    stim = nengo.Node(lambda t: compress(stim_image) if t < 0.1 else np.zeros(SV))
    ens = nengo.Ensemble(n_neurons=N, dimensions=SV, encoders=compress(encoders), eval_points=compress(samples), neuron_type=neuron_type)
    conn = nengo.Connection(ens, ens, synapse=0.1)
    nengo.Connection(stim, ens, transform=input_gain, synapse=None)
    probe = nengo.Probe(ens, synapse=0.01)

print now(), 'building model'
sim = nengo.Simulator(model)
print 'rmse:', np.linalg.norm(sim.data[conn].solver_info['rmses'])

print now(), 'running model'
sim.run(1)

print now(), 'done'
import pylab
data = uncompress(sim.data[probe])
gabor.plot_gabors(data[::-10])
gabor.plot_gabors(np.array([stim_image]))


similarity = np.dot(data / np.linalg.norm(data, axis=1)[:, None], stim_image)
pylab.figure()
pylab.plot(sim.trange(), similarity)
pylab.xlabel('time (seconds)')
pylab.ylabel('similarity to ideal image')
pylab.show()
