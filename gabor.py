import numpy as np

def make_gabors(N, width, rng=None):
    if rng is None:
        rng = np.random.RandomState()
    lambd = rng.uniform(0.3, 0.8, N)
    theta = rng.uniform(0, 2 * np.pi, N)
    psi = rng.uniform(0, 2 * np.pi, N)
    sigma = rng.uniform(0.2, 0.5, N)
    gamma = rng.uniform(0.4, 0.8, N)
    x_offset = rng.uniform(-1, 1, N)
    y_offset = rng.uniform(-1, 1, N)
    
    x = np.linspace(-1, 1, width)
    X, Y = np.meshgrid(x, x)
    X.shape = width * width
    Y.shape = width * width
    
    X = X[None,:] - x_offset[:,None]
    Y = Y[None,:] + y_offset[:,None]
    
    cosTheta = np.cos(theta)
    sinTheta = np.sin(theta)
    xTheta = X * cosTheta[:,None]  + Y * sinTheta[:,None]
    yTheta = -X * sinTheta[:,None] + Y * cosTheta[:,None]
    
    e = np.exp( -(xTheta**2 + yTheta**2 * gamma[:,None]**2) / (2 * sigma[:,None]**2) )
    cos = np.cos(2 * np.pi * xTheta / lambd[:,None] + psi[:,None])    
    gabor = e * cos
    
    gabor = gabor / np.linalg.norm(gabor, axis=1)[:, None]
    return gabor

def make_gabors_multi(N, width, processors=None):
    import multiprocessing

    if processors is None:
        processors = multiprocessing.cpu_count()
    from functools import partial
    pool = multiprocessing.Pool(processors)

    N_multi = N / processors
    if N_multi * processors < N:
        N_multi += 1

    result = pool.map(partial(make_gabors, width=width), 
                      [N_multi] * processors)
    pool.close()
    return np.vstack(result)[:N]

import pylab
def plot_gabors(data, columns=None):
    if columns is None:
        columns = int(np.sqrt(len(data)))
    pylab.figure(figsize=(10,10))
    vmin = np.min(data)
    vmax = np.max(data)
    width = int(np.sqrt(data.shape[1]))
    for i, d in enumerate(data):
        w = columns - 1 - (i % columns)
        h = i / columns            
        d.shape = width, width
        pylab.imshow(d, extent=(w+0.025, w+0.975, h+0.025, h+0.975), 
                 interpolation='none', vmin=vmin, vmax=vmax, cmap='gray')
        pylab.xticks([])
        pylab.yticks([])
    pylab.xlim((0, columns))
    pylab.ylim((0, len(data) / columns))

if __name__ == '__main__':
    
    import time
    start = time.time()
    a = make_gabors_multi(N=30000, width=75)
    #a = make_gabors(N=1000, width=75)

    end = time.time()
    print end - start

    print a.shape
    

    '''
    import multiprocessing
    pool = multiprocessing.Pool(processes=5)

    from functools import partial

    a = pool.map(partial(make_gabors, width=75), [1000, 1000, 1000])
    '''
