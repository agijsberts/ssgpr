import numpy as np
from ssgpr import SSGPR

# create non-linear data with 2-dim inputs and 2-dim outputs, corrupted with 
# Gaussian noise with std.dev. 0.1
def f(x):
    return np.column_stack((x[:,0] * np.sin(x[:,1]), x[:,0] * np.cos(x[:,1])))

m, n, p = 1000, 2, 2                     
x = np.random.randn(m, n)
y = f(x) + np.random.randn(m, p) * 0.1
(trainx, testx), (trainy, testy) = np.split(x, 2), np.split(y, 2)

# construct SSGPR, guess decent hyperparameters and then optimize them
machine = SSGPR(n, p, nproj=100)
machine.guessparams(trainx, trainy)
machine.optimize(trainx, trainy, maxIter=250)
print('est. noise:', machine.sigma_n)

# train the machine with optimal hyperparameters
machine.train(trainx, trainy)

# online test with incremental updates
predy = np.array([machine.update(x, y)[0] for x, y in zip(testx, testy)])
print('RMSE:', np.sqrt(((predy - testy)**2).mean(0)))
