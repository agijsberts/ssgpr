import numpy
from scipy.optimize import check_grad
import ssgpr

def dataset(m, noise=0.1):
    def f(x):
        return numpy.column_stack((x[:,0] * numpy.sin(x[:,1]), 
                                   x[:,0] * numpy.cos(x[:,1])))

    n, p = 2, 2 
    x = numpy.random.randn(m, n)
    y = f(x) + numpy.random.randn(m, p) * noise

    return x, y


def test_gradient():
    x, y = dataset(100)
    n, p = x.shape[1], y.shape[1]
    nproj = 50

    machine = ssgpr.SSGPR(n, p, nproj=nproj, fixed=False)
    machine.guessparams(x, y)
    x0 = machine.getparams()
    
    def f(params):
        machine.setparams(params)
        machine.train(x, y)
        return machine.lmlfunc()

    def df(params):
        machine.setparams(params)
        machine.train(x, y)
        return machine.lmlgradient()
        
    l2err = check_grad(f, df, x0)
    numpy.testing.assert_almost_equal(l2err / numpy.linalg.norm(df(x0)), 0)


def test_incremental():
    trainx, trainy = dataset(2000)
    n, p = trainx.shape[1], trainy.shape[1]
    nproj = 50
    seed = 1

    numpy.random.seed(seed)
    batch = ssgpr.SSGPR(n, p, nproj=nproj)
    batch.train(trainx, trainy)

    numpy.random.seed(seed)
    incremental = ssgpr.SSGPR(n, p, nproj=nproj)
    for x, y in zip(trainx, trainy):
        incremental.update(x, y)

    numpy.testing.assert_almost_equal(batch.L, incremental.L, decimal=10)
    numpy.testing.assert_almost_equal(batch.W, incremental.W, decimal=10)

def test_err_fixed():
    noise = 0.1
    x, y = dataset(1000, noise)
    n, p = x.shape[1], y.shape[1]
    nproj = 50

    (trainx, testx), (trainy, testy) = numpy.split(x, 2), numpy.split(y, 2)

    machine = ssgpr.SSGPR(n, p, nproj=nproj, fixed=True)
    machine.guessparams(trainx, trainy)
    machine.optimize(trainx, trainy, maxIter=250)
    numpy.testing.assert_almost_equal(machine.sigma_n, noise, decimal=2)

    machine.train(trainx, trainy)

    predy = numpy.array([machine.predict(x)[0] for x in testx])
    rmse = numpy.sqrt(((predy - testy)**2).mean(0))
    numpy.testing.assert_almost_equal(rmse, [noise, noise], decimal=2)


def test_err_nofixed():
    noise = 0.1
    x, y = dataset(1000, noise)
    n, p = x.shape[1], y.shape[1]
    nproj = 50

    (trainx, testx), (trainy, testy) = numpy.split(x, 2), numpy.split(y, 2)

    machine = ssgpr.SSGPR(n, p, nproj=nproj, fixed=False)
    machine.guessparams(trainx, trainy)
    machine.optimize(trainx, trainy, maxIter=250)
    numpy.testing.assert_almost_equal(machine.sigma_n, noise, decimal=2)

    machine.train(trainx, trainy)

    predy = numpy.array([machine.predict(x)[0] for x in testx])
    rmse = numpy.sqrt(((predy - testy)**2).mean(0))
    numpy.testing.assert_almost_equal(rmse, [noise, noise], decimal=2)


