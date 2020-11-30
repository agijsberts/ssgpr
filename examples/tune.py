#!/usr/bin/env python
"""
This script facilitates training, hyperparameter optimization, and testing 
of the Sparse Spectrum Gaussian Process for Regression. In addition, models of 
the algorithm can be exported to a format that can be imported directly in the 
iCub learningMachine framework (see example below).

Notes:
 - Please supply the --help parameter to see an overview of the available 
   args.
 - All outputs dimensions share the same hyperparameter configuration. If this 
   is problematic in your application domain, consider learning different 
   (sets of) outputs individually.
 - By design, the method uses a randomized initialization for the feature 
   mapping. The results may therefore vary depending on the seed of the PRNG.
   However, variance in results will decrease as the number of projections 
   increases.
 - Hyperparameter optimization is performed by minimizing the negative log 
   marginal likelihood. This is a non-convex, non-linear optimization problem 
   and may therefore result in local optima, depending on the initial 
   parameter configuration.


Dependencies: numpy, scipy


Example of using models in the learningMachine framework:
./lmtrain --load ssgpr.model
./lmtransform --trainport /lm/train/train:i --predictport /lm/train/predict:io --load ssf.model
./lmtest --trainport /lm/transform/train:i --predictport /lm/transform/predict:io --inputs '(1 2 ... n)' --outputs '(1 2 ... p)' --datafile dataset.dat


 Copyright (C) 2011 Arjan Gijsberts <arjan@liralab.it>

"THE BEER-WARE LICENSE" (Revision 43):
<arjan@liralab.it> wrote this file. As long as you retain this notice you
can do whatever you want with this stuff. If we meet some day, and you think
this stuff is worth it, you can buy me a beer in return. Arjan Gijsberts
"""

import numpy
import argparse
import time
import sys
import scipy

from ssgpr import SparseSpectrumFeatures, LinearGPR, NoPrior, \
                  NormalDistribution, LogNormalDistribution


def strtoidx(s):
    """Converts a string representation of indices to a list."""
    if s is None: 
        return None

    cols = []
    col_str = s.split(',')
    for col_elem in col_str:
        spl = col_elem.split('-', 2)
        if len(spl) == 1: cols.append(int(spl[0]) - 1)
        if len(spl) == 2: cols.extend(range(int(spl[0]) - 1, int(spl[1])))
    return cols


def load_array(fname):
    """Loads an array from a either a plain text or native numpy file."""
    import os
    if os.path.splitext(fname)[1] == '.npy':
        return numpy.load(fname)
    else:
        return numpy.loadtxt(fname)


def load_data(filename, input_cols, output_cols):
    """Loads a dataset of inputs and outputs given lists of indices for inputs 
    and outputs from a file."""
    data = load_array(filename)

    inputs = data[:,input_cols]
    outputs = data[:,output_cols]

    return inputs, outputs


def serialize(fp, *args):
    """Serializes strings, floats, integers, and arrays into a format 
    understood by the learningMachine framework."""
    for item in args:
        if type(item) == str:      
            fp.write('%s' % item)
        elif type(item) in [float, numpy.float32, numpy.float64]:  
            fp.write('%g ' % item)
        elif type(item) in [int, numpy.int16, numpy.int32, numpy.int64]:    
            fp.write('%d ' % item)
        elif type(item) == numpy.ndarray:
            numpy.savetxt(fp, item.flatten()[numpy.newaxis,:], fmt='%.10g', newline=' ')
            serialize(fp, *list(item.shape))


def serialize_preprocessor(filename, preprocessor):
    """Serializes the feature mapping, or preprocess, to a format understood by 
    the learningMachine framework."""
    with open(filename, 'w') as ppfp:
        serialize(ppfp, 'SparseSpectrumFeature\n', 
                        preprocessor.sigma_o, preprocessor.l, preprocessor.W,
                        preprocessor.n, preprocessor.outputdim())


def serialize_machine(filename, machine):
    """Serializes the machine to a format understood by the learningMachine 
    framework."""

    # copy upper triangular to lower triangular
    def reflect(arr):
        arr2 = arr.copy()
        arr2[numpy.tril_indices(arr.shape[0], -1)] = arr2.T[numpy.tril_indices(arr.shape[0], -1)]
        return arr2

    with open(filename, 'w') as lfp:
        try:
            m = machine.X.shape[0]
        except AttributeError:
            m = 0


        serialize(lfp, 'LinearGPR\n', 
                       reflect(machine.L), machine.B.T, machine.W.T, machine.sigma_n, m, 
                       machine.mapping.outputdim(), machine.p)


def printinfo(key, value):
    """Convenience function to print key, value pairs in a common format."""
    print('{:12s}: {}'.format(key, value))


def shapetostr(x):
    """Convenience function to represent an array shape as a string."""
    return '[' + ' x '.join(str(x) for x in x.shape) + ']'


def printerrors(y, y_p, y_pv):
    se = (y - y_p)**2
    nse = se / y.var(axis=0)
    lp = (se / y_pv) + numpy.log(2. * numpy.pi) + numpy.log(y_pv)

    printinfo('sign', (numpy.sign(y) == numpy.sign(y_p)).mean())
    printinfo('mse', se.mean(axis=0))
    printinfo('nmse', nse.mean(axis=0))
    printinfo('nmlp', 0.5 * lp.mean(axis=0))


class PrintTimer:
    def __init__(self, key='timing'):
        self.key = key

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, type, value, traceback):
        printinfo(self.key, '{:g} seconds'.format(time.time() - self.start))


def main():
    parser = argparse.ArgumentParser(description='SSGPR Tuning', 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    generalgroup = parser.add_argument_group('General Options')
    generalgroup.add_argument('--projections', type=int, default=100, metavar='D', 
                              help='number of spectral projections')
    generalgroup.add_argument('--nofixed', dest='fixed', action='store_false', default=True, 
                              help='tune sparse spectrum frequencies')
    generalgroup.add_argument('--params', type=float, metavar='X', nargs='+', 
                              help='set hyperparameters \sigma_n, \sigma_f, \ell (takes precedence over --guess)')
    generalgroup.add_argument('--seed', default=None, type=int, 
                              help='seed for the PRNG')
    generalgroup.add_argument('--yarp', default=False, action='store_true', 
                      help='serialize to YARP learningMachine compatible files')

    datasetgroup = parser.add_argument_group('Dataset Options')
    datasetgroup.add_argument('-i', '--inputs', default=[0], metavar='IDX[,IDX]*', 
                              type=strtoidx, help='input column indices')
    datasetgroup.add_argument('-o', '--outputs', default=[-1], metavar='IDX[,IDX]*', 
                              type=strtoidx, help='output column indices')
    datasetgroup.add_argument('--guess', metavar='DATASET', 
                              help='guess hyperparameters using specified dataset')
    datasetgroup.add_argument('--optimize', metavar='DATASET', 
                              help='optimize hyperparameters using specified dataset')
    datasetgroup.add_argument('--train', metavar='DATASET', 
                              help='train machine using specified dataset')
    datasetgroup.add_argument('--update', metavar='DATASET', 
                              help='update machine using specified dataset')
    datasetgroup.add_argument('--test', metavar='DATASET', 
                              help='test machine on specified dataset')

    optimizationgroup = parser.add_argument_group('Optimization Options')
    optimizationgroup.add_argument('--solver', default='CG', help='name of scipy solver')
    optimizationgroup.add_argument('--verboseopt', default=False, action='store_true', 
                                   help='enable verbose optimization output')
    optimizationgroup.add_argument('--ftol', type=float, default=None, metavar='TOL', 
                                   help='function tolerance for stop condition')
    optimizationgroup.add_argument('--gtol', type=float, default=None, metavar='TOL', 
                                   help='gradient tolerance for stop condition')
    optimizationgroup.add_argument('--maxiters', type=int, default=1000, metavar='ITERS', 
                                   help='maximum iterations')
    optimizationgroup.add_argument('--maxfevals', type=int, default=None, metavar='EVALS', 
                                   help='maximum function evaluations')


    args = parser.parse_args()
    numpy.random.seed(args.seed)


    n = len(args.inputs)
    p = len(args.outputs)

    # some arbitrary default parameters and no hyperpriors
    sigma_o, sigma_o_prior = 2., NoPrior()
    l, l_prior = [10.] * n, [NoPrior()] * n
    sigma_n, sigma_n_prior = 0.2, NoPrior()

    # construct machine and feature mapping
    ssf = SparseSpectrumFeatures(n, nproj=args.projections, 
                                 sigma_o=sigma_o, sigma_o_prior=sigma_o_prior, 
                                 l=l, l_prior=l_prior, fixed=args.fixed)
    ssgpr=LinearGPR(n, p, ssf, sigma_n=sigma_n, sigma_n_prior=sigma_n_prior)

    print('General Info')
    printinfo('columns', '{} -> {}'.format(args.inputs, args.outputs))
    printinfo('#proj', args.projections)
    printinfo('fixed', args.fixed)
    printinfo('dimensions', '({:d} -> {:d}) -> {:d}'.format(n, ssf.outputdim(), p))
    printinfo('seed', args.seed)


    # rudimentary guess of hyperparameters based on data
    if args.guess:
        print('\nHyperparameter Guess: {}'.format(args.guess))

        guessx, guessy = load_data(args.guess, args.inputs, args.outputs)
        printinfo('data', '{} -> {}'.format(shapetostr(guessx), shapetostr(guessy)))

        with PrintTimer():
            ssgpr.guessparams(guessx, guessy)


    # set hyperparameters if given
    if args.params is not None:
        ssgpr.setparams(list(eval(args.params)))

    # optimize hyperparameters
    if args.optimize:
        print('\nHyperparameter Optimization: {}'.format(args.optimize))

        hyperx, hypery = load_data(args.optimize, args.inputs, args.outputs)
        printinfo('data', '{} -> {}'.format(shapetostr(hyperx), shapetostr(hypery)))
        printinfo('solver', args.solver)
        printinfo('ftol', '{}'.format(args.ftol))
        printinfo('gtol', '{}'.format(args.gtol))
        printinfo('max fevals', args.maxfevals)
        printinfo('max iters', args.maxiters)

        with PrintTimer():
            res = ssgpr.optimize(hyperx, hypery, solver=args.solver, verbose=args.verboseopt, 
                                 ftol=args.ftol, gtol=args.gtol, maxIter=args.maxiters, 
                                 maxFunEvals=args.maxfevals, checkgrad=False)

        printinfo('opt -lml', res.fun)
        printinfo('stop cond', res.message)
        printinfo('fevals', res.nfev)
        try:
            printinfo('dfevals', res.njev)
        except AttributeError:
            pass
        printinfo('iters', res.nit)

    # train ssgpr using dataset
    if args.train:
        print('\nTraining: {}'.format(args.train))

        trainx, trainy = load_data(args.train, args.inputs, args.outputs)
        printinfo('data', '{} -> {}'.format(shapetostr(trainx), shapetostr(trainy)))

        with PrintTimer('train timing'):
            ssgpr.train(trainx, trainy)

        lml = ssgpr.lmlfunc()
        printinfo('lml', lml)

        with PrintTimer('test timing'):
            trainy_p, trainy_pv = ssgpr.predict(trainx)

        printerrors(trainy, trainy_p, trainy_pv)

    # incrementally update ssgpr on dataset
    if args.update:
        print('\nUpdating: {}'.format(args.update))

        updatex, updatey = load_data(args.update, args.inputs, args.outputs)
        printinfo('data', '{} -> {}'.format(shapetostr(updatex), shapetostr(updatey)))

        with PrintTimer():
            updatey_p, updatey_pv = numpy.empty_like(updatey), numpy.empty_like(updatey)
            for i in range(updatex.shape[0]):
                if i % 1000 == 0:
                    sys.stdout.write('{:7d}\r'.format(i))
                    sys.stdout.flush()
                updatey_p[i], updatey_pv[i] = ssgpr.update(updatex[i], updatey[i])

        printerrors(updatey, updatey_p, updatey_pv)

    # test ssgpr on dataset
    if args.test:
        print('\nTesting: {}'.format(args.test))

        testx, testy = load_data(args.test, args.inputs, args.outputs)
        printinfo('data', '{} -> {}'.format(shapetostr(testx), shapetostr(testy)))

        with PrintTimer():
            testy_p, testy_pv = ssgpr.predict(testx)

        printerrors(testy, testy_p, testy_pv)


    if args.yarp:
        print('\nYarp LearningMachine Serialization')

        machine_fn = 'ssgpr.model'
        serialize_machine(machine_fn, ssgpr)
        printinfo('machine', machine_fn)

        preprocessor_fn = 'ssf.model'
        serialize_preprocessor(preprocessor_fn, ssf)
        printinfo('preproc', preprocessor_fn)


if __name__ == "__main__":
    main()

