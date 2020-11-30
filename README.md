# SSGPR

This library implements the batch and incremental Sparse Spectrum Gaussian 
Process for Regression (SSGPR). A detailed description of the batch algorithm can be 
found in:

> Sparse Spectrum Gaussian Process Regression. 
  Miguel Lazaro-Gredilla, Joaquin Quinonero-Candela, Carl Edward Rasmussen, 
  and Anibal R. Figueiras-Vidal. 
  In Journal of Machine Learning Research, 2010.

The incremental variant is described in:

> Real-time model learning using Incremental Sparse Spectrum Gaussian Process 
  Regression.
  Arjan Gijsberts and Giorgio Metta.
  In Neural Networks Volume 41, 2013.


## Installation

After downloading, you can install `ssgpr` via the [pip](https://pip.pypa.io/en/stable/) package manager with

```bash
pip3 install .
```

or if that's not an option directly via `setup.py` with

```bash
python3 setup.py install --user
```
Either command will need to be executed from within the base directory. Also, you'll need `numpy`, `scipy`, and a compiler. The scripts seem to install and run fine with Python 2, but there's really no good excuse to use it any longer.

## Usage

The bare minimum code to use SSGPR would be something like 
```
from ssgpr import SSGPR

machine = SSGPR(inputdim, outputdim, nproj=100)
machine.guessparams(trainx, trainy)
machine.optimize(trainx, trainy)
machine.train(trainx, trainy)

for x, y in testdata:
    predy, predy_var = machine.update(x, y)
```
A self-contained version of this example is included in the `examples` directory.

That directory also contains a `tune.py` script, which allows running an experimental pipeline on text or `.npy` files. To evaluate the incremental performance on the problem of predicting [inverse dynamics of the iCub's robot arm](https://github.com/robotology/icub-main/blob/master/src/modules/learningMachine/exampledata/icubdyn.tgz), one might run
```
python3 examples/tune.py --inputs 1-12 --outputs 13-15 --projections 100 \
--guess icubdyn_hyper.dat --optimize icubdyn_hyper.dat --train icubdyn_train.dat \
--update icubdyn_test.dat --verboseopt --solver tnc --maxiter 1000
```
The `--yarp` flag allows saving the model to files that can be read with the [learningMachine](https://github.com/robotology/icub-main/tree/master/src/modules/learningMachine) module in the [iCub Software Repository](https://github.com/robotology/icub-main).

## Hints

* The number of random projections trades approximation accuracy for computation time. You should ideally choose the largest number of projections that you can afford computationally.

* SSPGR is a randomized method, so it’s best to average multiple runs with different initializations (seeds). The variability over multiple runs decreases as the number of projections increases.

* The model can learn multiple outputs jointly at negligible additional cost, but in this case the hyperparameters are the same for all outputs. This makes sense if all outputs behave similarly (e.g., forces in x, y, and z dimensions), but not if the problems are of different nature (e.g., forces versus torques). In the latter case you should just train a different model.

* The log marginal likelihood is non-convex and it's optimization via gradient ascent will therefore result in a *local* optimum. Make sure to choose sensible initial hyperparameters (e.g., via the `guessparams` method) and to experiment with different solvers.

* Optimizing the spectral frequencies essentially means optimizing the kernel function to your data. This can allow you to get better performance with fewer projections, but at increased risk of overfitting due to the large number of tunable hyperparameters. Use with care and ideally only if you have plenty of training data.


## Notes
This code was written ages ago and ported more recently to Python 3 to keep it functional. Pull requests to bring it more up to date in 2020 are welcome (e.g., adhering to a `sklearn` interface, `autograd`).

## Acknowledgements
Miguel Lazaro-Gredilla's original [Matlab implementation](http://www.tsc.uc3m.es/~miguel/downloads.php) was a very helpful source of information. The code for the the rank-1 Cholesky is an adaptation of Matthias Seeger's [MEX code](https://github.com/mseeger/chollrup/), which itself was based on [LINPACK](https://www.netlib.org/linpack/)'s `dchud`.

## License
[BeerWare](https://spdx.org/licenses/Beerware.html)
