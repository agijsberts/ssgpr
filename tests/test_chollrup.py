#
# tests for the cholesky rank-1 update.
# python variant of the on Matlab code by Matthias Seeger
# https://github.com/mseeger/chollrup/tree/main/chollrup
#
import scipy
import scipy.linalg
import numpy
import time
from ssgpr.chollrup import choluprk1

n = 5
count = 10

maxlam=2
minlam=0.1

numpy.random.seed(2)
#
# Main
#
def main():
    test_chollrup()

def test_chollrup():
    print('ChollRup test')

    errors = numpy.empty(count)
    times_chol = numpy.empty(count)
    times_up = numpy.empty(count)
    
    for i in range(count):
        #% Create matrix A with controlled spectrum
        q,r = scipy.linalg.qr(numpy.random.standard_normal((n,n)) + numpy.identity(n))
        a = numpy.dot((q * (numpy.random.rand(n) * (maxlam-minlam)+minlam)), q.T)
        
        # check for positive definiteness
        ev_a = numpy.linalg.eigvals(a)
        assert (ev_a > 0.).all()
        
        start = time.time()
        lfact = numpy.linalg.cholesky(a).T
        end = time.time()
        times_chol[i] = end - start

        #% Update
        b = numpy.random.standard_normal((3*n,n))
        y = numpy.random.standard_normal((3*n,n))
        z = numpy.linalg.solve(lfact, b.T).T
        vec = numpy.random.standard_normal((n,))
        start = time.time()
        choluprk1(lfact,vec)
        end = time.time()
        times_up[i] = end - start
        
        l_2 = numpy.linalg.cholesky(a + numpy.outer(vec, vec)).T
        errors[i] = numpy.max(numpy.abs(lfact-l_2))

        numpy.testing.assert_allclose(l_2, lfact)

    print('max_error:', errors.max())
    print('chol_avg_time: {:g} (+/- {:g})'.format(times_chol.mean(), times_chol.std()))
    print('up_avg_time: {:g} (+/- {:g})'.format(times_up.mean(), times_up.std()))

if __name__ == "__main__":
    main()
