from setuptools import setup, Extension
import numpy
blas_info = numpy.__config__.blas_opt_info

ext = Extension('ssgpr.chollrup', 
                sources=['ssgpr/chollrup.c'],
                libraries=blas_info.get('libraries', []),
                library_dirs=blas_info.get('library_dirs', []),
                include_dirs=[numpy.get_include()] + blas_info.get('include_dirs', [])
               )

setup(name='ssgpr',
      version='1.0.1',
      description='Batch and Incremental Sparse Spectrum Gaussian Process Regression',
      author='Arjan Gijsberts',
      packages=['ssgpr'],
      install_requires=['numpy>=1.15.0', 'scipy>=1.0.0'],
      ext_modules=[ext],
     )
