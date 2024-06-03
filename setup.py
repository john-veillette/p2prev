from setuptools import setup, find_packages
import sys

setup(
    name = 'p2prev',
    version = '0.0.1',
    description = 'estimating population prevalence from unthresholded p-values from repeated n=1 experiments',
    url = 'https://github.com/john-veillette/p2prev',
    author = 'John Veillette',
    author_email = 'johnv@uchicago.edu',
    license = 'BSD-3-Clause',
    packages = find_packages(),
    install_requires = ['pymc>=4.0', 'arviz', 'numpy', 'pandas'],
    classifiers = [
            'Intended Audience :: Science/Research',
            'Intended Audience :: Developers',
            'License :: OSI Approved',
            'Programming Language :: Python',
            'Topic :: Software Development',
            'Topic :: Scientific/Engineering',
            'Operating System :: Microsoft :: Windows',
            'Operating System :: POSIX',
            'Operating System :: Unix',
            'Operating System :: MacOS',
            'Programming Language :: Python :: 3',
    ]
)
