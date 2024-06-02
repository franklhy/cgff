from setuptools import setup, find_packages

setup(
    name='cgff',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'mpi4py',
        #'lammps',
        #'plato'
    ],
    package_data={'silc': ['example/*/**',]}
)
