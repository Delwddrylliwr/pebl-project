# download setuptools if needed
import ez_setup
ez_setup.use_setuptools()

from setuptools import setup, find_packages

setup(
    name='Pebl',
    version='0.2.1',
    description='Python Environment for Bayesian Learning',
    package_dir={'': 'src'},
    packages=find_packages('src'),

    ## Package metadata.. for upload to PyPI
    author='Abhik Shah',
    author_email='abhikshah@gmail.comm',
    url='http://sysbio.engin.umich.edu/Pebl/',

    # required dependencies
    install_requires=['numpy >= 0.9', 'nose >= 0.9', 'pydot'],
    
    # data files (mostly just the test data for unit tests)
    include_package_data = True,
)
