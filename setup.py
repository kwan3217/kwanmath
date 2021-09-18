"""
Setup file for the kwanmath package.

The only required fields for setup are name, version, and packages. Other fields to consider (from looking at other
projects): keywords, include_package_data, requires, tests_require, package_data
"""
from setuptools import setup

setup(
    name='kwanmath',
    version='0.1.0',
    description='Mathematics the Kwan Systems way',
    url='https://github.com/kwan3217/kwanmath/',
    author='kwan3217',
    author_email='kwan3217@gmail.com',
    license='BSD 2-clause',
    packages=['kwanmath.vector','kwanmath.bezier'],
    install_requires=[
                      'numpy',
                     ],

)
