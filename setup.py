#!/usr/bin/env python
from setuptools import setup, find_packages

__version__ = '0.999'

setup(
    name='coin',
    version=__version__,
    url='http://stck.co',
    packages=find_packages(),
    include_package_data=True,
    entry_points="""
        [console_scripts]
        coin-segment-extract=coin.extractor:main
    """
)
