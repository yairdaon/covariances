try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'none',
    'author': 'Yair Daon',
    'url': 'https://github.com/yairdaon/covs',
    'download_url': 'https://github.com/yairdaon/covs',
    'author_email': 'yair.daon@gmail.com',
    'version': '1.0',
    'install_requires': ['nose'],
    'packages': ['cov'],
    'scripts': [],
    'name': 'cov'
}

from distutils.core import setup, Extension
import numpy.distutils.misc_util

setup(
    ext_modules=[Extension("_cov", ["_cov.c", "cov.c"])],
    include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs(),
)

setup(**config)
