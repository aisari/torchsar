from distutils.core import setup
from Cython.Build import cythonize
setup(ext_modules = cythonize(['mapping_operation.py', 'sampling.py', 'draw_shapes.py']))



