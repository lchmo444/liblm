import sys
import os
from setuptools import setup, find_packages

if sys.platform == 'win32':
    LIB_NAME = "liblm.dll"
elif sys.platform == 'darwin':
    LIB_NAME = "libliblm.dylib"
else:
    LIB_NAME = "libliblm.so"


LIB_PATH = './liblm/' + LIB_NAME

if not os.path.exists(LIB_PATH):
    raise Exception("Can not find C-dll file!")

setup(name='liblm',
      version='beta',
      description="LibLM Python Package",
      install_requires=[
          'numpy',
          'scipy',
      ],
      author = "liu",
      author_email = "wisedoge@outlook.com",
      packages=find_packages(),
      include_package_data=True,
      data_files=[('cdll', LIB_PATH)],
      license = "MIT Licence",
      url='https://github.com/wisedoge/liblm')