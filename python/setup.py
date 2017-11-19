import sys
import os
from setuptools import setup, find_packages

if sys.platform == 'win32':
    LIB_NAME = "liblm.dll"
elif sys.platform == 'darwin':
    LIB_NAME = "libliblm.dylib"
else:
    LIB_NAME = "libliblm.so"

LIB_PATH = os.path.join('liblm', LIB_NAME)

if not os.path.exists(LIB_PATH):
    raise Exception("Can not find C-dll file!")


print('Install liblm from: ' + LIB_PATH)

setup(name='liblm',
      version='0.01',
      description="LibLM Python Package",
      install_requires=[
          'numpy',
          'scipy',
      ],
      author = "liu",
      author_email = "wisedoge@outlook.com",
      packages=find_packages(),
      zip_safe=False,
      include_package_data=True,
      data_files=[('liblm', [LIB_PATH])],
      license = "MIT Licence",
      url='https://github.com/wisedoge/liblm')