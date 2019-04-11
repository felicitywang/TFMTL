from setuptools import setup, find_packages

REQUIRED_PACKAGES = ['six', 'enum34']

setup(name='mtl',
      version='0.1',
      description='A TensorFlow Package for Multi-Task Learning',
      url='https://github.com/felicitywang/tfmtl',
      author='Johns Hopkins University',
      license='2-Clause BSD',
      packages=find_packages(),
      setup_requires=['pytest-runner'],
      tests_require=['pytest'],
      install_requires=REQUIRED_PACKAGES,
      zip_safe=False)
