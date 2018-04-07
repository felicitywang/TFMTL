from setuptools import setup, find_packages

REQUIRED_PACKAGES = ['six', 'enum34', 'sklearn']

setup(name='mtl',
      version='0.1',
      description='A TensorFlow Package for Multi-Task Learning',
      url='https://github.com/noa/tfmtl',
      author='Johns Hopkins University',
      author_email='noa@jhu.edu',
      license='Apache 2.0',
      packages=find_packages(),
      setup_requires=['pytest-runner'],
      tests_require=['pytest'],
      install_requires=REQUIRED_PACKAGES,
      zip_safe=False)
