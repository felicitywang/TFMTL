# README
Download Instructions -
1. Download glove.6B.zip from https://nlp.stanford.edu/projects/glove/
2. Add GloVe directory path for glove.6B.100d.txt in testBaseline.config file.
3. Download training and testing data from the location shared by organizers.
4. Update paths of train and test data in testBaseline.config file.

NOTE: Path should have '/' or  '\\'

### Keras Installation
1. Install Conda in python 3.5 or above from https://conda.io/miniconda.html
2. Run below command to create and activate Keras environment
```sh
conda create --name kerasEnv python=3.5
activate kerasEnv
```
3. Install CNTK followed by Keras using below command
```sh
(kerasEnv) pip install cntk
(kerasEnv) pip install keras
(kerasEnv) set "KERAS_BACKEND=cntk"
(kerasEnv) python baseline.py -config /path/to/configFile
```
NOTE: All above Instructions are tested on WINDOWS machines. 