# Copyright 2017 Johns Hopkins University. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from abc import ABCMeta
from abc import abstractmethod

import six
from six.moves.urllib.request import urlretrieve

import tensorflow as tf


@six.add_metaclass(ABCMeta)
class Dataset():
  SOURCE_URL = None
  DEST_ARCHIVE = None
  JSON_FILES = ('experimental', 'test')
  NAME = 'dataset'
  EOS = '<eos>'
  UNK = "<unk>"

  def __init__(self, work_directory=None):
    if work_directory is None:
      work_directory = self.__class__.NAME
    self._work_directory = work_directory

    self._vocab = None
    self.setup()
    assert self.vocab is not None
    assert self.json_ready(), str(os.listdir(self.work))

  @abstractmethod
  def json_ready(self):
    '''
      Dataset-specific check for whether data is correctly downloaded
    '''
    pass

  @abstractmethod
  def setup(self):
    '''
      Dataset-specific code for downloading, preprocessing,
      splitting, and jsonifying the data
    '''
    pass

  def maybe_download(self, url, filename, work_directory):
    """Download the data unless it's already here."""
    if not os.path.exists(work_directory):
      os.mkdir(work_directory)
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):
      tf.logging.info('Data not found. Downloading from {}'.format(
                      url))
      filepath, _ = urlretrieve(url, filepath)
      statinfo = os.stat(filepath)
      tf.logging.info('Succesfully downloaded {} bytes'.format(
                      statinfo.st_size))
    return filepath

  def extract(self, archive_path):
    tf.logging.info("Extracting archive {}".format(archive_path))
    if archive_path.endswith(".tgz"):
      import tarfile
      with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(self.work)
    elif archive_path.endswith(".zip"):
      import zipfile
      with zipfile.ZipFile(archive_path, 'r') as zipf:
        zipf.extractall(self.work)
    elif archive_path.endswith(".tar.bz2"):
      import tarfile
      with tarfile.open(archive_path, "r:bz2") as tar:
        tar.extractall(self.work)
    elif archive_path.endswith(".gz"):
      import gzip
      with gzip.open(archive_path, 'rb') as in_file:
        s = in_file.read()
        output_path = archive_path[:-3]  # remove the '.gz' from the filename
        with open(output_path, 'wb') as f:
          f.write(s)
    else:
      raise ValueError("DEST_ARCHIVE must match one of (*.tgz, *.zip)")


  @property
  def work(self):
    return self._work_directory
