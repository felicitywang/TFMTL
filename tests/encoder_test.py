import argparse
import tensorflow as tf

from mtl.util.encoder_factory import build_encoders


parser = argparse.ArgumentParser()
parser.add_argument('--architecture', default='paragram')
parser.add_argument('--datasets', default=['SSTb', 'LMRD'])
parser.add_argument('--encoder_config_file', default='.encoders.json')
args = parser.parse_args()

vocab_size = 1000
encoders = build_encoders(vocab_size, args)

print('Encoders: {}'.format(encoders))

inputs = tf.constant([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
lengths = tf.constant([3,3,3,3])
output = encoders['SSTb'](inputs=inputs, lengths=lengths)
print('Output: {}'.format(output))
