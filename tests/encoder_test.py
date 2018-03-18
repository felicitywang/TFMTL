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

inputs1 = tf.constant([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
lengths1 = tf.constant([3,3,3,3])

output_SSTb_1 = encoders['SSTb'](inputs=inputs1, lengths=lengths1)
output_LMRD_1 = encoders['LMRD'](inputs=inputs1, lengths=lengths1)
print('Output SSTb 1: {}'.format(output_SSTb_1))
print('Output LMRD 1: {}'.format(output_LMRD_1))

inputs2 = tf.constant([[1,1,1],[2,2,2]])
lengths2 = tf.constant([3,3])
output_SSTb_2 = encoders['SSTb'](inputs=inputs2, lengths=lengths2)
output_LMRD_2 = encoders['LMRD'](inputs=inputs2, lengths=lengths2)
print('Output SSTb 2: {}'.format(output_SSTb_2))
print('Output LMRD 2: {}'.format(output_LMRD_2))

print('All variables created...')
all_variables = tf.global_variables()
print(type(all_variables))
for var in all_variables:
  print(var)

print('Trainable variables created...')
trainable_variables = tf.trainable_variables()
print(type(trainable_variables))
for var in trainable_variables:
  print(var)
