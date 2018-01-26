def cnn(inputs,
        num_filter=64,
        max_width=3,
        encode_dim=256,
        activation_fn=tf.nn.elu):
  # inputs: word embeddings

  filter_sizes = []
  for i in xrange(2, max_width+1):
    filter_sizes.append((i + 1, num_filter))

  # Convolutional layers
  filters = []
  for width, num_filter in filter_sizes:
    conv_i = tf.layers.conv1d(
      inputs,
      num_filter,  # dimensionality of output space (num filters)
      width,  # length of the 1D convolutional window
      data_format='channels_last',  # (batch, time, embed_dim)
      strides=1,  # stride length of the convolution
      activation=tf.nn.relu,
      padding='SAME',  # zero padding (left and right)
      name='conv_{}'.format(width))

    # Max pooling
    pool_i = tf.reduce_max(conv_i, axis=1, keep_dims=False)

    # Append the filter
    filters.append(pool_i)

    # Increment filter index
    i += 1

  # Concatenate the filters
  inputs = tf.concat(filters, 1)

  # Return a dense transform
  return dense_layer(inputs, output_size=encode_dim, name='l1',
                     activation=activation_fn)
