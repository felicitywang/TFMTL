## TFMTL
One focus of this repository is a collection of off-the-shelf functions
for transforming inputs into feature representations. These transformations
are referred to as "encoders", and they consist of two steps: an embedding
step that turns input token IDs into token embeddings, and an extraction step
that turns the embeddings into feature representations. Together, the embedder
and extractor constitute an encoder.

An architecture is a specification of an encoder for each dataset in an experiment.
Users specify encoder architectures in a JSON configuration file. An architecture
is specified with a name (key) whose value is a dictionary with the following fields:

* `embedders_tied`:
  * `true` if all datasets should be embedded with the same function (i.e., shared parameters),
  * `false` otherwise
* `extractors_tied`:
  * `true` if all datasets should have features extracted with the same function (i.e., shared parameters),
  * `false` otherwise
* Name of dataset A
  * `embed_fn`: a string specifying an embedding function
  * `embed_kwargs`: a dictionary specifying `argument: value` pairs (arguments are strings) for the embedding function
  * `extract_fn`: a string specifying an extraction function
  * `extract_kwargs`: a dictionary specifying `argument: value` pairs (arguments are strings) for the extraction function
* Name of dataset B...

`embed_fn`, `embed_kwargs`, `extract_fn`, and `extract_kwargs` must be fully specified,
even if `embedders_tied` or `extractors_tied` is `true`. If `embedders_tied` is `true`
for an architecture, then `embed_fn` and `embed_kwargs` must have identical values for
all datasets specified in the given architecture (similarly if `extractors_tied` is `true`).

`true` is the JSON equivalent of Python's `True` value.

`false` is the JSON equivalent of Python's `False` value.

`null` is the JSON equivalent of Python's `None` value.

An example configuration file can be found at `tfmtl/tests/encoders.json`.

Embedding functions and extraction functions can be found in `tfmtl/mtl/embedders`
and `tfmtl/mtl/extractors`, respectively.

The architecture to use in an experiment is given by the `--architecture` flag,
and the file containing the architecture(s) is given by the `--encoder_config_file`
flag. Multiple architectures can be placed in the same configuration file. -->




## Encoder configuration files

Encoders are configured using a JSON file (often called `encoders.json`). Configuration files may contain specifications for any number of encoders, with each encoder in the file getting a unique identifier. The identifier is the “key” and the configuration is the “value”, where the configuration is itself a collection of (potentially nested) key-value pairs. An example is located at `tests/encoders.json` (as of this writing).

Each configuration requires an `embedders_tied` field and an `extractors_tied` field. These fields denote whether the embedders or extractors for your multiple tasks should be copies of each other (tied parameters), or whether they should be independent. If either of these fields is `true`, then the configuration of the corresponding functionality (below) must be identical across all your tasks’ sub-configurations.

Each of your tasks gets its own sub-configuration. These sub-configurations specify what embedder to use (`embed_fn`), what keyword arguments (kwargs) to use for the embedder (`embed_kwargs`), what extractor to use (`extract_fn`), and what kwargs to use for the extractor (`extract_kwargs`). Many options are enumerated in `mtl/util/hparams.py` (as of this writing). A lack of kwargs is denoted by an empty dictionary instead of no entry at all.

A list of currently supported embedders, extractors, and their arguments is given below.


## Arguments

### Embedders

`embed_sequence`: embeds tokens with a uniformly-distributed embedding matrix
* `word_ids`: word id sequences, shape of [batch_size, seq_len]
* `weights`: weight sequences, shape of [batch_size, seq_len]
* `vocab_size`: number of items in the vocabulary
* `embed_dim`: size of the embeddings
* returns (maybe weighted) sequence of word embeddings, shape of [batch_size, seq_len, embed_dim]

`no_op_embedding`: puts data in correct output shape but does not change data values; for use when the data is already embedded or encoded in preprocessing
* `word_ids`: word id sequences, shape of [batch_size, seq_len]
* `weights`: weight sequences, shape of [batch_size, seq_len]
* `vocab_size`: number of items in the vocabulary
* `embed_dim`: size of the embeddings

`only_pretrained`: uses the pre-trained word embeddings only to embed the word id sequences
* `word_ids`: word id sequences, shape of [batch_size, seq_len]
* `weights`: weight sequences, shape of [batch_size, seq_len]
* `vocab_size`: number of items in the vocabulary
* `embed_dim`: size of the embeddings
* `trainbale`: whether to fine-tune the pre-trained word embeddings
* `pretrained_path`: path to the pre-trained word embedding file
* `is_training`: set is_training to False to avoid re-initializing from scratch
* `proj_dim`: dimension of the projection layer if there's one

`expand_pretrained`: expands training vocab with pre-trained word embeddings
* `word_ids`: word id sequences, shape of [batch_size, seq_len]
* `weights`: weight sequences, shape of [batch_size, seq_len]
* `vocab_size`: number of items in the vocabulary
* `embed_dim`: size of the embeddings
* `trainbale`: whether to fine-tune the pre-trained word embeddings(the randomly initialized part is always adapted)
* `pretrained_path`: path to the pre-trained word embedding file
* `is_training`: set is_training to False to avoid re-initializing from scratch
* `proj_dim`: dimension of the projection layer if there's one

`init_pretrained`: initialized training vocab with pre-trained word embeddings
* `word_ids`: word id sequences, shape of [batch_size, seq_len]
* `weights`: weight sequences, shape of [batch_size, seq_len]
* `vocab_size`: number of items in the vocabulary
* `embed_dim`: size of the embeddings
* `trainbale`: whether to fine-tune the pre-trained word embeddings(the randomly initialized part is always adapted)
* `pretrained_path`: path to the pre-trained word embedding file
* `is_training`: set is_training to False to avoid re-initializing from scratch
* `proj_dim`: dimension of the projection layer if there's one

### Extractors

`paragram`: applies pooling to embedded sequences, optionally with (non-)linear layer; name may change in future
* `inputs`: embedded sequences
* `lengths`: sequence lengths
* `reducer`: pooling operation
* `apply_activation`: whether to apply an activation function
* `activation_fn`: activation function to apply after pooling; `None` denotes a linear function

`dan`: deep averaging network, mean-pool + (non-)linear layer(s), may apply word embedding dropout
* `inputs`: embedded sequences
* `lengths`: sequence lengths
* `word_dropout_rate`: how much word embedding to drop out
* `reducer`: pooling operation
* `apply_activation`: whether to apply activation
* `num_layers`: number of (non-)linear layers
* `is_training`: when not training, no word embedding is dropped out

`lbirnn_stock`: linear bi-directional RNN; uses stock implementation of TensorFlow bi-RNN
* `inputs`: embedded sequences
* `lengths`: sequence lengths
* `is_training`: whether model is in training mode or in inference mode
* `num_layers`: number of layers in stacked bi-RNN
* `cell_type`: kind of RNN cell (e.g., LSTM, GRU)
* `cell_size`: dimensionality of cell state
* `initial_state_fwd`: initial state of forward-direction RNN
* `initial_state_bwd`: initial state of backward-direction RNN
* `kwargs`: additional arguments such as whether to use skip connections, whether to use attention, whether to use dropout

`rnn_and_pool`: linear RNN
* `inputs`: embedded sequences
* `lengths`: sequence lengths
* `num_layers`: number of layers in stacked bi-RNN
* `cell_type`: kind of RNN cell (e.g., LSTM, GRU)
* `cell_size`: dimensionality of cell state
* `initial_state`: initial state of RNN
* `reducer`: pooling operation

`cnn_extractor`: CNN
* `inputs`: embedded sequences
* `lengths`: sequence lengths
* `num_filter`: number of filters for each width
* `max_width`: maximum filter width
* `activation_fn`: non-linearity to apply after convolutions; `None` denotes a linear function
* `reducer`: pooling operation

`concat_extractor`: no-op extractor for use when the data is already embedded or encoded in preprocessing
* `inputs`: embedded sequences
* `lengths`: sequence lengths



## Content and shape of input and output to embedder and extractors

Embedders take in tensors of shape `[batch_size, sequence_length]`. Each row in the tensor is a sequence of word IDs. Embedders output tensors of shape `[batch_size, sequence_length, embedding_size]`. Each row is a sequence of (word) embeddings (the terminology perhaps may be different in the case of, e.g., ELMo, in which case the “embedder” acts more as a “contextualizer”).

In general, extractors take in tensors of shape `[batch_size, sequence_length, embedding_size]`, almost always as the output of an embedder. They output tensors of shape `[batch_size, D]`, often via an intermediate tensor of shape `[batch_size, sequence_length, D]` and some pooling operation. Sometimes `D` is the embedding size, sometimes it is the size of an RNN cell, etc. Deviations from these trends should be noted in the extractors’ in-line documentation.

In the `Mult` model (as of this writing), the output of the extractor goes directly into some MLPs, which transparently handle the size of the second dimension.


## How to write your own embedder or extractor

Embedders take as their first argument a tensor of word IDs. Configuration arguments get passed in as kwargs (`mtl/util/embedder_factory.py`) when the embedder is turned into a partially applied function.

Extractors take as their first argument a tensor of embedded sequences, and as their second argument a tensor of sequence lengths. Configuration arguments get passed in as kwargs (`mtl/util/extractor_factory.py`) when the extractor is turned into a partially applied function. Additional kwargs get passed in at runtime (`mtl/util/encoder_factory.py::encoder_fn()`, `mtl/models/mult.py::encode()`).

Beyond that, your embedder or extractor can implement whatever functionality you want (in keeping with the general input/output shapes outlined above). Don’t forget to add new function-valued arguments to `mtl/util/hparams.py::str2func()` as necessary.


## Some Clarifications of the simple extractors
People have used different names for these similar extractors:
* `paragram-phrase`: mean-pool ("Towards universal paraphrastic sentence embedding", Wieting et al., 2016)
* `paragram-phrase prob`: mean-pool + 1-layer linear projection
* `dan`: mean-pool + 1 or more hidden layers(with non-linear activations)("Deep Unordered Composition Rivals Syntactic Methods for Text Classification", Iyyer et al., 2015)
* People have also used the idf score as the weight for each token to represent a document ("Improving word representations via global context and multiple word prototypes", Huang et al., 2012; "Inducing Crosslingual Distributed Representations of Words", Klementiev et al., 2012)

