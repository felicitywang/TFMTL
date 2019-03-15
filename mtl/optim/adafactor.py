# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
# Copyright 2018 Johns Hopkins University.
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

"""Optimization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import tensorflow as tf


class AdafactorOptimizer(tf.train.Optimizer):
    """Optimizer that implements the Adafactor algorithm.
  
    Adafactor is described in: https://arxiv.org/pdf/1804.04235.pdf
  
    Adafactor is most similar to Adam (Kingma and Ba), the major differences are:
  
    1. For a two-dimensional AxB weight matrix, Adafactor uses only A+B auxiliary
       parameters to maintain the second-moment estimator, instead of AB.
       This is advantageous on memory-limited systems.  In addition, beta1
       (momentum) is set to zero by default, saving an additional auxiliary
       parameter per weight.
  
    2. Adafactor incorporates "update-clipping" - a scale-invariant analog of
       gradient clipping.  This adds stability
  
    3. Adafactor does not require an external "learning rate".  By default, it
       incorporates a relative-update-scale schedule, corresponding to
       inverse-square-root learning-rate-decay in ADAM.  We hope this works well
       for most applications.
  
    ALGORITHM:
  
    parameter -= absolute_update_scale * clip(grad / grad_scale)
  
    where:
  
      absolute_update_scale := relative_update_scale * parameter_scale
      relative_update_scale := min((step_num + 1)**-0.5, 1e-2)
      parameter_scale := max(rms(var)), 1e-3)
      clip(x) := x / max(1.0, rms(x))
      grad_scale := tf.sqrt(v)   (v is the second-moment estimator)
  
    The second-moment estimator v is maintained in a manner similar to Adam:
    We initialize
    ```
    if var is 2-dimensional:
      v_r <- zeros([num_rows])
      v_c <- zeros([num_cols])
    else:
      v <- zeros(shape(var))
    ```
  
    The update rule is as follows:
    ```
    decay_rate = 1 - (step_num + 1) ^ -0.8
    grad_squared = tf.square(grad) + epsilon
    if var is 2-dimensional:
      v_r <- decay_rate * v_r + (1 - decay_rate) * reduce_mean(grad_squared, 1)
      v_c <- decay_rate * v_c + (1 - decay_rate) * reduce_mean(grad_squared, 0)
      v = outer_prod(v_r, v_c) / reduce_mean(v_r)
    else:
      v <- decay_rate * v + (1 - decay_rate) * grad_squared
    ```
  
  
    Several parts of this algorithm are configurable from the initializer.
  
      multiply_by_parameter_scale:  If True, then compute absolute_update_scale
        as described above.  If False, let absolute_update_scale be the externally
        supplied learning_rate.
      learning_rate: represents relative_update_scale if
        multiply_by_parameter_scale==True, or absolute_update_scale if
        multiply_by_parameter_scale==False.
      decay_rate: Decay rate of the second moment estimator (varies by step_num).
        This should be set to a function such that:
        1-1/(step_num + 1) <= decay_rate(step_num) < 1.0
      beta1: enables momentum, as in Adam.  Uses extra memory if nonzero.
      clipping_threshold: should be >=1.0 or None for no update clipping
      factored: whether to factor the second-moment estimator.  True means
        less memory usage.
  
    TODO(noam): we should also apply the 2d logic to the two final dimensions.
      of >2d convolutional kernels.
    """

    def __init__(self,
                 multiply_by_parameter_scale=True,
                 learning_rate=None,
                 decay_rate=None,
                 beta1=0.0,
                 clipping_threshold=1.0,
                 factored=True,
                 simulated_quantize_bits=None,
                 use_locking=False,
                 name="Adafactor"):
        """Construct a new Adafactor optimizer.
    
        See class comment.
    
        Args:
          multiply_by_parameter_scale: a boolean
          learning_rate: an optional Scalar.
          decay_rate: an optional Scalar.
          beta1: a float value between 0 and 1
          clipping_threshold: an optional float >= 1
          factored: a boolean - whether to use factored second-moment estimator
            for 2d variables
          simulated_quantize_bits: train with simulated quantized parameters
            (experimental)
          use_locking: If True use locks for update operations.
          name: Optional name for the operations created when applying gradients.
            Defaults to "AdafactorOptimizer".
    
        Raises:
          ValueError: if absolute_update_scale and relative_update_scale_fn are both
            present or both absent.
        """
        super(AdafactorOptimizer, self).__init__(use_locking, name)
        self._multiply_by_parameter_scale = multiply_by_parameter_scale
        if learning_rate is None:
            learning_rate = self._learning_rate_default(multiply_by_parameter_scale)
        self._learning_rate = learning_rate
        if decay_rate is None:
            decay_rate = self._decay_rate_default()
        self._decay_rate = decay_rate
        self._beta1 = beta1
        self._clipping_threshold = clipping_threshold
        self._factored = factored
        self._simulated_quantize_bits = simulated_quantize_bits
        if self._simulated_quantize_bits:
            self._quantization_noise = _quantization_noise_from_step_num()

    def _should_use_factored_second_moment_estimate(self, shape):
        """Should we use a factored second moment estimator.
    
        Based on the shape of the variable.
    
        Args:
          shape: a list of integers
        Returns:
          a boolean
        """
        return self._factored and len(shape) == 2

    def _create_slots(self, var_list):
        for var in var_list:
            shape = var.get_shape().as_list()
            if self._beta1:
                self._zeros_slot(var, "m", self._name)
            if self._should_use_factored_second_moment_estimate(shape):
                r_val = tf.zeros([shape[0]], dtype=tf.float32)
                c_val = tf.zeros([shape[1]], dtype=tf.float32)
                self._get_or_make_slot(var, r_val, "vr", self._name)
                self._get_or_make_slot(var, c_val, "vc", self._name)
            else:
                v_val = tf.zeros(shape, dtype=tf.float32)
                self._get_or_make_slot(var, v_val, "v", self._name)

    def _apply_dense(self, grad, var):
        return self._resource_apply_dense(grad, var)

    def _apply_sparse(self, grad, var):
        return self._apply_dense(tf.convert_to_tensor(grad), var)

    def _parameter_scale(self, var):
        """Estimate the scale of the parameters from the current values.
    
        We include a minimum value of 0.001 to give it a chance to escape 0
        if it was zero-initialized.
    
        Instead of using the value, we could impute the scale from the shape,
        as initializers do.
    
        Args:
          var: a variable or Tensor.
        Returns:
          a Scalar
        """
        return tf.maximum(reduce_rms(var), 0.001)

    def _resource_apply_dense(self, grad, var):
        grad = tf.to_float(grad)
        grad_squared = tf.square(grad) + 1e-30
        grad_squared_mean = tf.reduce_mean(grad_squared)
        decay_rate = self._decay_rate
        update_scale = self._learning_rate
        if self._multiply_by_parameter_scale:
            update_scale *= tf.to_float(self._parameter_scale(var))
        # HACK: Make things dependent on grad.
        # This confounds the XLA rewriter and keeps it from fusing computations
        # across different variables.  This fusion is a bad for HBM usage, since
        # it causes the gradients to persist in memory.
        decay_rate += grad_squared_mean * 1e-30
        update_scale += grad_squared_mean * 1e-30
        # END HACK
        mixing_rate = 1.0 - decay_rate
        shape = var.get_shape().as_list()
        updates = []
        if self._should_use_factored_second_moment_estimate(shape):
            grad_squared_row_mean = tf.reduce_mean(grad_squared, 1)
            grad_squared_col_mean = tf.reduce_mean(grad_squared, 0)
            vr = self.get_slot(var, "vr")
            new_vr = (decay_rate * vr + mixing_rate * grad_squared_row_mean)
            vc = self.get_slot(var, "vc")
            new_vc = (decay_rate * vc + mixing_rate * grad_squared_col_mean)
            vr_update = tf.assign(vr, new_vr, use_locking=self._use_locking)
            vc_update = tf.assign(vc, new_vc, use_locking=self._use_locking)
            updates = [vr_update, vc_update]
            long_term_mean = tf.reduce_mean(new_vr)
            r_factor = tf.rsqrt(new_vr / long_term_mean)
            c_factor = tf.rsqrt(new_vc)
            x = grad * tf.expand_dims(r_factor, 1) * tf.expand_dims(c_factor, 0)
        else:
            v = self.get_slot(var, "v")
            new_v = decay_rate * v + mixing_rate * grad_squared
            v_update = tf.assign(v, new_v, use_locking=self._use_locking)
            updates = [v_update]
            x = grad * tf.rsqrt(new_v)
        if self._clipping_threshold is not None:
            clipping_denom = tf.maximum(1.0, reduce_rms(x) / self._clipping_threshold)
            x /= clipping_denom
        subtrahend = update_scale * x
        if self._beta1:
            m = self.get_slot(var, "m")
            new_m = self._beta1 * tf.to_float(m) + (1.0 - self._beta1) * subtrahend
            subtrahend = new_m
            new_m = tf.cast(new_m, var.dtype)
            updates.append(tf.assign(m, new_m, use_locking=self._use_locking))
        new_val = tf.to_float(var) - subtrahend
        if var.dtype == tf.bfloat16:
            new_val = _to_bfloat16_unbiased(new_val)
        if self._simulated_quantize_bits:
            new_val = _simulated_quantize(
                var - subtrahend, self._simulated_quantize_bits,
                self._quantization_noise)
        var_update = tf.assign(var, new_val, use_locking=self._use_locking)
        updates = [var_update] + updates
        return tf.group(*updates)

    def _decay_rate_default(self):
        return adafactor_decay_rate_pow(0.8)

    def _learning_rate_default(self, multiply_by_parameter_scale):
        learning_rate = tf.minimum(tf.rsqrt(step_num() + 1.0), 0.01)
        if not multiply_by_parameter_scale:
            learning_rate *= 0.05
        return learning_rate


def adafactor_decay_rate_adam(beta2):
    """Second-moment decay rate like Adam, subsuming the correction factor.
  
    Args:
      beta2: a float between 0 and 1
    Returns:
      a scalar
    """
    t = tf.to_float(tf.train.get_or_create_global_step()) + 1.0
    decay = beta2 * (1.0 - tf.pow(beta2, t - 1.0)) / (1.0 - tf.pow(beta2, t))
    # decay = tf.cond(tf.equal(t, 1.0), lambda: beta2, lambda: decay)
    return decay


def adafactor_decay_rate_pow(exponent):
    """Second moment decay rate where memory-length grows as step_num^exponent.
  
    Args:
      exponent: a float between 0 and 1
    Returns:
      a scalar
    """
    return 1.0 - tf.pow((step_num() + 1.0), -exponent)


def step_num():
    return tf.to_float(tf.train.get_or_create_global_step())


def adafactor_optimizer_from_hparams(hparams, lr):
    """Create an Adafactor optimizer based on model hparams.
  
    Args:
      hparams: model hyperparameters
      lr: learning rate scalar.
    Returns:
      an AdafactorOptimizer
    Raises:
      ValueError: on illegal values
    """
    if hparams.optimizer_adafactor_decay_type == "Adam":
        decay_rate = adafactor_decay_rate_adam(
            hparams.optimizer_adafactor_beta2)
    elif hparams.optimizer_adafactor_decay_type == "pow":
        decay_rate = adafactor_decay_rate_pow(
            hparams.optimizer_adafactor_memory_exponent)
    else:
        raise ValueError("unknown optimizer_adafactor_decay_type")
    return AdafactorOptimizer(
        multiply_by_parameter_scale=(
            hparams.optimizer_adafactor_multiply_by_parameter_scale),
        learning_rate=lr,
        decay_rate=decay_rate,
        beta1=hparams.optimizer_adafactor_beta1,
        clipping_threshold=hparams.optimizer_adafactor_clipping_threshold,
        factored=hparams.optimizer_adafactor_factored,
        simulated_quantize_bits=getattr(
            hparams, "simulated_parameter_quantize_bits", 0),
        use_locking=False,
        name="Adafactor")


def reduce_rms(x):
    return tf.sqrt(tf.reduce_mean(tf.square(x)))


def _simulated_quantize(x, num_bits, quantization_noise):
    """Simulate quantization to num_bits bits, with externally-stored scale.
  
    num_bits is the number of bits used to store each value.
    quantization_noise is a float32 Tensor containing values in [0, 1).
    Each value in quantization_noise should take different values across
    different steps, approximating a uniform distribution over [0, 1).
    In the case of replicated TPU training, quantization_noise should be identical
    across replicas in order to keep the parameters identical across replicas.
  
    The natural choice for quantization_noise would be tf.random_uniform(),
    but this is not possible for TPU, since there is currently no way to seed
    the different cores to produce identical values across replicas.  Instead we
    use _quantization_noise_from_step_num() (see below).
  
    The quantization scheme is as follows:
  
    Compute the maximum absolute value by row (call this max_abs).
    Store this either in an auxiliary variable or in an extra column.
  
    Divide the parameters by (max_abs / (2^(num_bits-1)-1)).  This gives a
    float32 value in the range [-2^(num_bits-1)-1, 2^(num_bits-1)-1]
  
    Unbiased randomized roundoff by adding quantization_noise and rounding down.
  
    This produces a signed integer with num_bits bits which can then be stored.
  
    Args:
      x: a float32 Tensor
      num_bits: an integer between 1 and 22
      quantization_noise: a float Tensor broadcastable to the shape of x.
  
    Returns:
      a float32 Tensor
    """
    shape = x.get_shape().as_list()
    if not (len(shape) >= 2 and shape[-1] > 1):
        return x
    max_abs = tf.reduce_max(tf.abs(x), -1, keep_dims=True) + 1e-9
    max_int = 2 ** (num_bits - 1) - 1
    scale = max_abs / max_int
    x /= scale
    x = tf.floor(x + quantization_noise)
    # dequantize before storing (since this is a simulation)
    x *= scale
    return x


def _quantization_noise_from_step_num():
    """A quantization noise equal to (phi * (step_num + 1)) mod 1.0.
  
    See _simulated_quantize.
  
    Returns:
      a float32 scalar
    """
    step = tf.to_int32(tf.train.get_or_create_global_step()) + 1
    phi = ((5 ** 0.5) - 1) / 2
    # Naive computation tf.mod(phi * step, 1.0) in float32 would be disastrous
    # due to loss of precision when the step number gets large.
    # Computation in doubles does not work on TPU, so we use this complicated
    # alternative computation which does not suffer from these roundoff errors.
    ret = 0.0
    for i in xrange(30):
        ret += (((phi * (2 ** i)) % 1.0)  # double-precision computation in python
                * tf.to_float(tf.mod(step // (2 ** i), 2)))
    return tf.mod(ret, 1.0)


def _randomized_roundoff_to_bfloat16(x, quantization_noise, cand1, cand2):
    """Round-off x to cand1 or to cand2 in an unbiased way.
  
    Cand1 and cand2 are the same shape as x.
    For every element of x, the corresponding elements of cand1 and cand2 should
    be the two closest bfloat16 values to x.  Order does not matter.
    cand1 and cand2 must differ from each other.
  
    Args:
      x: A float32 Tensor.
      quantization_noise: A Tensor broadcastable to the shape of x containing
      random uniform values in [0.0, 1.0].
      cand1: A bfloat16 Tensor the same shape as x.
      cand2: A bfloat16 Tensor the same shape as x.
  
    Returns:
      A bfloat16 Tensor.
    """
    cand1_f = tf.to_float(cand1)
    cand2_f = tf.to_float(cand2)
    step_size = cand2_f - cand1_f
    fpart = (x - cand1_f) / step_size
    ret = tf.where(tf.greater(fpart, quantization_noise), cand2, cand1)
    return ret


def _to_bfloat16_unbiased(x):
    """Convert a float32 to a bfloat16 using randomized roundoff.
  
    Note: If this ever produces worse results than using float32 all the way
    through, we should try to diagnose and fix it.  There are several things
    to try:
  
    1. Encode parameter x for storage purposes as
       _to_bfloat16_unbiased(tf.pow(x, 5)) .  This gives 5x the
       resolution while incurring overflow and underflow at 10^9 and 10^-9
       instead of 10^37 and 10^-37.  Comes at a cost of extracting fifth roots
       to decode parameters.  Or use some other such scheme.
  
    2. In this function, use actual random numbers, different for each parameter
       as opposed to the same for every parameter in the graph.
  
    3. Look for bugs in this function.
  
    Args:
      x: A float32 Tensor.
    Returns:
      A float32 Tensor.
    """
    # Not using random_uniform here due to a problem on TPU in that random seeds
    # are not respected, which may cause the parameters on different replicas
    # to go out-of-sync.
    quantization_noise = _quantization_noise_from_step_num()
    x_sign = tf.sign(x)
    # Make sure x is positive.  If it is zero, the two candidates are identical.
    x = x * x_sign + 1e-30
    cand1 = tf.to_bfloat16(x)
    cand1_f = tf.to_float(cand1)
    # This relies on the fact that for a positive bfloat16 b,
    # b * 1.005 gives you the next higher bfloat16 and b*0.995 gives you the
    # next lower one. Both 1.005 and 0.995 are ballpark estimation.
    cand2 = tf.to_bfloat16(
        tf.where(tf.greater(x, cand1_f), cand1_f * 1.005, cand1_f * 0.995))
    ret = _randomized_roundoff_to_bfloat16(x, quantization_noise, cand1, cand2)
    return ret * tf.to_bfloat16(x_sign)
