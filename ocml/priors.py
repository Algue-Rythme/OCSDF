"""Prior for adversarial sampling and negative data augmentation."""

import abc

import tensorflow as tf
import numpy as np


def compute_batch_norm(vec):
  square_norm = tf.reduce_sum(vec ** 2., axis=tuple(range(1, len(vec.shape))), keepdims=True)
  return square_norm ** 0.5

def uniform_sampler_tabular(gen, batch_size, input_shape, domain):
  """Return a batch of "seeds" uniformly sampled in ball of radius that depends on enclosing domain.
  
  Args:
    gen: tf.random.Generator for reproducibility and speed.
    batch_size: B.
    input_shape: tuple corresponding to the shape of the input.
    domain: tuple of two tensors of shape (F,) corresponding to the lower and upper bounds of the domain. Or tuple of two floats.
  
  Return:
    tensor of shape (B, F).
  """
  dim = np.prod(input_shape)
  radius_max = ((domain[1] - domain[0])/2.)
  radius     = gen.uniform(shape=(batch_size,) + (1,)*len(input_shape), minval=0., maxval=1.)
  radius     = radius ** (1/dim)
  seeds = gen.normal(shape=(batch_size,)+input_shape, mean=0., stddev=1.)
  seeds = seeds / compute_batch_norm(seeds)
  seeds = seeds * radius * radius_max
  return seeds

def uniform_sampler_image(gen, batch_size, input_shape, domain, *, sample_corners=False):
  """Return a batch of "seeds" uniformly sampled in ball of radius that depends on enclosing domain.
  
  Args:
    gen: tf.random.Generator for reproducibility and speed.
    batch_size: B.
    input_shape: tuple corresponding to the shape of the input.
    domain: tuple of two tensors of shape (F,) corresponding to the lower and upper bounds of the domain. Or tuple of two floats.
    sample_corners: sample corners of the domain (Default: False).
  
  Return:
    tensor of shape (B, F).
  """
  seeds = gen.uniform(shape=(batch_size,) + input_shape, minval=domain[0], maxval=domain[1])
  if sample_corners:
    seeds = tf.math.sign(seeds)
  return seeds

class NegativeDataAugmentation:
  """Base class for Negative data augmentation.
  
  Must run code than can be run in graph mode (with tf.function).
  Hence every non deterministic operation must be done with tf.random.Generator.
  """

  @abc.abstractclassmethod
  def __call__(self, gen, x_batch):
    """Return a batch of adversarial samples - distribution Q_t in paper.
    
    Args:
      gen: tf.random.Generator for reproducibility and speed.
      x_batch: tensor of shape (B, F) of initial proposal for adversarial samples. They will be improved by following the gradient of the classifier.
    
    Returns:
        tensor of shape (B, F).
    """
    raise NotImplementedError