"""Prior for adversarial sampling and negative data augmentation."""

import abc
from dataclasses import dataclass
from functools import partial

import tensorflow as tf
import numpy as np

try:
  from perlin_numpy import generate_perlin_noise_2d
except ImportError as e:
  print("Perlin noise not available. Please install perlin_numpy package with `pip3 install git+https://github.com/pvigier/perlin-numpy`.")
  pass


def compute_batch_norm(vec):
  square_norm = tf.reduce_sum(vec ** 2., axis=tuple(range(1, len(vec.shape))), keepdims=True)
  return square_norm ** 0.5

def uniform_tabular(gen, batch_size, input_shape, domain):
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

def uniform_image(gen, batch_size, input_shape, domain, *, sample_corners=False):
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

def perlin_noise(gen, batch_size, input_shape, *, res=4):
  """Return a batch of images sampled from Perlin noise.
  
  Args:
    gen: tf.random.Generator for reproducibility and speed.
    batch_size: B.
    input_shape: tuple corresponding to the shape of the input (H, W, 1).
    res: resolution of the Perlin noise.
    
  Return:
    tensor of shape (B, H, W, 1).
  """
  height, width = input_shape[0], input_shape[1]
  images = []
  for _ in range(batch_size):
    img = generate_perlin_noise_2d((height, width), (res, res))
    images.append(img[..., np.newaxis])
  images = np.stack(images, axis=0)
  return tf.constant(images, dtype=tf.float32)


class NegativeDataAugmentation:
  """Base class for Negative Data Augmentation (NDA).
  
  Must run code than can be run in graph mode (with tf.function).
  Hence every non deterministic operation must be done with tf.random.Generator.
  """

  @abc.abstractclassmethod
  def transform(self, gen, ds):
    """Return a batch of adversarial samples - distribution Q_t in paper.
    
    Args:
      gen: tf.random.Generator for reproducibility and speed.
      ds: tf.data.Dataset that yields batchs of shape (B, F).
    
    Returns:
        tensor of shape (B, F).
    """
    raise NotImplementedError


@dataclass
class Mnist_NDA(NegativeDataAugmentation):
  """NDA for MNIST.

  Attributes:
    noise: float, standard deviation of the noise.
    domain: tuple of two tensors of shape (F,) corresponding to the lower and upper bounds of the domain. Or tuple of two floats.
  """
  batch_size: int
  input_shape: tuple
  noise: float = 0.2
  domain: tuple = (-1., 1.)

  def __post_init__(self):
    self.batch_shape = (self.batch_size,) + self.input_shape

  def transform(self, gen, ds):
    def affine(gen, batch):
      k = np.random.randint(0, 4)
      batch = tf.image.rot90(batch, k=k)
      batch = tf.image.random_flip_left_right(batch)
      batch = tf.image.random_flip_up_down(batch)
      return batch

    def salt_and_pepper(gen, batch):
      img = gen.normal(shape=self.batch_shape)
      img = img * self.noise * (self.domain[1] - self.domain[0])
      return img

    def aug(batch):
      other = tf.random.shuffle(batch)
      other = affine(gen, other)
      batch = affine(gen, batch)
      t = gen.uniform(shape=(1,), minval=0., maxval=1.)
      mixed = t * batch + (1-t) * other
      salt = salt_and_pepper(gen, batch)
      mixed = mixed + salt
      mixed = tf.clip_by_value(mixed, self.domain[0], self.domain[1])
      return mixed

    return ds.map(aug, num_parallel_calls=tf.data.experimental.AUTOTUNE)
