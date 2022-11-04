"""Prior for adversarial sampling and negative data augmentation."""

import abc
from dataclasses import dataclass
from functools import partial

import tensorflow as tf
import numpy as np
import pandas as pd

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
  noise: float = 0.1
  domain: tuple = (-1., 1.)

  def __post_init__(self):
    self.batch_shape = (self.batch_size,) + self.input_shape

  def transform(self, gen, ds):
    def affine(gen, batch):
      ks = np.random.randint(0, 4, size=self.batch_size)
      imgs = tf.unstack(batch, axis=0)
      batch = [tf.image.rot90(img, k=k) for img, k in zip(imgs, ks)]
      batch = tf.stack(batch, axis=0)
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
      t = gen.uniform(shape=(1,), minval=0.25, maxval=0.75)
      mixed = t * batch + (1-t) * other
      salt = salt_and_pepper(gen, batch)
      mixed = mixed + salt
      mixed = tf.clip_by_value(mixed, self.domain[0], self.domain[1])
      return mixed

    py_aug = lambda x: tf.py_function(aug, inp=[x], Tout=tf.float32)

    return ds.map(py_aug, num_parallel_calls=tf.data.experimental.AUTOTUNE)


class Categorical:
  def __init__(self, num_classes):
    self.num_classes = max(num_classes, 2)
  def scale_bounds(self, bounds):
    pass
  def is_outlier(self, x):
    return np.full(x.shape[0], False, dtype=bool)
  def encode(self, x):
    if self.num_classes == 2:
      return x.copy()
    one_hot = np.eye(self.num_classes)[x]
    return one_hot
  def sample(self, batch_size):
    if self.num_classes == 2:
      return np.random.randint(2, size=batch_size)[:,np.newaxis]
    classes = np.random.randint(self.num_classes, size=batch_size)
    one_hot = np.eye(self.num_classes)[classes]
    return one_hot


class Gaussian:
  def __init__(self, mean, std, bounds):
    self.mean = mean
    self.std = std
    self.threshold = bounds
  def scale_bounds(self, bounds):
    self.threshold *= bounds
  def is_outlier(self, x):
    return np.abs(x) > self.threshold * self.std
  def encode(self, x):
    return (x - self.mean) / self.std
  def sample(self, batch_size):
    emp_std = self.threshold * self.std
    return np.random.normal(self.mean, emp_std, (batch_size,1))


class LogUniform:
  def __init__(self, min_v, max_v, shift):
    self.min_v = min_v
    self.max_v = max_v
    self.shift = shift
    self.bounds = 1.
  def scale_bounds(self, bounds):
    self.bounds *= bounds
  def is_outlier(self, x):
    return np.logical_or(x < self.min_v, x > self.max_v)
  def encode(self, x):
    return np.log(x + self.shift)
  def sample(self, batch_size):
    min_v, max_v = self.min_v * self.bounds, self.max_v * self.bounds
    return np.random.uniform(min_v, max_v, (batch_size,1))  # uniform in log space


class TabularSampler:
  """Sampler for tabular data.
  
  Fit the sampler on the training data and sample from a surrogate of the training data.

  The surrogate is a Gaussian with mean and std computed from the training data.
  Categories are encoded with one-hot encoding.

  Bounds are used to detect outliers and to scale the surrogate.

  Differents policies can be used to sample from the surrogate:
    - z_score: fit empirical mean and std.
    - robust_z_score: fit empirical mean and median absolute deviation.
    - logscale: softplus parametrization of input space when different scales are present.
  """
  def __init__(self, bounds, samplers=None):
    self.samplers = [] if samplers is None else samplers
    self.bounds = bounds
    self.shift = 0.1
  def scale_bounds(self, bounds):
    self.bounds *= bounds
    for sampler in self.samplers:
        sampler.scale_bounds(bounds)
  def add(self, sampler):
    self.samplers.append(sampler)
  def check_integrity(self, batch_size, batch_size_ref):
    assert self.sample(batch_size).shape == (batch_size,) + batch_size_ref
  def encode_numeric_zscore(self, df, df_source, df_train, name, mean=None, sd=None):
    if mean is None:
        mean = float(df_train[name].mean())
    if sd is None:
        sd = df_train[name].std()
    if sd < 10 * np.finfo(sd.dtype).eps:
        sd = 1
    df[name] = (df_source[name] - mean) / sd
    bounds = max(df[name].max(), -df[name].min()) / 2
    sampler = Gaussian(0., 1., bounds)
    self.add(sampler)
  def encode_logscale(self, df, df_source, name):
    df[name] = np.log(df_source[name] + self.shift)
    min_v = df[name].min()
    max_v = df[name].max()
    sampler = LogUniform(min_v, max_v, self.shift)
    self.add(sampler)
  def encode_robust_zscore(self, df, df_source, df_train, name, median=None, mad=None):
    """Robust Z-Score for better robustness against outliers in train set."""
    if median is None:
      median = df_train[name].median()
    if mad is None:
      absolute_deviation = (df_train[name] - median).abs()
      mad = absolute_deviation.median()
    if mad < 10 * np.finfo(mad.dtype).eps:
      mad = 1
    df[name] = (df_source[name] - median) / mad * 0.6745
    bounds = max(df[name].max(), -df[name].min()) / 2
    sampler = Gaussian(0., 1., bounds)
    self.add(sampler)
  def encode_text_dummy(self, df, df_source, name):
    uniques = df_source[name].nunique()
    if uniques == 1:
      dummy_name = f"{name}-{df_source[name].iloc[0]}"
      df[dummy_name] = 1.
    elif uniques <= 2:
      dummy_name = f"is-{name}"
      dummies = pd.get_dummies(df_source[name], drop_first=True)
      df[dummy_name] = dummies[list(dummies.columns)[0]]
    else:  # No sparse when more than 1 class to ensure same distance between everyone
      dummies = pd.get_dummies(df_source[name])
      for x in dummies.columns:
        dummy_name = f"{name}-{x}"
        df[dummy_name] = dummies[x]
    sampler = Categorical(uniques)
    self.add(sampler)
  def fit_transform(self, df_source, df_train, continuous_policy, discrete_cols=[]):
    assert continuous_policy in ['robust', 'logscale', 'zscore']
    cols = list(df_source.columns)
    df = pd.DataFrame(index=df_source.index)
    for col in cols:
      if col == 'label':
        continue
      if col in discrete_cols:
        self.encode_text_dummy(df, df_source, col)
      elif continuous_policy == 'robust':
        self.encode_robust_zscore(df, df_source, df_train, col)
      elif continuous_policy == 'logscale':
        self.encode_logscale(df, df_source, col)
      elif continuous_policy == 'zscore':
        self.encode_numeric_zscore(df, df_source, df_train, col)
    df['label'] = df_source['label'].copy()
    self.check_integrity(16, (df.shape[1]-1,))
    return df
  def sample(self, batch_size):
    samples = [sampler.sample(batch_size) for sampler in self.samplers]
    samples = np.concatenate(samples, axis=1)
    return samples
