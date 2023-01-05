"""Utility loaders for various datasets."""

from functools import partial
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import sklearn.preprocessing as preprocessing
from sklearn.datasets import make_moons, make_circles, make_blobs
import pandas as pd

# Transform Numpy dataset into Tf.Dataset.
def build_ds_from_numpy(x, batch_size):  # create a tf.Dataset from numpy array
  """Build a tf.dataset from np.array of shape (N, F) with some batch_size.

  Perform automatic shuffling.
  """
  x = np.random.permutation(x).astype('float32')
  ds = tf.data.Dataset.from_tensor_slices(x)
  to_shuffle = 2
  ds = ds.repeat().shuffle(to_shuffle*batch_size).batch(batch_size)
  ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
  return ds

# Rescale input and change labels.
def default_process(X, y):
  """Pre-processed dataset.

  Args:
      X: dataset of shape (N, F) with N the number of samples, F the dimension.
      y: vector of shape (N,) of binary labels.

  Returns: 
      pre-processed pair (X, y).
  """
  y = 2*y - 1
  scaler = preprocessing.StandardScaler()
  X = scaler.fit_transform(X)
  return X, y

def load_toy_2d(name, num_pts, noise):
  """Return a numpy toy dataset.

  Args:
    name: string, name of the dataset.
    num_pts: number of examples in the train set.
    noise: scale of the noise for randomness.

  Return:
    pair of numpy arrays (X, y).
  """
  blobs_params = dict(random_state=0, n_samples=num_pts, n_features=2)
  blob_noise = noise*10
  datasets = {
    "one-blob": partial(make_blobs, centers=[[0, 0], [0, 0]], cluster_std=blob_noise, **blobs_params),
    "two-blob": partial(make_blobs, centers=[[2, 2], [-2, -2]], cluster_std=[blob_noise,blob_noise], **blobs_params),
    "two-blob-unbalanced": partial(make_blobs, centers=[[2, 2], [-2, -2]], cluster_std=[blob_noise*3, blob_noise*0.6], **blobs_params),
    "two-circles": partial(make_circles, n_samples=num_pts, noise=noise),
    "two-moons": partial(make_moons, n_samples=num_pts, noise=noise)
  }
  ds_fun = datasets[name]
  return ds_fun()  # create dataset as numpy array.

def preprocess_mnist(image, label, domain):
  """Renormalize images in suitable range."""
  image = tf.cast(image, tf.float32) / 255.  # [0, 1] range
  domain_range = domain[1] - domain[0] 
  image = image * domain_range # [0, 1] -> [0, domain_range]
  image = image - domain_range*0.5 # [0, domain_range] -> [-domain_range/2, domain_range/2]
  return image

def preprocess_cifar10(image, label, domain):
  """Renormalize images in suitable range."""
  image = tf.cast(image, tf.float32) / 255.  # [0, 1] range
  if domain is None:
    image = image - tf.constant([0.49139968, 0.48215827, 0.44653124], dtype=tf.float32)  # [0, 1] -> [-0.4914, 0.5086] for red channel
    image = image / tf.constant([0.24703233, 0.24348505, 0.26158768], dtype=tf.float32)  # [-0.4914, 0.5086] -> [-2.0, 2.0] for red channel (approx)
  return image

def filter_labels(white_list):
  """Select images belonging to a white list of labels."""
  white_list = tf.constant(white_list, dtype=tf.int64)
  def filter_fun(image, label):
    return tf.math.reduce_any(tf.equal(label, white_list))
  return filter_fun

def build_ds(ds_name, batch_size, in_labels, domain, split='train', preprocess_fun=None):
  """Convert dataset from Tf repository into iterable tf.Dataset."""
  ds = tfds.load(ds_name, split='test' if split == 'ood' else split, as_supervised=True, shuffle_files=True)
  if split in ['train', 'test'] :
    label_set = in_labels
  elif split == 'ood':
    label_set = list(set(range(0, 10)).difference(in_labels))
  ds = ds.filter(filter_labels(label_set))
  if preprocess_fun is None:
    preprocess_fun = preprocess_mnist
  ds = ds.map(partial(preprocess_fun, domain=domain))
  to_shuffle = 2
  if split == 'train':
    ds = ds.repeat().shuffle(to_shuffle*batch_size)  # always repeat a dataset
  ds = ds.batch(batch_size).prefetch(4)
  return ds

def tf_from_tfds(ds):
  """Build a numpy dataset from a tf.Dataset."""
  X = tf.concat([batch for batch in ds], axis=0)
  return X

def tfds_from_sampler(sampler, gen, batch_size, input_shape, **kwargs):
  def generator():
    while True:
      yield sampler(gen, batch_size, input_shape, **kwargs)
  output_signature = tf.TensorSpec(shape=(batch_size,)+input_shape, dtype=tf.float32)
  ds = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
  return ds.prefetch(tf.data.experimental.AUTOTUNE)

def zip_ds(*ds):
  zipped = tf.data.Dataset.zip(ds)
  full_batch = zipped.map(lambda *t: tf.concat(t, axis=0))
  return full_batch
