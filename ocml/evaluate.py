"""Utility functions to evaluate and calibrate a model."""

import numpy as np
import tensorflow as tf
import scipy 
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt

from deel.lip.initializers import SpectralInitializer
from deel.lip.layers import SpectralDense, SpectralConv2D, LipschitzLayer, Condensable, InvertibleDownSampling, FrobeniusDense
from deel.lip.activations import MaxMin, GroupSort, GroupSort2, FullSort
from deel.lip.model import Sequential as DeelSequential
from deel.lip.activations import PReLUlip, GroupSort, FullSort
from tensorflow.keras.layers import InputLayer, Dense, Conv2D
from deel.lip.normalizers import reshaped_kernel_orthogonalization

from ocml.layers import infinity_norm_normalization, two_to_infinity_norm_normalization
from ocml.layers import NormalizedDense, NormalizedConv2D



def compute_batch_norm(vec):
  square_norm = tf.reduce_sum(vec ** 2., axis=tuple(range(1, len(vec.shape))), keepdims=True)
  return square_norm ** 0.5

def check_formal_LLC(model, plot_wandb, condense=False):
  if plot_wandb:
    import wandb
    table = wandb.Table(columns=['Singular', 'min-2-inf', 'max-2-inf', 'min-inf', 'max-inf'])
  
  if condense:
    print("WARNING: check_grads performs model.condense()")
    model.condense()

  for layer in model.layers:
    if isinstance(layer, SpectralDense)\
      or isinstance(layer, FrobeniusDense)\
      or isinstance(layer, NormalizedDense)\
      or isinstance(layer, SpectralConv2D)\
      or isinstance(layer, NormalizedConv2D)\
      or isinstance(layer, Dense)\
      or isinstance(layer, Conv2D):
      if isinstance(layer, NormalizedDense) or isinstance(layer, NormalizedConv2D):
        kernel = layer.normalizer_fun(layer.kernel, layer.inf_norm_bounds)
      elif isinstance(layer, FrobeniusDense):
        kernel = layer.kernel / tf.norm(layer.kernel, axis=layer.axis_norm) * layer._get_coef()
      elif type(layer) == Dense or type(layer) == Conv2D:
        kernel = layer.kernel
      else:
        kernel = reshaped_kernel_orthogonalization(
                layer.kernel,
                layer.u,
                layer._get_coef(),
                layer.eps_spectral,
                layer.eps_bjorck,
                layer.beta_bjorck)[0]
            
        reshaped = kernel.numpy().reshape((-1,kernel.shape[-1]))
        singular = scipy.linalg.svdvals(reshaped)
        inf2norm = tf.reduce_sum(reshaped**2, axis=0, keepdims=True) ** 0.5
        infnorm = tf.reduce_sum(tf.math.abs(reshaped), axis=0, keepdims=True)
        datum = [f"{np.max(singular)}", f"{np.min(inf2norm)}", f"{np.max(inf2norm)}", f"{np.min(infnorm)}", f"{np.max(infnorm)}"]
        if plot_wandb:
          table.add_data(*datum)
        print(f"S={datum[0]} 2-inf=[{datum[1],datum[2]}] inf=[{datum[3],datum[4]}] 2=[{np.min(singular),np.max(singular)}]")

  if plot_wandb:
    wandb.log({"Kernel Norms": table})


def check_empirical_LLC(model, seeds, plot_wandb):
  if plot_wandb:
    import wandb

  #Identify example from train set with highest gradient norm (highest LLC).
  with tf.GradientTape(watch_accessed_variables=False) as tape:
    tape.watch(seeds)
    y = model(seeds, training=True)
    grad = tape.gradient(y, seeds)
  grad_norm = compute_batch_norm(grad)
  
  idx = np.argmax(grad_norm)
  print(f"Example n°{int(idx)+1} from train set has norm {float(grad_norm[idx])}.")
  seed = seeds[idx,:]
  seed = seed[np.newaxis,:]
    
  with tf.GradientTape(watch_accessed_variables=False) as tape:
    tape.watch(seed)
    y = model(seed, training=True)
  grad_seed  = tape.gradient(y, seed)
  grad_seed_norm = compute_batch_norm(grad_seed)
    
  print("max_i \|nabla_x f(x_i)\|:", grad_seed_norm)
  if plot_wandb:
      wandb.log({"max_i \|nabla_x f(x_i)\|:": float(np.squeeze(grad_seed_norm.numpy()))})
  
  # Finite difference approximation of the gradient.
  eps = 1e-2
  seed_next = seed + eps * grad_seed
  
  z_next, z = seed_next, seed
  dx = seed_next - seed
  df = dx
  print("LLC = Local Lipschitz Constant")
  print("Input, LLC wrt Input, LLC wrt previous layer")
  datum = ["Input_Norms", f"{np.sum(dx**2)**0.5}", f"{np.sum(dx**2)**0.5}"]
  data = [datum]
  print(datum[0], datum[1], datum[2])
  for i, layer in enumerate(model.layers):
    z_next = layer(z_next, training=True)
    z  = layer(z, training=True)
    dd = df
    df = z_next - z
    dd_ratio = np.sum(df**2)**0.5 / np.sum(dx**2)**0.5
    df_ratio = np.sum(df**2)**0.5 / np.sum(dd**2)**0.5
    datum = [str(type(layer)), f"{dd_ratio:.2f}", f"{df_ratio:.2f}"]
    print(i, datum[0], datum[1], datum[2])
    data.append(datum)
  datum = ["Output_Norms", f"{np.sum(df**2)**0.5}", f"{np.sum(df**2)**0.5}"]
  data.append(datum)
  print(datum[0], datum[1], datum[2])
  
  print(np.min(grad_norm), np.max(grad_norm))
  plt.hist(np.squeeze(grad_norm))
  if plot_wandb:
    wandb.log({"gradients": wandb.Histogram(np.squeeze(grad_norm))})
    columns = ["Layer name", "Global Lipschitz", "Local Lipschitz"]
    table = wandb.Table(data=data, columns=columns)
    wandb.log({"Lipschitz": table})

  plt.show()

def check_LLC(model, seeds, plot_wandb, condense=False):
  check_formal_LLC(model, plot_wandb, condense=condense)
  check_empirical_LLC(model, seeds, plot_wandb)

def calibrate(y_pos, y_neg):
  """Calibrate the model on the test set."""
  y = np.concatenate([y_neg, y_pos], axis=0)
  labels = np.concatenate([np.zeros_like(y_neg), np.ones_like(y_pos)])
  roc_auc = roc_auc_score(labels, y)
  indices = np.argsort(y)
  sorted_labels = labels[indices]
  y_sorted = y[indices]
  forward = np.cumsum(1-sorted_labels)
  backward = np.cumsum(sorted_labels[::-1]) - sorted_labels[-1]
  scores = forward + backward[::-1]
  idx_max = np.argmax(scores)
  return y_sorted[idx_max], (scores[idx_max] / len(scores)) * 100, roc_auc

def log_metrics(pb, losses, infos, plot_wandb=True):
  """Plot useful metrics.
  
  Args:
    pb: progress bar object (from tqdm).
    losses: list of loss per step.
    infos: tuple of useful information.
    plot_wandb: whether to plot on wandb.
  """
  y_Qt, y_P, y_Q0, GN_Qt, lipschitz_ratio = infos
  y_neg_P = tf.constant(-1.)  # TODO: fix this hack.
  recall = tf.reduce_mean(tf.cast(y_P > 0., dtype=tf.float32))
  false_positive = tf.reduce_mean(tf.cast(y_Qt > 0., dtype=tf.float32))
  GN_Qt = float(GN_Qt.numpy())
  pb.set_postfix(R=f'{recall:.2f}%', FP=f'{false_positive:.2f}%',
                  loss=f'{float(np.array(losses).mean()):.3f}',
                  GN_Qt=f'{GN_Qt:.3f}', lipschitz_ratio=f'{float(lipschitz_ratio):.3f}',
                  Qt=f'{float(y_Qt.numpy().mean()):.3f}', P=f'{float(y_P.numpy().mean()):.3f}',
                  Q0=f'{float(y_Q0.numpy().mean()):.3f}', neg_P=f'{float(y_neg_P.numpy().mean()):.3f}')
  if plot_wandb:
    import wandb
    wandb.log({'R':recall, 'FP':false_positive, 'loss':losses[-1],
                'GN_Qt':GN_Qt, 'lipschitz_ratio':lipschitz_ratio,
                'Qt' :float(y_Qt.numpy().mean()), 'P':float(y_P.numpy().mean()),
                'Q0':float(y_Q0.numpy().mean()), 'neg_P' :float(y_neg_P.numpy().mean())})
