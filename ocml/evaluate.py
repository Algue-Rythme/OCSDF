"""Utility functions to evaluate and calibrate a model."""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score, f1_score
import scipy

import matplotlib.pyplot as plt

from deel.lip.initializers import SpectralInitializer
from deel.lip.layers import SpectralDense, SpectralConv2D, LipschitzLayer, Condensable, InvertibleDownSampling, FrobeniusDense
from deel.lip.activations import MaxMin, GroupSort, GroupSort2, FullSort
from deel.lip.model import Sequential as DeelSequential
from deel.lip.activations import PReLUlip, GroupSort, FullSort
from tensorflow.keras.layers import InputLayer, Dense, Conv2D
from deel.lip.normalizers import reshaped_kernel_orthogonalization

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
    print(layer.name, i, datum[0], datum[1], datum[2])
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


def calibrate_accuracy(y_pos, y_neg):
  """Calibrate the model on the test set."""
  y = np.concatenate([y_neg, y_pos], axis=0)
  labels = np.concatenate([np.zeros_like(y_neg), np.ones_like(y_pos)])
  roc_auc = roc_auc_score(labels, y) * 100
  indices = np.argsort(y)
  sorted_labels = labels[indices]
  y_sorted = y[indices]
  forward = np.cumsum(1-sorted_labels)
  backward = np.cumsum(sorted_labels[::-1]) - sorted_labels[-1]
  scores = forward + backward[::-1]
  idx_max = np.argmax(scores)
  acc = (scores[idx_max] / len(scores)) * 100
  T = y_sorted[idx_max]
  return T, acc, roc_auc


def log_metrics(pb, losses, infos, plot_wandb=True):
  """Plot useful metrics.
  
  Args:
    pb: progress bar object (from tqdm).
    losses: list of loss per step.
    infos: tuple of useful information.
    plot_wandb: whether to plot on wandb.
  """
  y_Qt, y_P, y_Q0, GN_Qt, lipschitz_ratio = infos
  recall = float(tf.reduce_mean(tf.cast(y_P > 0., dtype=tf.float32)).numpy())
  false_positive = float(tf.reduce_mean(tf.cast(y_Qt > 0., dtype=tf.float32)).numpy())
  to_float = lambda t: float(t.numpy().mean())
  y_Qt, y_P, y_Q0, GN_Qt, lipschitz_ratio = tuple(map(to_float, infos))
  pb.set_postfix(R=f'{recall:.2f}%', FP=f'{false_positive:.2f}%',
                  loss=f'{float(np.array(losses).mean()):.3f}',  # in tqdm the average is easier to monitor.
                  GN_Qt=f'{GN_Qt:.3f}', lipschitz_ratio=f'{lipschitz_ratio:.3f}',
                  Qt=f'{y_Qt:.3f}', P=f'{y_P:.3f}',
                  Q0=f'{y_Q0:.3f}')
  if plot_wandb:
    import wandb
    wandb.log({'R':recall, 'FP':false_positive,
                'loss':losses[-1],  # on Wandb it is better to log the last value and average later.
                'GN_Qt':GN_Qt, 'lipschitz_ratio':lipschitz_ratio,
                'Qt' :y_Qt, 'P':y_P, 'Q0':y_Q0})


def compute_precision_recall(ytest, yanomalies, T):
  """Compute precision and recall.
  in: in-class.
  out: anomalies.
  """
  pred_normal = ytest >= T
  pred_anomalies = yanomalies < T
  tp = pred_anomalies.sum()  # true anomaly spoted ! accept anomaly.
  tn = pred_normal.sum()  # true normal spoted ! reject anomaly.
  fn = len(yanomalies) - tp  # it was anomalous... yet it was not spoted.
  fp = len(ytest) - tn  # it was normal... yet it was spoted as anomaly.
  recall_an = tp / (tp + fn + 1e-8) * 100  # TPR True Positive Rate, Recall, Sensivity
  recall_no = tn / (tn + fp + 1e-8) * 100  # TNR=1-FPR True Negative Rate, Specificity, Selectivity 
  precision_an = tp / (tp + fp + 1e-8) * 100  # PPV Predictive Position Value, Precision
  precision_no = tn / (tn + fn + 1e-8) * 100  # FDR False Discovery Rate
  preds = pred_normal, pred_anomalies
  precision_recall = (recall_an, recall_no, precision_an, precision_no)
  return preds, precision_recall, T


def seek_threshold_tabular(ytest, yanomalies, protocol='recall95'):
  """
  Protocols:
  
    r=pOC: Apply protocol of HRN to make precision=recall=F1 at One Class.
    r=pOC: Apply protocol of TQM to make precision=recall=F1 on anomalies.
    recall95OC: Apply protocol for 95% recall at One Class.
    recall95AD: Apply protocol for 95% recall on anomalies.
  """
  a = min(ytest.min(), yanomalies.min())
  b = max(ytest.max(), yanomalies.max())
  def fun(T):
    preds, precision_recall, T = compute_precision_recall(ytest, yanomalies, T)
    recall_an, recall_no, precision_an, precision_no = precision_recall
    if protocol == 'recall95OC':
      return recall_no - 95
    elif protocol == 'recall95AD':
      return recall_an - 95
    elif protocol == 'r=pOC':
      return recall_no - precision_no
    elif protocol == 'r=pAD':
      return recall_an - precision_an
  T = scipy.optimize.bisect(fun, a, b)
  return T


def evaluate_tabular(epoch, model, xtest, anomalies, manual_threshold=None, plot_wandb=True):
    test_size, anomalies_size = len(xtest), len(anomalies)
    xx = np.concatenate([xtest, anomalies], axis=0)
    _ = model(anomalies, training=True) # garbage-in to apply bjorck projection automatically.
    yy = model.predict(xx, verbose=1, batch_size=2048).flatten()
    ytest, yanomalies = np.split(yy, indices_or_sections=[test_size])
    mean_in, std_in = ytest.mean(), ytest.std()
    mean_out, std_out = yanomalies.mean(), yanomalies.std()
    print(f"Mean-In={mean_in:.2f}±{std_in:.2f} Mean-Out={mean_out:.2f}±{std_out:.2f}")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    percentiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    trainstats = pd.DataFrame(ytest).describe(percentiles).transpose()
    teststats = pd.DataFrame(yanomalies).describe(percentiles).transpose()
    stats = pd.concat([trainstats,teststats], ignore_index=True).round(4)
    stats.index = ['test','anomalies']
    if (epoch+1)%5== 0:
      print(stats)
    true_labels = np.concatenate([np.zeros(test_size), np.ones(anomalies_size)], axis=0)
    roc_auc = roc_auc_score(true_labels, -yy) * 100.  # swap snormality <=> anormality score !
    print(f"AUROC={roc_auc:.2f}")
    protocols = ['r=pOC', 'r=pAD', 'recall95OC', 'recall95AD']
    for protocol in protocols:
      if manual_threshold is None:
        threshold = seek_threshold_tabular(ytest, yanomalies, protocol)
      else:
        threshold = manual_threshold
      preds, precision_recall, threshold = compute_precision_recall(ytest, yanomalies, threshold)
      pred_normal, pred_anomalies = preds
      recall_an, recall_no, precision_an, precision_no = precision_recall
      pred_yy = np.concatenate([1-pred_normal, pred_anomalies], axis=0)
      f1 = f1_score(true_labels, pred_yy) * 100
      f1_rev = f1_score(1-true_labels, 1-pred_yy) * 100  # F1-score is not symmetric: we monitore both directions.
      if len(protocols) == 1:
        print(f"False-Alarm={100-recall_no:.2f}%")
      print(f"[{protocol}] Recall-Anomalies={recall_an:.2f} Precision-Anomaly={precision_an:.2f}% Precision-Normal={precision_no:.2f}% Recall-Normal={recall_no:.2f}%")
      print(f"[{protocol}] F1={f1:.2f} F1-rev={f1_rev:.2f} T={threshold:.4f} ")
      if plot_wandb:
        import wandb
        wandb.log({'protocol':protocol, 'roc_auc': roc_auc, 'recall':recall_an, 'precision':precision_an, 'f1':f1, 'f1_rev':f1_rev, 'T':threshold})
    return threshold
