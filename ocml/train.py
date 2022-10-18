"""Training loop with Metric One Class Learning."""

import tensorflow as tf
import tqdm

from priors import uniform_sampler_tabular
from evaluate import plot_metrics_short


class SH_KR:
  def __init__(self, margin, lbda):
    self.margin = margin  # must be small but not too small.
    self.lbda   = lbda  # must be high.
  def __call__(self, y):
    """Return loss.
    
    Args:
        y: vector of predictions of shape (B,).
    """
    return  tf.nn.relu(self.margin - y) + (1./self.lbda) * tf.reduce_mean(-y)

class BCE():  # fake cross-entropy loss with useless `margin` parameter for adversarial generation.
  def __call__(self, y):
    """Return loss.
    
    Args:
        y: vector of predictions of shape (B,).
    """
    labels = tf.ones_like(y)  # symmetry property allows this (specific to sigmoid).
    return tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=y)

def compute_batch_norm(vec):
  square_norm = tf.reduce_sum(vec ** 2., axis=tuple(range(1, len(vec.shape))), keepdims=True)
  return square_norm ** 0.5

def newton_raphson(model,
                   x_batch,
                   gen,
                   maxiter,
                   domain, 
                   level_set=0.,
                   deterministic=False,
                   overshoot_boundary=False,
                   *, infos=False):
    """Return a batch of adversarial samples - distribution Q_t in paper.
    
    Args:
      model: classifier - function f_t in paper.
      x_batch: tensor of shape (B, F) of initial proposal for adversarial samples. They will be improved by following the gradient of the classifier.
      gen: tf.random.Generator for reproducibility and speed.
      maxiiter: number of iterations of the Newton-Raphson algorithm.
      domain: tuple of two tensors of shape (F,) corresponding to the lower and upper bounds of the domain. Or tuple of two floats.
      infos: return useful infos for plot and debuging (Default: False).
    
    Returns:
        tensor of shape (B, F).
    """
    
    if maxiter == 0:
      if infos:
        y_seed = model(x_batch, training=False) 
        return x_batch, (float(-1.), y_seed)
      return x_batch
    
    step_size = 1. / maxiter  # normalizing factors to tune number of steps independantly.
    
    for step in range(maxiter):
        
      with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(x_batch)
        y = model(x_batch, training=False)  # estimate the score of the current sample.
      
      if step == 0:  
        y_seed = y  # Useful for debuging only.
      
      # Retrieve gradients of the score wrt the input.
      grad  = tape.gradient(y, x_batch)
      grad_norm = compute_batch_norm(grad)
      grad  = grad / (grad_norm)**2
      
      # Broadcasting.
      shape = (len(x_batch),)+(1,)*(len(x_batch.shape)-1)
      y     = tf.reshape(y, shape)
      
      # Level set we target.
      target = -y-level_set  # tf.nn.relu ?
      if overshoot_boundary:
        target = tf.nn.relu(target)  # do not go back to the boundary.
      
      if not deterministic:
        target = gen.uniform(shape=shape, minval=tf.zeros_like(y), maxval=target)
      
      # Perform one step.
      advs  = x_batch + step_size * target * grad
      
      # Ensure it remains inside the domain of interest - stabilize training.
      advs = tf.clip_by_value(advs, domain[0], domain[1])
      
      # Use current adversarial example as starting point for next iterations.
      x_batch = advs
    
    if infos:
      return advs, (float(tf.reduce_mean(grad_norm)), y_seed)
    return advs

def train_loop(model, opt_cls, loss_fn, gen, dataset, epoch_length, domain, maxiter, *,
               pgd=False):
  """Perform an epoch of training on the model.
  
  Args:
    model: Lipschitz model.
    opt_cls: optimizer class to be instantiated during current epoch. Re-init momentum.  
    loss_fn: loss object.
    gen: tf.random.Generator for reproducibility and scaling of randomness.
    dataset: tf.Dataset that yields an infinite sequence of batchs (or at least `epoch_length` steps).
    epoch_length: number of gradient step in the epoch.
  """
  opt = opt_cls()  # instantiation with default parameters for current epoch
  with tqdm.tqdm(total=epoch_length) as pb:
    losses = []
    for step, x_batch in zip(range(epoch_length), dataset):
        
      # generate following uniform distribution.
      seeds = uniform_sampler_tabular(gen, len(x_batch), x_batch.shape[1:], domain)
      
      # generate Q_t.
      advs, (grad_norm, y_seed) = newton_raphson(model, seeds, gen, maxiter=maxiter, domain=domain, infos=True)
      
      weights = model.trainable_weights
      with tf.GradientTape() as tape:
        y_adv = model(advs, training=True)  # forward negative samples from Q.
        y_in  = model(x_batch, training=True)  # forward positive samples from P.
        loss_in  = tf.reduce_mean(loss_fn(y_in))  # positive labels.
        loss_adv = tf.reduce_mean(loss_fn(-y_adv))  # negative labels.
        loss     = loss_in + loss_adv  # optimize loss on all samples.
      
      grad = tape.gradient(loss, weights)
      opt.apply_gradients(zip(grad, weights))
      
      if pgd:
        model.condense()  # Projected gradient (does not work well).
      
      losses.append(loss.numpy())
      plot_metrics_short(pb, losses, y_adv, y_in, y_seed, grad_norm)  # update metrics in tqdm.
      pb.update()