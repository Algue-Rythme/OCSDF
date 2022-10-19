"""Training loop with Metric One Class Learning."""

import tensorflow as tf
import tqdm

from ocml.priors import uniform_sampler_tabular
from ocml.evaluate import plot_metrics_short


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

def compute_batch_norm(vec, squared=False):
  """Return the norm of each vector in the batch."""
  square_norm = tf.reduce_sum(vec ** 2., axis=tuple(range(1, len(vec.shape))), keepdims=True)
  if squared:
    return square_norm
  return square_norm ** 0.5

def newton_raphson(model,
                   Q0,
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
      Q0: tensor of shape (B, F) of initial proposal for adversarial samples. They will be improved by following the gradient of the classifier.
      gen: tf.random.Generator for reproducibility and speed.
      maxiiter: number of iterations of the Newton-Raphson algorithm.
      domain: tuple of two tensors of shape (F,) corresponding to the lower and upper bounds of the domain. Or tuple of two floats.
      infos: return useful infos for plot and debuging (Default: False).
    
    Returns:
        tensor of shape (B, F).
    """
    
    if maxiter == 0:
      if infos:
        y_Q0 = model(Q0, training=False) 
        return Q0, (float(-1.), y_Q0)
      return Q0
  
    step_size = 1. / maxiter  # normalizing factors to tune number of steps independantly.

    if not deterministic:
      shape = (Q0.shape[0], 1)
      lr = gen.uniform(shape=shape, minval=0., maxval=1.)  # random step size.
      step_size = step_size * lr

    Qt = Q0
    
    for step in range(maxiter):
        
      with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(Q0)
        y = model(Q0, training=False)  # estimate the score of the current sample.
      
      if step == 0:  
        y_Q0 = y  # Useful for debuging only.
      
      # Retrieve gradients of the score wrt the input.
      grad  = tape.gradient(y, Q0)
      grad_norm_squared = compute_batch_norm(grad, squared=True)
      grad  = grad / (grad_norm_squared + 1e-8)
      
      # Broadcasting.
      shape = (len(Q0),)+(1,)*(len(Q0.shape)-1)
      y     = tf.reshape(y, shape)
      
      # Level set we target.
      target = -y-level_set
      if overshoot_boundary:
        target = tf.nn.relu(target)  # do not go back to the boundary and stay inside support.
      
      # Perform one step.
      Q_next  = Qt + step_size * target * grad
      
      # Ensure it remains inside the domain of interest - stabilize training.
      Q_next = tf.clip_by_value(Q_next, domain[0], domain[1])
      
      # Use current adversarial example as starting point for next iterations.
      Qt = Q_next
    
    if infos:
      return Qt, (float(tf.reduce_mean(grad_norm_squared**0.5)), y_Q0)
    return Qt

# No need to compile if not using graph mode.
def train_step(model, opt, loss_fn, x_batch, gen, maxiter, domain, pgd):
  """Perform one step of training on the model.

  This function is meant to be compiled with tf.function for speed.
  
  Args:
    model: Lipschitz model.
    opt: optimizer.
    loss_fn: loss function.
    x_batch: tensor of shape (B, F) of initial proposal for adversarial samples. They will be improved by following the gradient of the classifier.
    gen: tf.random.Generator for reproducibility and speed.
    maxiiter: number of iterations of the Newton-Raphson algorithm.
    domain: tuple of two tensors of shape (F,) corresponding to the lower and upper bounds of the domain. Or tuple of two floats.
    pgd: boolean - use PGD instead of Newton-Raphson.
    
  Returns:
    tuple of (loss, infos) where loss is a scalar and infos is a tuple of useful infos for debuging.
  """
  # generate following uniform distribution.
  seeds = uniform_sampler_tabular(gen, len(x_batch), x_batch.shape[1:], domain)
  
  # generate Q_t.
  Qt, (grad_norm, y_Q0) = newton_raphson(model, seeds, gen, maxiter=maxiter, domain=domain, infos=True)
  
  weights = model.trainable_weights
  with tf.GradientTape() as tape:
    y_Qt = model(Qt, training=True)  # forward negative samples from Q.
    y_P  = model(x_batch, training=True)  # forward positive samples from P.
    loss_in  = tf.reduce_mean(loss_fn(y_P))  # positive labels.
    loss_adv = tf.reduce_mean(loss_fn(-y_Qt))  # negative labels.
    loss     = loss_in + loss_adv  # optimize loss on all samples.
  
  grad = tape.gradient(loss, weights)
  opt.apply_gradients(zip(grad, weights))
  
  if pgd:
    model.condense()  # Projected gradient (does not work well).

  return loss, (y_Qt, y_P, y_Q0, grad_norm)


# Pre-compile in global scope to avoid un-necessary re-compilation.
train_step_compiled = tf.function(train_step)



def train_loop(model, opt, loss_fn, gen, dataset, epoch_length, domain, maxiter, *,
               graph_mode=True, pgd=False, plot_wandb=True):
  """Perform an epoch of training on the model.

  WARNING: only support tabular sampler for now. Needs further re-factoring to suppport images.
  
  Args:
    model: Lipschitz model.
    opt: optimizer instance.  
    loss_fn: loss object.
    gen: tf.random.Generator for reproducibility and scaling of randomness.
    dataset: tf.Dataset that yields an infinite sequence of batchs (or at least `epoch_length` steps).
    epoch_length: number of gradient step in the epoch.
  """
  train_step_fn = train_step_compiled if graph_mode else train_step
  with tqdm.tqdm(total=epoch_length) as pb:
    losses = []
    for step, x_batch in zip(range(epoch_length), dataset):  # do not replace with `enumerate` as it will not work with infinite dataset.
        
      loss, infos = train_step_fn(model, opt, loss_fn, x_batch, gen, maxiter, domain, pgd)
      
      losses.append(loss.numpy())
      plot_metrics_short(pb, losses, infos, plot_wandb=plot_wandb)  # update metrics in tqdm.
      pb.update()
