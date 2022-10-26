"""Training loop with Metric One Class Learning."""

import tensorflow as tf
import tqdm


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

class BCE():
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
                   *,
                   infos=False,
                   level_set=0.,
                   deterministic=False,
                   overshoot_boundary=True,
                   eta=1.):
    """Return a batch of adversarial samples - distribution Q_t in paper.
    
    Args:
      model: classifier - function f_t in paper.
      Q0: tensor of shape (B, F) of initial proposal for adversarial samples. They will be improved by following the gradient of the classifier.
      gen: tf.random.Generator for reproducibility and speed.
      maxiter: number of iterations of the Newton-Raphson algorithm.
      domain: tuple of two tensors of shape (F,) corresponding to the lower and upper bounds of the domain. Or tuple of two floats.
      level_set: float - level set of the classifier to find.
      deterministic: boolean - do not use random learning rate.
      overshoot_boundary: boolean - allow the adversarial samples to overshoot the boundary of the domain.
      infos: return useful infos for plot and debuging (Default: False).
    
    Returns:
        tensor of shape (B, F).
    """
    
    if maxiter == 0:
      if infos:
        y_Q0 = model(Q0, training=True)
        GN_Qt, lipschitz_ratio = 0., 1.
        return Q0, (y_Q0, GN_Qt, lipschitz_ratio)
      return Q0
  
    if not deterministic:
      shape = (Q0.shape[0],) + (1,)*(Q0.shape.ndims-1)
      lr = gen.uniform(shape=shape, minval=0., maxval=1.)  # random step size.
    else:
      lr = 1.

    step_size = eta / maxiter  # normalizing factors to tune number of steps independently.
    Qt = Q0
    
    for step in range(maxiter):
        
      with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(Qt)
        y = model(Qt, training=(step == 0))  # estimate the score of the current sample.
      
      if step == 0:  
        y_Q0 = y  # Useful for debuging only.
      
      # Retrieve gradients of the score wrt the input.
      grad  = tape.gradient(y, Qt)
      grad_norm_squared = compute_batch_norm(grad, squared=True)
      grad  = grad / (grad_norm_squared + 1e-8)
      
      # Broadcasting.
      shape = (len(Qt),)+(1,)*(len(Qt.shape)-1)
      y     = tf.reshape(y, shape)
      
      # Level set we target.
      target = -y+level_set
      if overshoot_boundary:
        target = tf.nn.relu(target)  # do not go back to the boundary and stay inside support.
      
      # Perform one step.
      Q_next  = Qt + step_size * lr * target * grad
      
      # Ensure it remains inside the domain of interest - stabilize training.
      Q_next = tf.clip_by_value(Q_next, domain[0], domain[1])
      
      # Use current adversarial example as starting point for next iterations.
      Qt = Q_next
    
    if infos:
      grad_norm = tf.reduce_mean(grad_norm_squared**0.5)
      lipschitz_ratio = compute_batch_norm(Qt - Q0) / (tf.reshape(tf.abs(y_Q0), (-1,)+tuple((1,)*(Qt.shape.ndims-1)))+1e-2)
      lipschitz_ratio = tf.reduce_mean(lipschitz_ratio**0.5)
      return Qt, (y_Q0, grad_norm, lipschitz_ratio)
    return Qt

# No need to compile if not using graph mode.
def train_step(model, opt, loss_fn,  p_batch, q_batch, gen, maxiter, domain, **kwargs):
  """Perform one step of training on the model.

  This function is meant to be compiled with tf.function for speed.
  
  Args:
    model: Lipschitz model.
    opt: optimizer.
    loss_fn: loss function.
    p_batch: tensor of shape (B, F) of positive samples.
    q_batch: tensor of shape (B, F) of initial proposal for adversarial samples. They will be improved by following the gradient of the classifier.
    gen: tf.random.Generator for reproducibility and speed.
    maxiter: number of iterations of the Newton-Raphson algorithm.
    domain: tuple of two tensors of shape (F,) corresponding to the lower and upper bounds of the domain. Or tuple of two floats.
    pgd: boolean - use PGD in network step.
    kwargs: additional arguments for newton_raphson.
    
  Returns:
    tuple of (loss, infos) where loss is a scalar and infos is a tuple of useful infos for debuging.
  """
  
  # generate Q_t.
  Qt, infos = newton_raphson(model, q_batch, gen, maxiter=maxiter, domain=domain, infos=True, **kwargs)
  
  weights = model.trainable_weights
  with tf.GradientTape() as tape:
    y_P  = model(p_batch, training=True)  # forward positive samples from P.
    y_Qt = model(Qt,      training=True)  # forward negative samples from Q.
    loss_in  = tf.reduce_mean(loss_fn(y_P))  # positive labels.
    loss_adv = tf.reduce_mean(loss_fn(-y_Qt))  # negative labels.
    loss     = loss_in + loss_adv  # optimize loss on all samples.
  
  grad = tape.gradient(loss, weights)
  opt.apply_gradients(zip(grad, weights))

  return loss, ((y_Qt, y_P,) + infos)


# Pre-compile in global scope to avoid un-necessary re-compilation.
train_step_compiled = tf.function(train_step)


def train(model, opt, loss_fn, gen, P_ds, Q_ds, epoch_length, *,
          domain, maxiter, log_metrics_fn,
          graph_mode=True, **kwargs):
  """Perform an epoch of training on the model.

  WARNING: only support tabular sampler for now. Needs further re-factoring to suppport images.
  
  Args:
    model: Lipschitz model.
    opt: optimizer instance.  
    loss_fn: loss object.
    gen: tf.random.Generator for reproducibility and scaling of randomness.
    P_ds: tf.Dataset that yields an infinite sequence of batchs (or at least `epoch_length` steps) from the positive distribution.
    Q_ds: tf.Dataset that yields an infinite sequence of batchs (or at least `epoch_length` steps) from the negative distribution.
    epoch_length: number of gradient step in the epoch.
    domain: tuple of two tensors of shape (F,) corresponding to the lower and upper bounds of the domain. Or tuple of two floats.
    maxiter: number of iterations of the Newton-Raphson algorithm.
    plot_metrics_fn: function that takes a tuple of (progress_bar, losses, metrics) and plot them.
    graph_mode: boolean - use tf.function for speed.
    **kwargs: additional arguments for the Newton-Raphson algorithm.
  """
  train_step_fn = train_step_compiled if graph_mode else train_step
  with tqdm.tqdm(total=epoch_length) as pb:
    losses = []
    for step, p_batch, q_batch in zip(range(epoch_length), P_ds, Q_ds):  # do not replace with `enumerate` as it will not work with infinite dataset.
        
      loss, infos = train_step_fn(model, opt, loss_fn, p_batch, q_batch, gen, maxiter, domain, **kwargs)
      
      losses.append(loss.numpy())
      log_metrics_fn(pb, losses, infos)  # update metrics in tqdm.
      pb.update()
  pb.close()
