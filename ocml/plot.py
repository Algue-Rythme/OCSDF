"""Plot with Seaborn, Matplotlib or Plotly useful vizualisations."""

import os
import math

import pandas as pd
import numpy as np
import tensorflow as tf
from scipy import stats
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

from ocml.evaluate import calibrate
from ocml.train import newton_raphson


def get_contour(model, domain, resolution):
  """Return grid with level sets from model.
  
  Args:
      model: the classifier to plot
      domain: domain on which to plot the classifier level sets
      resolution: number of pixels in discretization
      
  Returns:
      grid with predictions, x and y coordinnaates of the grid. 
  """
  xmin, xmax = domain[0], domain[1]
  ymin, ymax = domain[0], domain[1]
  x_coords = np.linspace(xmin, xmax, resolution)
  y_coords = np.linspace(ymin, ymax, resolution)
  x_grid, y_grid = np.meshgrid(x_coords, y_coords)
  grid = np.stack([x_grid.flatten(), y_grid.flatten()], axis=-1)
  z_grid = model(grid, training=True).numpy()
  z_grid = z_grid.reshape(x_grid.shape)
  return z_grid, x_coords, y_coords, grid

def plot_2D_contour(model, loss_fn, P,
                    domain,
                    *,
                    histogram=True,  # vizualize histogram
                    plot_Qt=None,  # plot initial Q0 of adversarial sampling (for debuging) 
                    plot_wandb=True,  # also upload to wandb
                    height=1.,
                    ncontours=20):  # domain on which to plot the data.
  """Plot the 2D contour of the classifier and the histogram of the data.
  
  Args:
      model: the classifier to plot.
      loss_fn: the loss function for the margin.
      P: the in-distribution.
      domain: domain on which to plot the classifier level sets.
      histogram: vizualize histogram of predictions.
      plot_Qt: plot initial Q0 of adversarial sampling (for debuging).
      plot_wandb: also upload to wandb.
      height: height bars in the histogram (useless parameter).
      ncontours: number of contours to plot (20 by default).
  """
  z_grid, x_coords, y_coords, grid = get_contour(model, domain, resolution=300)
  num_cols = 2 if histogram else 1
  fig = make_subplots(rows=1, cols=num_cols, # shared_xaxes=True, shared_yaxes=True,
                      subplot_titles=['Level Sets', 'Score Histogram'])
  fig.add_trace(go.Contour(z=z_grid, x=x_coords, y=y_coords, ncontours=ncontours), row=1, col=1)
  fig.add_trace(go.Scatter(x=P[:,0], y=P[:,1], mode='markers', marker=dict(size=1.5, color='black')), row=1, col=1)
  if plot_Qt is not None:
      Q0, Qt = plot_Qt
      fig.add_trace(go.Scatter(x=Qt[:,0], y=Qt[:,1], mode='markers', marker=dict(size=1.5, color='red')), row=1, col=1)
      fig.add_trace(go.Scatter(x=Q0[:,0], y=Q0[:,1], mode='markers', marker=dict(size=1.5, color='red', symbol='cross')), row=1, col=1)
      if histogram:
        y_Qt = model(tf.constant(Qt), training=True).numpy().flatten()
        fig.add_trace(go.Histogram(x=y_Qt, name='out-distribution', histnorm='probability', marker_color='orange'), row=1, col=2)
  if histogram:
      y_P = model(tf.constant(P), training=True).numpy().flatten()
      fig.add_trace(go.Histogram(x=y_P, name='in-distribution', histnorm='probability', marker_color='green'), row=1, col=2)
      extra_infos = True
      if extra_infos:
          fig.add_shape(type="rect",
            x0=0, y0=0, x1=loss_fn.margin, y1=height,
            line=dict(
                color="black",
                width=2,
            ),
            fillcolor="black",
            opacity=0.3, row=1, col=2,
          )
          fig.add_shape(type="rect",
            x0=-loss_fn.margin, y0=height, x1=0, y1=0,
            line=dict(
                color="red",
                width=2,
            ),
            fillcolor="MediumOrchid",
            opacity=0.3, row=1, col=2
          )
          fig.add_shape(type="line",
            x0=loss_fn.margin, y0=0, x1=loss_fn.margin, y1=height,
            line=dict(
                color="blue",
                width=2
            ), row=1, col=2
          )
          fig.add_shape(type="line",
            x0=0, y0=0, x1=0, y1=height,
            line=dict(
                color="black",
                width=2
            ), row=1, col=2
          )
          fig.add_shape(type="line",
            x0=-loss_fn.margin, y0=0, x1=-loss_fn.margin, y1=height,
            line=dict(
                color="red",
                width=2
            ), row=1, col=2
          )
  width = num_cols * 500
  fig.update_layout(autosize=False, width=width, height=500, showlegend=False, barmode='overlay')
  fig.update_yaxes(
      scaleanchor = "x",
      scaleratio = 1,
      row=1,
      col=1,
  )
  fig.update_traces(opacity=0.75, row=1, col=2)
  model_weights_path = os.path.join("weights", "model_weights.h5")
  model.save_weights(model_weights_path)
  if plot_wandb:
    import wandb
    filename = f"images/contour.png"
    fig.write_image(filename)
    wandb.save(filename)
    wandb.save(model_weights_path)
  fig.show()

def plot_imgs_grid(batch, filename,
                  num_rows=2, num_cols=8,
                  *,
                  plot_wandb=True,
                  save_file=False):
  """Plot a batch of images on a grid."""
  to_plot = num_rows * num_cols
  fig = make_subplots(rows=num_rows, cols=num_cols, x_title=filename.split(".")[0])
  batch = batch.numpy() if isinstance(batch, tf.Tensor) else batch
  images = np.concatenate([batch[0::2], batch[1::2]], axis=0)[:to_plot]  # interleave for better viz.
  for i, img in enumerate(images):
    row = i // num_cols + 1
    col = i %  num_cols + 1
    img = np.concatenate((img,)*3, axis=-1) * 255
    trace = go.Image(z=img, zmax=(255,255,255,255), zmin=(-255,-255,-255,-255), colormodel="rgb")
    fig.add_trace(trace, row=row, col=col)
  if save_file and plot_wandb:
    import wandb
    filename = os.path.join("images", filename)
    fig.write_image(filename)
    wandb.save(filename)
  fig.show()

def plot_gan(epoch, model, P, Q0, gen, maxiter, *, save_file=False, plot_wandb=True, **kwargs):
  """Plot the GAN adversarial samples, the negative data augmentation, along original images."""
  _ = model(P[:8], training=True)  # cache computations of (u, sigma) using fake batch of small size.
  y_P = model.predict(P, batch_size=256, verbose=1)
  quantile = 0.5
  avg_P = float(np.quantile(y_P.flatten(), q=quantile))
  keys = ['deterministic', 'level_set', 'overshoot_boundary', 'eta', 'domain']
  kwargs = {k:v for (k,v) in kwargs.items() if k in keys}
  Qt   = newton_raphson(model, Q0, gen, maxiter=maxiter, **kwargs)
  kwargs = {k:v for (k,v) in kwargs.items() if k not in ['level_set', 'deterministic', 'overshoot_boundary']}
  Qinf = newton_raphson(model, Q0, gen, maxiter=maxiter*100, level_set=avg_P, deterministic=True, overshoot_boundary=True, **kwargs)
  y_Q0    = model.predict(Q0)
  y_Qt    = model.predict(Qt)
  y_Qinf  = model.predict(Qinf)
  print(f"P={y_P.mean()} y_Q0={y_Q0.mean()}  y_Qt={y_Qt.mean()} y_Qinf={y_Qinf.mean()}")
  plot_imgs_grid(Q0, f'Q0_{epoch}.png', save_file=save_file, plot_wandb=plot_wandb)
  plot_imgs_grid(Qt, f'Qt_{epoch}.png', save_file=save_file, plot_wandb=plot_wandb)
  plot_imgs_grid(Qinf,  f'Qinf_{epoch}.png', save_file=save_file, plot_wandb=plot_wandb)

def plot_preds_ood(epoch, model, X_train, X_test, X_ood, *, plot_histogram=False, plot_wandb=True):
  """Plot the predictions of the model on the in-distribution and out-of-distribution data."""
  _          = model(X_train[:8], training=True)  # cache computations using fake batch of small size.
  y_train    = model.predict(X_train, batch_size=256, verbose=1).flatten()
  y_test     = model.predict(X_test, batch_size=256, verbose=1).flatten()
  y_ood      = model.predict(X_ood, batch_size=256, verbose=1).flatten()
  r_train    = stats.describe(y_train, axis=None)
  r_test     = stats.describe(y_test, axis=None)
  r_ood      = stats.describe(y_ood, axis=None)
  print(f"Train examples {r_train}")
  print(f"Test examples {r_test}")
  print(f"OOD examples {r_ood}")
  T_train, acc_train, roc_auc_train = calibrate(y_train, y_ood)
  T_test, acc_test, roc_auc_test = calibrate(y_test, y_ood)
  print(f"[train/ood] Avg Dist={r_train.mean-r_ood.mean:.3f} T={T_train:.3f} Acc={acc_train:.2f}%")
  print(f"[train/ood] ROC-AUC ={roc_auc_train:.3f}")
  print(f"[test /ood] Avg Dist ={r_test.mean-r_ood.mean} T={T_test:.3f} Acc={acc_test:.2f}%")
  print(f"[test /ood] ROC-AUC ={roc_auc_test:.3f}")
  model_weights_path = os.path.join("weights", "model_weights.h5")
  model.save_weights(model_weights_path)
  if plot_wandb:
    import wandb
    wandb.log({'roc_auc_train': roc_auc_train, 'T_acc_train': acc_train})
    wandb.log({'roc_auc_test' : roc_auc_test, 'T_acc_test': acc_test})
    wandb.save(model_weights_path)
  if plot_histogram:
    df = pd.DataFrame({'distribution' : ['train'] * len(y_train) + ['test'] * len(y_test) + ['ood'] * len(y_ood),
                        'score'        : np.concatenate([y_train, y_test, y_ood], axis=0)})
    frac = 0.1  # plot 10% of input data (~1000 examples).
    df = df.sample(frac=frac, axis=0)  # produce histogram with small part of input data.
    fig = px.histogram(df, x="score", color="distribution", hover_data=df.columns, 
                        histnorm='density', marginal="rug",  # can be `box`, `violin`
                        opacity=0.75)
    height = round(max(len(y_test), len(y_ood) * frac))
    fig.add_shape(type="line", x0=T_test, y0=0, x1=T_test, y1=height,
                  line=dict(color="black", width=2))
    fig.add_shape(type="line", x0=T_train, y0=0, x1=T_train, y1=height,
                  line=dict(color="orange", width=2, dash='dash'))
    fig.show()
    if plot_wandb:
      import wandb
      filename = os.path.join("images", f"ood_histogram_{epoch}.png")
      fig.write_image(filename)
      wandb.save(filename)
      

def display_levelset_binary(model, X, levels=20, coeff=1.1):
  """Display the level set of the model on the binary classification problem.

  Origin: code copy/pasted from "When adversarial attacks become interpretable counterfactual explanations".
  
  Args:
    model: the model to display.
    X: the input data.
    levels: the number of levels to display.
    coeff: window size coefficient.
  """
  import matplotlib.pyplot as plt
  import seaborn as sns

  x_min = X[:,0].min()*coeff
  x_max = X[:,0].max()*coeff
  y_min = X[:,1].min()*coeff
  y_max = X[:,1].max()*coeff
  x = np.linspace(x_min, x_max, 120)
  y = np.linspace(y_min, y_max,120)
  xx, yy = np.meshgrid(x, y, sparse=False)
  print(xx.min(),xx.max(),yy.min(),yy.max())

  X_pred=np.stack((xx.ravel(),yy.ravel()),axis=1)
  pred = model.predict(X_pred)
  # pred=pred-pred[:,0].mean()
  Y_pred = pred
  Y_pred_f = pred
  Y_pred_f = Y_pred_f.reshape(x.shape[0], y.shape[0])

  fig = plt.figure(figsize=(10, 10))
  ax1 = fig.add_subplot(111)
  ax1.spines['left'].set_position('zero')
  ax1.spines['right'].set_color('none')
  ax1.spines['bottom'].set_position('zero')
  ax1.spines['top'].set_color('none')

  grid_x_ticks = np.arange(x_min, x_max, 0.2)
  grid_y_ticks = np.arange(y_min, y_max, 0.2)
  # ax1.set_ticks_position('both')
  ax1.set_xticks(grid_x_ticks , minor=True)
  ax1.set_yticks(grid_y_ticks , minor=True)
  # ax1.grid(which='both')
  ax1.grid(True, 'major', ls='solid', lw=0.5, color='gray')
  ax1.grid(True, 'minor', ls='solid', lw=0.2, color='gray')
  # ax1.set_minor_locator(mpl.ticker.AutoMinorLocator())
  # ax1.grid(which='minor', alpha=0.3)
  # ax2 = fig.add_subplot(312)
  # ax3 = fig.add_subplot(313)
  sns.scatterplot(x=X[:, 0],y=X[:, 1], color=sns.color_palette()[0], alpha=0.2, ax=ax1)
  cset = ax1.contour(xx, yy, Y_pred_f, cmap='plasma', levels = levels)
  ax1.clabel(cset, inline=1, fontsize=10)
  cset = ax1.contour(xx, yy, Y_pred_f, [0.0], colors='red', linestyles='dashed', linewidths=6)
  ax1.clabel(cset, inline=1, fontsize=14)
  ax1.patch.set_edgecolor('black')
  #plt.show()
  return ax1
