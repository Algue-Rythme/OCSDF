"""Plot with Seaborn, Matplotlib or Plotly useful vizualisations."""

import os
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
                    save_file=True,  # save the plot in a file
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
      save_file: save the plot and the weights in a file.
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
            x0=0, y0=0, x1=loss_fn.margin, y1=height*height,
            line=dict(
                color="black",
                width=2,
            ),
            fillcolor="black",
            opacity=0.3, row=1, col=2,
          )
          fig.add_shape(type="rect",
            x0=-loss_fn.margin, y0=height*height, x1=0, y1=0,
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
  if plot_wandb and save_file:
    import wandb
    filename = f"images/contour.png"
    fig.write_image(filename)
    wandb.save(filename)
    wandb.save(model_weights_path)
  return fig

def plot_imgs_grid(batch, filename,
                  num_rows=2, num_cols=8,
                  *,
                  plot_wandb=True,
                  save_file=False):
  """Plot a batch of images on a grid."""
  to_plot = num_rows * num_cols
  fig = make_subplots(rows=num_rows, cols=num_cols, x_title=filename.split(".")[0])
  images = batch[:to_plot]
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
  return fig

def plot_gan(epoch, model, P, Q0, gen, maxiter, *, save_file=False, **kwargs):
  """Plot the GAN adversarial samples, the negative data augmentation, along original images."""
  deterministic = kwargs.get("deterministic", False)
  kwargs = {k:v for (k,v) in kwargs.items() if k not in ['log_metrics_fn', 'deterministic']}
  y_P = model(P, training=True)
  level_set = float(tf.reduce_mean(y_P).numpy())
  Qt   = newton_raphson(model, Q0, gen, maxiter=maxiter,     level_set=0,         deterministic=deterministic, **kwargs)
  Qinf = newton_raphson(model, Q0, gen, maxiter=maxiter*100, level_set=level_set, deterministic=True,          **kwargs)
  y_Q0    = model(Q0   , training=True)
  y_Qt    = model(Qt   , training=True)
  y_Qinf  = model(Qinf , training=True)
  print(f"P={y_P.numpy().mean()} y_Q0={y_Q0.numpy().mean()}  y_Qt={y_Qt.numpy().mean()} y_Qinf={y_Qinf.numpy().mean()}")
  fig = plot_imgs_grid(Q0, f'y_Q0_{epoch}.png', save_file=save_file)
  fig.show()
  fig = plot_imgs_grid(Qt, f'y_Qt_{epoch}.png', save_file=save_file)
  fig.show()
  fig = plot_imgs_grid(Qinf,  f'y_Qinf_{epoch}.png', save_file=save_file)
  fig.show()

def plot_preds_ood(epoch, model, X_train, X_test, X_ood, *, plot_histogram=False, plot_wandb=True, save_file=True):
  """Plot the predictions of the model on the in-distribution and out-of-distribution data."""
  y_train    = model(tf.constant(X_train), training=True).numpy().flatten()
  y_test     = model(tf.constant(X_test), training=True).numpy().flatten()
  y_ood      = model(tf.constant(X_ood), training=True).numpy().flatten()
  r_train    = stats.describe(y_train, axis=None)
  r_test     = stats.describe(y_test, axis=None)
  r_ood      = stats.describe(y_ood, axis=None)
  print(f"Train examples {r_train}")
  print(f"Test examples {r_test}")
  print(f"OOD examples {r_ood}")
  T_train, acc_train, roc_auc_train = calibrate(y_train, y_ood)
  T_test, acc_test, roc_auc_test = calibrate(y_test, y_ood)
  print(f"[train/ood] Avg Dist={r_train.mean-r_ood.mean} T={T_train} Acc={acc_train}%")
  print(f"[train/ood] ROC-AUC ={roc_auc_train}")
  print(f"[test /ood] Avg Dist ={r_test.mean-r_ood.mean} T={T_test} Acc={acc_test}%")
  print(f"[test /ood] ROC-AUC ={roc_auc_test}")
  if plot_histogram:
    df = pd.DataFrame({'distribution' : ['train'] * len(y_train) + ['test'] * len(y_test) + ['ood'] * len(y_ood),
                        'score'        : np.concatenate([y_train, y_test, y_ood], axis=0)})
    fig = px.histogram(df, x="score", color="distribution", hover_data=df.columns, 
                        histnorm='density', marginal="rug",  # can be `box`, `violin`
                        opacity=0.75)
    fig.add_shape(type="line", x0=T_test, y0=0, x1=T_test, y1=max(len(y_test), len(y_ood)),
                  line=dict(color="black", width=2))
    fig.add_shape(type="line", x0=T_train, y0=0, x1=T_train, y1=max(len(y_train), len(y_ood)),
                  line=dict(color="orange", width=2, dash='dash'))
    fig.show()
    if plot_wandb:
      import wandb
      wandb.log({'roc_auc_train': roc_auc_train, 'T_acc_train': acc_train})
      wandb.log({'roc_auc_test' : roc_auc_test, 'T_acc_test': acc_test})
      if save_file and epoch%10 == 0:
          filename = os.path.join("images", f"ood_histogram_{epoch}.png")
          fig.write_image(filename)
          wandb.save(filename)