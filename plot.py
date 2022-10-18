"""Plot with Seaborn, Matplotlib or Plotly useful vizualisations."""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from scipy import stats
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

from evaluate import calibrate


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

def plot_2D_contour(model, loss_fn, X,
                    domain,
                    *,
                    histogram=True,  # vizualize histogram
                    plot_seeds=None,  # plot initial seeds of adversarial sampling (for debuging) 
                    save_file=True,  # save the plot in a file
                    plot_wandb=True,  # also upload to wandb
                    height=1.):  # domain on which to plot the data.
  z_grid, x_coords, y_coords, grid = get_contour(model, domain, resolution=300)
  num_cols = 2 if histogram else 1
  fig = make_subplots(rows=1, cols=num_cols, # shared_xaxes=True, shared_yaxes=True,
                      subplot_titles=['Level Sets', 'Score Histogram'])
  fig.add_trace(go.Contour(z=z_grid, x=x_coords, y=y_coords, ncontours=20), row=1, col=1)
  fig.add_trace(go.Scatter(x=X[:,0], y=X[:,1], mode='markers', marker=dict(size=1.5, color='black')), row=1, col=1)
  if plot_seeds is not None:
      seeds, X_adv = plot_seeds
      fig.add_trace(go.Scatter(x=X_adv[:,0], y=X_adv[:,1], mode='markers', marker=dict(size=1.5, color='red')), row=1, col=1)
      fig.add_trace(go.Scatter(x=seeds[:,0], y=seeds[:,1], mode='markers', marker=dict(size=1.5, color='red', symbol='cross')), row=1, col=1)
  if histogram:
      y_pred_neg = model(tf.constant(X_adv), training=True).numpy().flatten()
      y_pred_pos = model(tf.constant(X), training=True).numpy().flatten()
      fig.add_trace(go.Histogram(x=y_pred_neg, name='out-distribution', histnorm='probability', marker_color='red'), row=1, col=2)
      fig.add_trace(go.Histogram(x=y_pred_pos, name='in-distribution', histnorm='probability', marker_color='blue'), row=1, col=2)
      extra_infos = True
      if extra_infos:
          fig.add_shape(type="rect",
            x0=0, y0=0, x1=loss_fn.margin, y1=height*height,
            line=dict(
                color="blue",
                width=2,
            ),
            fillcolor="MediumSlateBlue",
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
  model_weights_path = "model_weights.h5"
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

def plot_gan(epoch, model, advs_out, advs_in, adv_gan, *, save_file=False):
  """Plot the GAN adversarial samples, the negative data augmentation, along original images."""
  y_advs_out = model(advs_out, training=True)
  y_advs_in  = model(advs_in , training=True)
  y_advs_gan  = model(adv_gan , training=True)
  print(f"Generated y_advs_out={y_advs_out.numpy().mean()}  y_advs_in={y_advs_in.numpy().mean()} y_adv_gan={y_advs_gan.numpy().mean()}")
  fig = plot_imgs_grid(advs_out, 'advs_out_{epoch}.png', save_file=save_file)
  fig.show()
  fig = plot_imgs_grid(advs_in,  'advs_in_{epoch}.png', save_file=save_file)
  fig.show()
  fig = plot_imgs_grid(adv_gan,  f'gan_{epoch}.png', save_file=save_file)
  fig.show()

def plot_preds_ood(epoch, model, X_train, X_test, X_ood, *, save_file=True, plot_ood=False, plot_wandb=True):
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
  if plot_ood:
    filename = os.path.join("images", "mnist_ood.png")
    return plot_imgs_grid(X_ood, filename)