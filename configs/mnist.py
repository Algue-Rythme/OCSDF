from types import SimpleNamespace
import math

def get_config(debug=False):
  domain = [-1., 1.]
  margin = (2/100) * (28 * 28 * (domain[1] - domain[0]))**0.5  # 5% pixels for real images
  lbda = 200.  # weak Hinge regularization, less KR.
  config = SimpleNamespace(
      dataset_name = "mnist",
      optimizer = "adam",
      maxiter = 16,
      eta = 10.,
      batch_size = 128,
      domain = domain,
      margin = margin,
      lbda = lbda,
      k_coef_lip = 1.,
      strides = False,
      num_epochs = 1 if debug else 101,
      spectral_dense = True,
      domain_clip = True,
      deterministic = True,
      pooling = True,
      groupsort = False,
      conv_widths = [128, 128, 128],
      dense_widths = [128, 128, 128],
      in_labels = [9]
    )
  return config
