from types import SimpleNamespace

def get_config(debug=False):
  config = SimpleNamespace(
    num_pts = 4000,
    dataset_name = "two-moons",
    optimizer = 'rmsprop',  # optimizer; good default value.
    batch_size = 256,  # should be not too small to ensure diversity.
    domain = [-5., 5.],  # domain on which to sample points.
    maxiter = 4,  # very important on high dimensional dataset.
    margin = 0.05,  # very important !
    lbda = 100.,  # important but not as much as `margin`. Must be high for best results.
    k_coef_lip = 1.,  # no reason to change this.
    noise = 0.05,  # to introduce noise in dataset (for better plots)
    num_epochs = 50 if not debug else 1,
    spectral_dense = True,  # Mandatory for orthogonal networks. 
    pgd = False,  # Does not work well.
    deterministic = False,  # Better with random learning rates.
    domain_clip = False,  # To ensure it remains in the domain. Not so important.
    boundary = True,  # Whether to stop at the boundary or at the margin (not very different).
    conventional = False,  # Conventional training (i.e without hKR and Lipschitz constraint) for sanity check.
  )
  return config
