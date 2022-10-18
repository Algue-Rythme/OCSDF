## One Class Metric Learning

Learn to approximate the Sign Distance Function (SDF) to the boundary of a distribution, i.e a dataset.

Use a combination of Lipschitz networks, adversarial training and Hinge Kantorovich Rubinstein loss (HKR).

![2D Toy example](figures/all_methods_grid.PNG)

## Structure of the repository

The repository is organized as follow:
  * `run_*.ipynb` notebooks:
    - `run_toy2d.ipynb` and `run_mnist.ipynb` launchs predefined configurations, datasets, create model, train it, and log the results.
    - `run_toy2d_baselines`: baselines from Scikit-Learn to compare against and reproduce figures from the paper.
  * `ocml`: contains all source files.
    - `train.py`: adversarial generation and main training loop.
    - `priors.py`: prior of complementary distributions and data augmentation.
    - `evaluate.py`: tools to evaluate Local Lipschitz Constant (LLC) of the networks, monitore weights, and calibrate the predictions.
    - `plot.py`: plotting utilities (image, 2D contour plots, images).
    - `models.py`: definition of common Lipschitz and conventional architectures.
    - `layers.py`: additional layers for Lipschitz networks, more compliant with the theorem of Anil et al. (2018).
    - `datasets.py`: pre-processing of common datasets in an unified framework.
  * `draft_notebooks/`: old notebooks for early experiments and prototypes.

### Remarks

`Wandb` is used experiment tracking, `plotly` and `seaborn` are used for plotting. Latest version of `deel-lip` is recommanded.

### Run the code

The following directories will be populated:

  * `images/`: record images produced for uploading to `wandb`.
  * `weights/`: contain weights of the network architecture in `.h5` format.
  * `wandb/`: if wandb is used - to store local variables.

## Contribution 

### Conventions

Newly added functions should contain a docstring that specifies arguments, attributes and return values.

When performance is an issue a tf.function decorator should be added to the outermost function. tf.function usage is discouraged for small functions.

All non deterministic functions should take a tf.random.Generator as an argument. This allows for reproducibility and fast random number generation.

The code uses 2-spaces indent.

Whenever possible, the strategy pattern should be preferred over boolean arguments to reduce the number of arguments.

### TODO list

* Perlin noise for image priors.
* Negative data augmentation pipeline.
* Empirical Robustness against adversarial attacks.
* Test on Fashion-MNIST, Cifar-10, Omniglot, and other tabular data.
* Re-factor `ocml/train.py` and `ocml/priors.py` to allow arbitrary parametrization of `Q_t` using negative data augmentation, prior distribution, and adversarial training.

![Mnist GAN like images](figures/mnist_grid.PNG)

## Legacy notebooks

Old notebooks are still soted under `draft_notebooks.py` for completeness. They were used for early experiments and workshop paper. They should not be used for future experiments.

### Toy examples

The entry point is the notebook `OneClassDraft.ipynb` with 2D toy examples. It contains useful comments and documentation, before moving on to other notebooks. The algorithm is competitive against conventional networks and older ML methods.

### Mnist examples

Larger scale experiments can be found in notebook `OneClassMnist.ipynb`. It outperforms Deep-SVDD on Mnist. It also contains the code for image generation in a GAN-like procedure.


