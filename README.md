## One Class Metric Learning

Learn to approximate the Sign Distance Function (SDF) to the boundary of a distribution, i.e a dataset.

Use a combination of Lipschitz networks, adversarial training and Hinge Kabntorovich Rubinstein loss (HKR).

![2D Toy example](figures/all_methods_grid.PNG)

## Structure of the repository

The repository is organized as follow:
  * `main.py`: load datasets and configurations, create model, train it, and log the results.
  * `train.py`: adversarial generation and main training loop.
  * `priors.py`: prior of complementary distributions and data augmentation.
  * `evaluate.py`: tools to evaluate Local Lipschitz Constant (LLC) of the networks, monitore weights, and calibrate the predictions.
  * `plot.py`: plotting utilities (image, 2D contour plots, images).
  * `models.py`: definition of common Lipschitz and conventional architectures.
  * `datasets.py`: pre-processing of common datasets in an unified framework.
  * `configs/`: configurations for different training sets with hyper-parameters that allow reproducibility.
  * `draft_notebooks/`: old notebooks for early experiments and POC.

### Remarks

`Wandb` is used experiment tracking, `plotly` and `seaborn` are used for plotting. Latest version of `deel-lip` is recommanded.

### Run the code

The following directories will be populated:

  * `images/`: record images produced for uploading to `wandb`.
  * `weights/`: contain weights of the network architecture in `.h5` format.
  * `wandb/`: if wandb is used - to store local variables.

## TODO list

* Perlin noise for image priors.
* Negative data augmentation pipeline.
* 

## Old notebooks

Old notebooks are still soted under `draft_notebooks.py` for completeness. They were used for early experiments and workshop paper. They should not be used for future experiments.

### Toy examples

The entry point is the notebook `OneClassDraft.ipynb` with 2D toy examples. It contains useful comments and documentation, before moving on to other notebooks. The algorithm is competitive against conventional networks and older ML methods.

### Mnist examples

Larger scale experiments can be found in notebook `OneClassMnist.ipynb`. It outperforms Deep-SVDD on Mnist. It also contains the code for image generation in a GAN-like procedure.

![Mnist GAN like images](figures/mnist_grid.PNG)
