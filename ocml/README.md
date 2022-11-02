## Source files

- `train.py`: adversarial generation and main training loop.
- `priors.py`: prior of complementary distributions and data augmentation.
- `evaluate.py`: tools to evaluate Local Lipschitz Constant (LLC) of the networks, monitore weights, and calibrate the predictions.
- `plot.py`: plotting utilities (image, 2D contour plots, images).
- `models.py`: definition of common Lipschitz and conventional architectures.
- `layers.py`: additional layers for Lipschitz networks, more compliant with the theorem of Anil et al. (2018).
- `datasets.py`: pre-processing of common datasets in an unified framework.

## Contribution 

### Conventions

Newly added functions should contain a docstring that specifies arguments, attributes and return values.

When performance is an issue a tf.function decorator should be added to the outermost function. tf.function usage is discouraged for small functions.

All non deterministic functions should take a tf.random.Generator as an argument. This allows for reproducibility and fast random number generation.

The code uses 2-spaces indent.

Whenever possible, the strategy pattern should be preferred over boolean arguments to reduce the number of arguments.

### TODO list

* Empirical Robustness against adversarial attacks.
* Test on Cifar-10, Omniglot.
