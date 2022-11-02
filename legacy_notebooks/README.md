## Notebooks

These old notebooks are still stored under `legacy_notebooks/` for completeness. They were used for early experiments and workshop paper. They should not be used for future experiments.

### Baselines

Baselines from Scikit-Learn to compare against and reproduce figures from the paper is in `run_toy2d_baselines`.

### Toy examples

The entry point is the notebook `OneClassDraft.ipynb` with 2D toy examples. It contains useful comments and documentation, before moving on to other notebooks. The algorithm is competitive against conventional networks and older ML methods.

### Mnist examples

Larger scale experiments can be found in notebook `OneClassMnist.ipynb`. It outperforms Deep-SVDD on Mnist. It also contains the code for image generation in a GAN-like procedure.