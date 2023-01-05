"""Pre-defined architectures of Lipschitz layers."""
from functools import partial
import tensorflow as tf
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Sequential
from deel.lip.initializers import SpectralInitializer
from deel.lip.layers import SpectralDense, SpectralConv2D, LipschitzLayer, Condensable, InvertibleDownSampling
from deel.lip.layers import ScaledGlobalAveragePooling2D, FrobeniusDense, ScaledAveragePooling2D
from deel.lip.activations import MaxMin, GroupSort2, FullSort
from deel.lip.model import Sequential as DeelSequential
from deel.lip.activations import PReLUlip, FullSort
from ocml.layers import NormalizedDense, NormalizedConv2D
from tensorflow.keras.layers import InputLayer, AveragePooling2D

SpectralDense = partial(SpectralDense, kernel_initializer="orthogonal")
SpectralConv2D = partial(SpectralConv2D, kernel_initializer="orthogonal")


def load_VGG_V2_from_run(entity, project, run_id, model, patch=True):
  import wandb
  api = wandb.Api()
  try:
    run = api.run(f"{entity}/{project}/{run_id}")
    prefix = f'downloaded/{project}/run_{run.name}/'
    run.file("weights/model_weights.h5").download(prefix, replace=True)

    if patch:
      model.load_weights(f"{prefix}weights/model_weights.h5", by_name=False, skip_mismatch=False)
      model.layers[-1] = NormalizedDense(1, normalizer='2-inf', V2=True)
  except OSError as e:
    print(f"Failed to load {entity}/{project}/{run_id}")
    raise e


def froze_everything_except_last(model):
  for layer in model.layers[:-1]:
    layer.trainable = False
  model.layers[-1].trainable = True


def spectral_VGG(input_shape,
                conv_widths,
                dense_widths,
                k_coef_lip=1.,
                *,
                groupsort=True,
                strides=False,
                two_infinity_start=False,
                global_pooling=True,
                pooling=True,
                pgd=False
                ):
  # multivariate universal approximation holds
  layers = [InputLayer(input_shape)]
  activation = GroupSort2 if groupsort else FullSort
  window_size = (5, 5)
  strides = (2, 2) if strides else (1, 1)

  for i, width in enumerate(conv_widths):
    if i == 0 and two_infinity_start:
      layers.append(NormalizedConv2D(conv_widths[0], window_size, normalizer='2-inf', strides=strides, projection=pgd))
    else:
      layers.append(SpectralConv2D(width, window_size, strides=strides))
    layers.append(activation())
    if pooling:
      layers.append(ScaledAveragePooling2D((2, 2)))

  if global_pooling:
    layers.append(ScaledGlobalAveragePooling2D())
  else:
    layers.append(Flatten())

  for width in dense_widths:
    layers.append(SpectralDense(width))
    layers.append(activation())

  layers.append(NormalizedDense(1, normalizer='2-inf', projection=pgd))
  model = DeelSequential(layers, k_coef_lip=k_coef_lip)
  return model


def spectral_VGG_V2(input_shape, k_coef_lip=1., scale=1, legacy=False):
  layers = [InputLayer(input_shape)]
  activation = GroupSort2
  window_size = (3, 3)
  layers = [
    SpectralConv2D(64*scale, window_size),
    activation(),
    SpectralConv2D(64*scale, window_size),
    activation(),
    SpectralConv2D(64*scale, window_size),
    activation(),
    SpectralConv2D(64*scale, window_size),
    ScaledAveragePooling2D((2, 2)),
    SpectralConv2D(128*scale, window_size),
    activation(),
    SpectralConv2D(128*scale, window_size),
    activation(),
    SpectralConv2D(128*scale, window_size),
    activation(),
    SpectralConv2D(128*scale, window_size),
    activation(),
    ScaledAveragePooling2D((2, 2)),
    SpectralConv2D(256*scale, window_size),
    activation(),
    SpectralConv2D(256*scale, window_size),
    activation(),
    SpectralConv2D(256*scale, window_size),
    activation(),
    ScaledGlobalAveragePooling2D(),
    NormalizedDense(1, normalizer='2-inf', V2=not legacy)
    # FrobeniusDense(1)
  ]
  model = DeelSequential(layers, k_coef_lip=k_coef_lip)
  return model


def spectral_VGG_V3(input_shape, k_coef_lip=1.):
  layers = [InputLayer(input_shape)]
  activation = GroupSort2
  window_size = (3, 3)
  layers = [
    SpectralConv2D(64, window_size),
    activation(),
    SpectralConv2D(64, window_size),
    activation(),
    SpectralConv2D(64, window_size),
    activation(),
    SpectralConv2D(64, window_size),
    SpectralConv2D(128, window_size, strides=(2, 2)),
    activation(),
    SpectralConv2D(128, window_size),
    activation(),
    SpectralConv2D(128, window_size),
    activation(),
    SpectralConv2D(128, window_size),
    activation(),
    SpectralConv2D(256, window_size, strides=(2, 2)),
    activation(),
    SpectralConv2D(256, window_size),
    activation(),
    SpectralConv2D(256, window_size),
    activation(),
    ScaledGlobalAveragePooling2D(),
    NormalizedDense(1, normalizer='2-inf')
  ]
  model = DeelSequential(layers, k_coef_lip=k_coef_lip)
  return model


def normalized_VGG(input_shape,
                conv_widths,
                dense_widths,
                k_coef_lip=1.,
                *,
                groupsort=True,
                strides=False,
                pgd=False
                ):
    # multivariate universal approximation holds
    layers = [InputLayer(input_shape)]
    activation = GroupSort2 if groupsort else FullSort
    window_size = (5, 5)
    strides = (2, 2) if strides else (1, 1)
    for i, width in enumerate(conv_widths):
      normalizer = '2-inf' if i == 0 else 'inf'
      layers.append(NormalizedConv2D(width, window_size, normalizer=normalizer, strides=strides, projection=pgd))
      layers.append(activation())
      if not strides:
        layers.append(AveragePooling2D((2, 2), padding='valid'))
    layers.append(Flatten())
    for width in dense_widths:
      layers.append(NormalizedDense(width, normalizer='inf', projection=pgd))
      layers.append(activation())
    layers.append(NormalizedDense(1, normalizer='inf', projection=pgd))
    model = DeelSequential(layers, k_coef_lip=k_coef_lip)
    return model


def spectral_dense(widths,
                  input_shape,
                  k_coef_lip=1.,
                  groupsort=False):
    """Create Lipschitz network.
    
    Args:
        widths: sequence of widths for the network.
        input_shape: tuple corresponding to the shape of the input.
        k_coef_lip: correcting factor (default: 1.).
        groupsort: whether to use GroupSort (if True) or FullSort( if False). Default to False.
        
    Returns:
        A Lipschitz network.
    """
    # multivariate universal approximation holds.
    layers = [InputLayer(input_shape)]
    activation = GroupSort2 if groupsort else FullSort
    for width in widths:
        layers.append(SpectralDense(width))
        layers.append(activation())
    # linear decision without activation
    layers.append(FrobeniusDense(1))
    model = DeelSequential(layers, k_coef_lip=k_coef_lip)
    return model


def normalized_dense(widths,
                    input_shape,
                    k_coef_lip=1.,
                    groupsort=True,
                    projection=False):
    """Create Lipschitz network.
    
    Args:
        widths: sequence of widths for the network.
        input_shape: tuple corresponding to the shape of the input.
        k_coef_lip: correcting factor (default: 1.).
        groupsort: whether to use GroupSort (if True) or FullSort( if False). Default to False.
        projection: whether to perform condense() after each step or not (default: False). Only available when spectral_dense=True.
        
    Returns:
        A Lipschitz network.
    """
    # multivariate universal approximation holds.
    layers = [InputLayer(input_shape)]
    layers.append(NormalizedDense(widths[0], normalizer='2-inf', projection=projection))
    activation = GroupSort2 if groupsort else FullSort
    layers.append(activation())
    for width in widths[1:]:
        layers.append(NormalizedDense(width, normalizer='inf', projection=projection))
        layers.append(activation())
    # linear decision without activation
    layers.append(NormalizedDense(1, normalizer='inf', projection=projection))
    model = DeelSequential(layers, k_coef_lip=k_coef_lip)
    return model


def conventional_dense(widths, input_shape):
    """Create conventional network.
    
    Args:
        widths: sequence of widths for the network.
        input_shape: tuple corresponding to the shape of the input.
        
    Returns:
        A conventional network.
    """
    # multivariate universal approximation holds
    layers = [InputLayer(input_shape)]
    activation =  tf.keras.layers.ReLU
    for width in widths:
        layers.append(Dense(width))
        layers.append(activation())
    layers.append(Dense(1))
    model = Sequential(layers)
    return model
