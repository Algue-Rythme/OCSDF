"""Pre-defined architectures of Lipschitz layers."""

import tensorflow as tf
import tensorflow as tf
from tensorflow.keras.layers import Flatten
from deel.lip.initializers import SpectralInitializer
from deel.lip.layers import SpectralDense, SpectralConv2D, LipschitzLayer, Condensable, InvertibleDownSampling
from deel.lip.layers import ScaledGlobalAveragePooling2D, FrobeniusDense, ScaledAveragePooling2D
from deel.lip.activations import MaxMin, GroupSort2, FullSort
from deel.lip.model import Sequential as DeelSequential
from deel.lip.activations import PReLUlip, FullSort
from lipschitz_normalizers import NormalizedDense, SpectralInfConv2D
from tensorflow.keras.layers import InputLayer, AveragePooling2D


def spectral_VGG(input_shape,
                conv_widths,
                dense_widths,
                k_coef_lip=1.,
                *,
                groupsort=True,
                strides=False,
                two_infinity_start=False,
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
      layers.append(SpectralInfConv2D(conv_widths[0], window_size, normalizer='2-inf', strides=strides, projection=pgd))
    else:
      layers.append(SpectralConv2D(width, window_size, strides=strides))
    layers.append(activation())
    if not strides:
      layers.append(ScaledAveragePooling2D((2, 2)))

  if pooling:
    layers.append(ScaledGlobalAveragePooling2D())
  else:
    layers.append(Flatten())

  for width in dense_widths:
    layers.append(SpectralDense(width))
    layers.append(activation())

  layers.append(NormalizedDense(1, normalizer='2-inf', projection=pgd))
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
      layers.append(SpectralInfConv2D(width, window_size, normalizer=normalizer, strides=strides, projection=pgd))
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
    layers.append(SpectralDense(widths[0]))
    activation = GroupSort2 if groupsort else FullSort
    layers.append(activation())
    for width in widths[1:]:
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
