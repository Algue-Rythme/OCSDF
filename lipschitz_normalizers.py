from tensorflow.keras.layers import Input, Lambda, Flatten, AveragePooling2D, BatchNormalization, Conv2D, MaxPool2D, Dense, Activation
from tensorflow.keras.models import Sequential
from tensorflow.data import Dataset
from tensorflow.keras.activations import relu
import tensorflow as tf
from tensorflow.keras import backend as K
from deel.lip.layers import Condensable, LipschitzLayer
from tensorflow.keras.initializers import Orthogonal
import math


#@tf.function(experimental_relax_shapes=True)
def infinity_norm_normalization(kernel, inf_norm_bounds=tf.constant(1.)):
    # assuming: tf.matmul(x, wbar)
    kernel_shape = kernel.shape
    weights = tf.reshape(kernel, [-1, kernel_shape[-1]])
    eps = 1e-7
    # well known fact in litterature on infinite norm:
    # maximum absolute column sum.
    column_wise_norms = tf.reduce_sum(tf.math.abs(weights), axis=0, keepdims=True) + eps
    mask = tf.where(column_wise_norms > inf_norm_bounds, inf_norm_bounds / column_wise_norms, 1.)
    weights = weights * mask
    wbar = tf.reshape(weights, kernel_shape)
    return wbar

#@tf.function(experimental_relax_shapes=True)
def two_to_infinity_norm_normalization(kernel, inf_norm_bounds=tf.constant(1.)):
    # https://arxiv.org/pdf/1705.10735.pdf
    # Page 20, proposition 6.1 for formal justification
    # assuming: tf.matmul(x, wbar)
    kernel_shape = kernel.shape
    weights = tf.reshape(kernel, [-1, kernel_shape[-1]])
    eps = 1e-7
    column_wise_norms = tf.reduce_sum(weights**2, axis=0, keepdims=True) ** 0.5 + eps
    mask = tf.where(column_wise_norms > inf_norm_bounds, inf_norm_bounds / column_wise_norms, 1.)
    weights = weights * mask
    wbar = tf.reshape(weights, kernel_shape)
    return wbar


class NormalizedDense(Dense, LipschitzLayer, Condensable):
    def __init__(
        self,
        units,
        normalizer,
        activation=None,
        use_bias=True,
        kernel_initializer=Orthogonal(gain=1.0),
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        k_coef_lip=1.0,
        inf_norm_bounds=1.0,
        projection=False,
        **kwargs
    ):
        super().__init__(
            units=units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )
        self._kwargs = kwargs
        self.set_klip_factor(k_coef_lip)
        self.inf_norm_bounds = tf.constant(inf_norm_bounds)
        self.normalizer = normalizer
        self.built = False
        self.projection = projection
        if normalizer == 'inf':
            self.normalizer_fun = infinity_norm_normalization
        elif normalizer == '2-inf':
            self.normalizer_fun = two_to_infinity_norm_normalization
            
    def build(self, input_shape):
        super(NormalizedDense, self).build(input_shape)
        self._init_lip_coef(input_shape)
        self.built = True

    def _compute_lip_coef(self, input_shape=None):
        return 1.0  # this layer doesn't require a corrective factor

    @tf.function
    def call(self, x, training=True):
        wbar = self.kernel
        if training:
            wbar = self.normalizer_fun(wbar, self.inf_norm_bounds)
            if self.projection:
                self.kernel.assign(wbar)
        outputs = tf.matmul(x, wbar) * self._get_coef()
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs
    
    def condense(self):
        wbar = self.normalizer_fun(self.kernel, self.inf_norm_bounds)
        self.kernel.assign(wbar)
        
    def get_config(self):
        config = {
            "k_coef_lip": self.k_coef_lip,
            "inf_norm_bounds": self.inf_norm_bounds,
            "normalizer": self.normalizer
        }
        base_config = super(SpectralDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def vanilla_export(self):
        self._kwargs["name"] = self.name
        layer = Dense(
            units=self.units,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer=Orthogonal(gain=1.0),
            bias_initializer="zeros",
            **self._kwargs
        )
        layer.build(self.input_shape)
        layer.kernel.assign(self.wbar)
        if self.use_bias:
            layer.bias.assign(self.bias)
        return layer


class SpectralInfConv2D(Conv2D, LipschitzLayer, Condensable):
    def __init__(
        self,
        filters,
        kernel_size,
        normalizer,
        strides=(1, 1),
        padding="same",
        data_format=None,
        dilation_rate=(1, 1),
        activation=None,
        use_bias=True,
        kernel_initializer=Orthogonal(gain=1.0),
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        k_coef_lip=1.0,
        inf_norm_bounds=1.0,
        projection=False,
        **kwargs
    ):
        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )
        self._kwargs = kwargs
        self.set_klip_factor(k_coef_lip)
        self.inf_norm_bounds = tf.constant(inf_norm_bounds)
        self.built = False
        self.projection = projection
        if normalizer == 'inf':
            self.normalizer_fun = infinity_norm_normalization
        elif normalizer == '2-inf':
            self.normalizer_fun = two_to_infinity_norm_normalization
            
    def build(self, input_shape):
        super(SpectralInfConv2D, self).build(input_shape)
        self._init_lip_coef(input_shape)
        self.built = True

    def _compute_lip_coef(self, input_shape=None):
        return 1.0  # this layer don't require a corrective factor

    @tf.function
    def call(self, x, training=True):
        wbar = self.kernel
        if training:
            wbar = self.normalizer_fun(wbar, self.inf_norm_bounds)
            if self.projection:
                self.kernel.assign(wbar)
        outputs = K.conv2d(
            x,
            wbar,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )
        if self.use_bias:
            outputs = K.bias_add(outputs, self.bias, data_format=self.data_format)
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs
    
    def condense(self):
        wbar = self.normalizer_fun(self.kernel, self.inf_norm_bounds)
        self.kernel.assign(wbar)
        
    def get_config(self):
        config = {
            "k_coef_lip": self.k_coef_lip,
            "inf_norm_bounds": self.inf_norm_bounds,
            "normalizer": self.normalizer
        }
        base_config = super(SpectralDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def vanilla_export(self):
        self._kwargs["name"] = self.name
        layer = Dense(
            units=self.units,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer=Orthogonal(gain=1.0),
            bias_initializer="zeros",
            **self._kwargs
        )
        layer.build(self.input_shape)
        layer.kernel.assign(self.kernel)
        if self.use_bias:
            layer.bias.assign(self.bias)
        return layer
