# -*- coding: utf-8 -*-

# Cell
import tensorflow as tf, tensorflow_datasets as tfds, numpy as np
import math
from datetime import datetime
import time, pickle

# Cell
class StandardDeviationInit(tf.keras.initializers.Initializer):

    def __init__(self, minval=0, maxval=1):
        self.minval = minval
        self.maxval = maxval
    
    def __call__(self, shape, dtype=None):
        return tf.random.uniform(shape, self.minval, self.maxval, dtype=dtype)

# Cell
class PointConv2D(tf.keras.layers.Conv2D):
    
    def __init__(self, filters, kernel_size, strides, padding, activation=None, dtype=tf.float32, **kwargs):
        super(PointConv2D, self).__init__(filters, kernel_size, strides, activation=activation, 
                                          dtype=dtype, padding=padding, use_bias=True, **kwargs)
    
    def build(self, input_shape):
        super(PointConv2D, self).build(input_shape)
    
    def call(self, inputs):
        return super(PointConv2D, self).call(inputs)

class PointDense(tf.keras.layers.Dense):

    def __init__(self, units, activation=None, **kwargs):
        super(PointDense, self).__init__(units, activation=activation, use_bias=True, **kwargs)
    
    def build(self, input_shape):
        super(PointDense, self).build(input_shape)
    
    def call(self, inputs):
        return super(PointDense, self).call(inputs)

# Cell
minrho, maxrho = -9, -8
class PerturbedConv2D(tf.keras.layers.Conv2D):
    
    def __init__(self, filters, kernel_size, strides, padding, activation=None, dtype=tf.float32, **kwargs):
        super(PerturbedConv2D, self).__init__(filters, kernel_size, strides, activation=activation, 
                                          dtype=dtype, padding=padding, use_bias=True, **kwargs)
        self._name = 'PerturbedConv2D'
    
    def build(self, input_shape):
        super(PerturbedConv2D, self).build(input_shape)
        self.kernel_mu = self.add_weight(name='kernel_mu', shape=self.kernel.shape,
                                         initializer=self.kernel_initializer,
                                         trainable=True,
                                         dtype=self.dtype)
        self.kernel_rho = self.add_weight(name='kernel_rho', shape=self.kernel.shape,
                                          initializer=StandardDeviationInit(minrho, maxrho),
                                          trainable=True,
                                          dtype=self.dtype)
        self.bias_mu = self.add_weight(name='bias_mu', shape=self.bias.shape,
                                         initializer=self.bias_initializer,
                                         trainable=True,
                                         dtype=self.dtype)
        self.bias_rho = self.add_weight(name='bias_rho', shape=self.bias.shape,
                                          initializer=StandardDeviationInit(minrho, maxrho),
                                          trainable=True,
                                          dtype=self.dtype)
        
    
    def _reparametrize(self, avoid_sampling):
        if avoid_sampling:
            return self.kernel_mu, self.bias_mu
        eps_w_shape = self.kernel_mu.shape
        eps_w = tf.random.normal(eps_w_shape, 0, 1, dtype=tf.float32)
        term_w = tf.math.multiply(eps_w, tf.math.exp(self.kernel_rho))
        kernel = tf.math.add(self.kernel_mu, term_w)
        eps_b_shape = self.bias_mu.shape
        eps_b = tf.random.normal(eps_b_shape, 0, 1, dtype=tf.float32)
        term_b = tf.math.multiply(eps_b, tf.math.exp(self.bias_rho))
        bias = tf.math.add(self.bias_mu, term_b)
        return kernel, bias
    
    def call(self, inputs, avoid_sampling=False):
        k, b = self._reparametrize(avoid_sampling)
        self.kernel, self.bias = k + 0, b + 0
        return super(PerturbedConv2D, self).call(inputs)

class PerturbedDense(tf.keras.layers.Dense):

    def __init__(self, units, activation=None, **kwargs):
        super(PerturbedDense, self).__init__(units, activation=activation, use_bias=True, **kwargs)
        self._name = 'PerturbedDense'
    
    def build(self, input_shape):
        super(PerturbedDense, self).build(input_shape)
        self.kernel_mu = self.add_weight(name='kernel_mu', shape=self.kernel.shape,
                                         initializer=self.kernel_initializer,
                                         trainable=True,
                                         dtype=self.dtype)
        self.kernel_rho = self.add_weight(name='kernel_rho', shape=self.kernel.shape,
                                          initializer=StandardDeviationInit(minrho, maxrho),
                                          trainable=True,
                                          dtype=self.dtype)
        self.bias_mu = self.add_weight(name='bias_mu', shape=self.bias.shape,
                                         initializer=self.bias_initializer,
                                         trainable=True,
                                         dtype=self.dtype)
        self.bias_rho = self.add_weight(name='bias_rho', shape=self.bias.shape,
                                          initializer=StandardDeviationInit(minrho, maxrho),
                                          trainable=True,
                                          dtype=self.dtype)
    
    def _reparametrize(self, avoid_sampling):
        if avoid_sampling:
            return self.kernel_mu, self.bias_mu
        eps_w_shape = self.kernel_mu.shape
        eps_w = tf.random.normal(eps_w_shape, 0, 1, dtype=tf.float32)
        term_w = tf.math.multiply(eps_w, tf.math.exp(self.kernel_rho))
        kernel = tf.math.add(self.kernel_mu, term_w)
        eps_b_shape = self.bias_mu.shape
        eps_b = tf.random.normal(eps_b_shape, 0, 1, dtype=tf.float32)
        term_b = tf.math.multiply(eps_b, tf.math.exp(self.bias_rho))
        bias = tf.math.add(self.bias_mu, term_b)
        return kernel, bias
    
    def call(self, inputs, avoid_sampling=False):
        k, b = self._reparametrize(avoid_sampling)
        self.kernel, self.bias = k + 0, b + 0
        return super(PerturbedDense, self).call(inputs)

# Cell
class PointResNetLayer(tf.keras.layers.Layer):

    def __init__(self, num_filters=16, kernel_size=3, strides=1,
                 activation='relu', batch_normalization=True, conv_first=True):
        self._name = 'PointResNetLayer'
        self.Layers = []
        self.TransferLayers = []
        self.TrainableLayers = []
        self.conv_first, self.batch_normalization, self.activation = conv_first, batch_normalization, activation
        if conv_first:
            layer = PointConv2D(num_filters, kernel_size=kernel_size, strides=strides, padding='same')
            self.Layers.append(layer)
            self.TransferLayers.append(layer)
            self.TrainableLayers.append(layer)
            if batch_normalization:
                layer = tf.keras.layers.BatchNormalization()
                self.Layers.append(layer)
                self.TransferLayers.append(layer)
            if activation is not None:
                layer = tf.keras.layers.Activation(activation)
                self.Layers.append(layer)
        else:
            if batch_normalization:
                layer = tf.keras.layers.BatchNormalization()
                self.Layers.append(layer)
                self.TransferLayers.append(layer)
            if activation is not None:
                layer = tf.keras.layers.Activation(activation)
                self.Layers.append(layer)
            layer = PointConv2D(num_filters, kernel_size=kernel_size, strides=strides, padding='same')
            self.Layers.append(layer)
            self.TransferLayers.append(layer)
            self.TrainableLayers.append(layer)
        self._output_shape = None
    
    def build(self, input_shape):
        for layer in self.Layers:
            layer.build(input_shape)
            input_shape = layer.compute_output_shape(input_shape)
        self._output_shape = input_shape

    def compute_output_shape(self, input_shape):
        return self._output_shape
    
    def call(self, inputs):
        x = inputs
        itr = iter(self.Layers)
        conv_first, batch_normalization, activation = self.conv_first, self.batch_normalization, self.activation
        if conv_first:
            layer = next(itr)
            x = layer.call(x)
            if batch_normalization:
                layer = next(itr)
                x = layer.call(x)
            if activation is not None:
                layer = next(itr)
                x = layer.call(x)
        else:
            if batch_normalization:
                layer = next(itr)
                x = layer.call(x)
            if activation is not None:
               layer = next(itr)
               x = layer.call(x)
            layer = next(itr)
            x = layer.call(x)
        return x

# Cell
class ResNetLayer(tf.keras.layers.Layer):

    def __init__(self, num_filters=16, kernel_size=3, strides=1,
                 activation='relu', batch_normalization=True, conv_first=True):
        self._name = 'ResNetLayer'
        self.Layers = []
        self.TransferLayers = []
        self.TrainableLayers = []
        self.conv_first, self.batch_normalization, self.activation = conv_first, batch_normalization, activation
        if conv_first:
            layer = PerturbedConv2D(num_filters, kernel_size=kernel_size, strides=strides, padding='same')
            self.Layers.append(layer)
            self.TransferLayers.append(layer)
            self.TrainableLayers.append(layer)
            if batch_normalization:
                layer = tf.keras.layers.BatchNormalization()
                self.Layers.append(layer)
                self.TransferLayers.append(layer)
            if activation is not None:
                layer = tf.keras.layers.Activation(activation)
                self.Layers.append(layer)
        else:
            if batch_normalization:
                layer = tf.keras.layers.BatchNormalization()
                self.Layers.append(layer)
                self.TransferLayers.append(layer)
            if activation is not None:
                layer = tf.keras.layers.Activation(activation)
                self.Layers.append(layer)
            layer = PerturbedConv2D(num_filters, kernel_size=kernel_size, strides=strides, padding='same')
            self.Layers.append(layer)
            self.TransferLayers.append(layer)
            self.TrainableLayers.append(layer)
        self._output_shape = None
    
    def build(self, input_shape):
        for layer in self.Layers:
            layer.build(input_shape)
            input_shape = layer.compute_output_shape(input_shape)
        self._output_shape = input_shape

    def compute_output_shape(self, input_shape):
        return self._output_shape
    
    def call(self, inputs):
        x = inputs
        itr = iter(self.Layers)
        conv_first, batch_normalization, activation = self.conv_first, self.batch_normalization, self.activation
        if conv_first:
            layer = next(itr)
            x = layer.call(x)
            if batch_normalization:
                layer = next(itr)
                x = layer.call(x)
            if activation is not None:
                layer = next(itr)
                x = layer.call(x)
        else:
            if batch_normalization:
                layer = next(itr)
                x = layer.call(x)
            if activation is not None:
               layer = next(itr)
               x = layer.call(x)
            layer = next(itr)
            x = layer.call(x)
        return x

# Cell
class PointCNN(tf.keras.Model):

    def __init__(self, num_res_blocks, num_filters, num_classes, optimizer=None):
        super(PointCNN, self).__init__()
        self.Layers = dict()
        self.TransferLayers = []
        self.TrainableLayers = []
        layer = PointResNetLayer()
        self.Layers[-1] = []
        self.Layers[-1].append(layer)
        self.TransferLayers.extend(layer.TransferLayers)
        self.TrainableLayers.extend(layer.TrainableLayers)
        self.num_stack, self.num_res_blocks = 3, num_res_blocks
        # Instantiate the stack of residual units
        for stack in range(self.num_stack):
            self.Layers[stack] = []
            for res_block in range(num_res_blocks):
                strides = 1
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    strides = 2  # downsample
                layer = PointResNetLayer(num_filters=num_filters, strides=strides)
                self.Layers[stack].append(layer)
                self.TransferLayers.extend(layer.TransferLayers)
                self.TrainableLayers.extend(layer.TrainableLayers)
                layer = PointResNetLayer(num_filters=num_filters, activation=None)
                self.Layers[stack].append(layer)
                self.TransferLayers.extend(layer.TransferLayers)
                self.TrainableLayers.extend(layer.TrainableLayers)
                if stack > 0 and res_block == 0:  
                    # first layer but not first stack
                    # linear projection residual shortcut connection to match
                    # changed dims
                    layer = PointResNetLayer(num_filters=num_filters, kernel_size=1,
                                        strides=strides, activation=None,
                                        batch_normalization=False)
                    self.Layers[stack].append(layer)
                    self.TransferLayers.extend(layer.TransferLayers)
                    self.TrainableLayers.extend(layer.TrainableLayers)
                self.Layers[stack].append(tf.keras.layers.Activation('relu'))
            num_filters *= 2
        
        self.Layers[self.num_stack] = []
        self.Layers[self.num_stack].append(tf.keras.layers.AveragePooling2D(pool_size=8))
        self.Layers[self.num_stack].append(tf.keras.layers.Flatten())
        layer = PointDense(num_classes)
        self.Layers[self.num_stack].append(layer)
        self.TransferLayers.append(layer)
        self.TrainableLayers.append(layer)
        self.LayerWiseOutputs = []
        self.optimizer = optimizer
    
    def build(self, input_shape):
        self.Layers[-1][0].build(input_shape)
        x = self.Layers[-1][0].compute_output_shape(input_shape)
        for stack in range(self.num_stack):
            layer = 0
            for res_block in range(self.num_res_blocks):
                self.Layers[stack][layer].build(x)
                y = self.Layers[stack][layer].compute_output_shape(x)
                layer += 1
                self.Layers[stack][layer].build(y)
                y = self.Layers[stack][layer].compute_output_shape(y)
                layer += 1
                if stack > 0 and res_block == 0:
                    self.Layers[stack][layer].build(x)
                    x = self.Layers[stack][layer].compute_output_shape(x)
                    layer += 1
                self.Layers[stack][layer].build(x)
                x = self.Layers[stack][layer].compute_output_shape(x)
                layer += 1

        stack = self.num_stack
        self.Layers[stack][0].build(x)
        x = self.Layers[stack][0].compute_output_shape(x)
        self.Layers[stack][1].build(x)
        y = self.Layers[stack][1].compute_output_shape(x)
        self.Layers[stack][2].build(y)
    
    def call(self, inputs, is_training=False):
        x = self.Layers[-1][0].call(inputs)
        for stack in range(self.num_stack):
            layer = 0
            for res_block in range(self.num_res_blocks):
                y = self.Layers[stack][layer].call(x)
                layer += 1
                y = self.Layers[stack][layer].call(y)
                layer += 1
                if stack > 0 and res_block == 0:
                    x = self.Layers[stack][layer].call(x)
                    layer += 1
                x = tf.keras.layers.Add()([x, y])
                x = self.Layers[stack][layer].call(x)
                layer += 1

        stack = self.num_stack
        x = self.Layers[stack][0].call(x)
        y = self.Layers[stack][1].call(x)
        outputs = self.Layers[stack][2].call(y)
        self.LayerWiseOutputs.append(outputs)
        if not is_training:
            outputs = tf.nn.softmax(outputs)
        return outputs

    # def setstate(self, kernel_mu):
    #     i = 0
    #     for layer in self.TransferLayers:
    #         if isinstance(layer, PointConv2D):
    #             layer.kernel.assign(tf.convert_to_tensor(kernel_mu[i], tf.float32))
    #             layer.bias.assign(tf.convert_to_tensor(kernel_mu[i+1], tf.float32))
    #             i += 2
    #     for layer in self.TransferLayers:
    #         if isinstance(layer, tf.keras.layers.BatchNormalization):
    #             layer.gamma.assign(tf.convert_to_tensor(kernel_mu[i], tf.float32))
    #             layer.beta.assign(tf.convert_to_tensor(kernel_mu[i+1], tf.float32))
    #             layer.moving_mean.assign(tf.convert_to_tensor(kernel_mu[i+2], tf.float32))
    #             layer.moving_variance.assign(tf.convert_to_tensor(kernel_mu[i+3], tf.float32))
    #             i += 4
    #     for layer in self.TransferLayers:
    #         if isinstance(layer, PointDense):
    #             layer.kernel.assign(tf.convert_to_tensor(kernel_mu[i], tf.float32))
    #             layer.bias.assign(tf.convert_to_tensor(kernel_mu[i+1], tf.float32))
    #             i += 2

    def setstate(self, kernels):
        itr = iter(self.TransferLayers)
        i = 0
        for layer in self.TransferLayers:
            if isinstance(layer, (PointConv2D, PointDense)):
                layer.kernel.assign(tf.convert_to_tensor(kernels[i], tf.float32))
                layer.bias.assign(tf.convert_to_tensor(kernels[i+1], tf.float32))
                i += 2
            elif isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.gamma.assign(tf.convert_to_tensor(kernels[i], tf.float32))
                layer.beta.assign(tf.convert_to_tensor(kernels[i+1], tf.float32))
                layer.moving_mean.assign(tf.convert_to_tensor(kernels[i+2], tf.float32))
                layer.moving_variance.assign(tf.convert_to_tensor(kernels[i+3], tf.float32))
                i += 4

class PointCNNTScaled(PointCNN):

    def call(self, inputs, T=1, is_training=False):
        x = self.Layers[-1][0].call(inputs)
        for stack in range(self.num_stack):
            layer = 0
            for res_block in range(self.num_res_blocks):
                y = self.Layers[stack][layer].call(x)
                layer += 1
                y = self.Layers[stack][layer].call(y)
                layer += 1
                if stack > 0 and res_block == 0:
                    x = self.Layers[stack][layer].call(x)
                    layer += 1
                x = tf.keras.layers.Add()([x, y])
                x = self.Layers[stack][layer].call(x)
                layer += 1

        stack = self.num_stack
        x = self.Layers[stack][0].call(x)
        y = self.Layers[stack][1].call(x)
        outputs = self.Layers[stack][2].call(y)
        if not is_training:
            outputs = tf.nn.softmax(outputs/T)
        return outputs

# Cell
class PerturbedNN(tf.keras.Model):

    def __init__(self, num_res_blocks, num_filters, num_classes, optimizer=None):
        super(PerturbedNN, self).__init__()
        self.Layers = dict()
        self.TransferLayers = []
        self.TrainableLayers = []
        layer = ResNetLayer()
        self.Layers[-1] = []
        self.Layers[-1].append(layer)
        self.TransferLayers.extend(layer.TransferLayers)
        self.TrainableLayers.extend(layer.TrainableLayers)
        self.num_stack, self.num_res_blocks = 3, num_res_blocks
        # Instantiate the stack of residual units
        for stack in range(self.num_stack):
            self.Layers[stack] = []
            for res_block in range(num_res_blocks):
                strides = 1
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    strides = 2  # downsample
                layer = ResNetLayer(num_filters=num_filters, strides=strides)
                self.Layers[stack].append(layer)
                self.TransferLayers.extend(layer.TransferLayers)
                self.TrainableLayers.extend(layer.TrainableLayers)
                layer = ResNetLayer(num_filters=num_filters, activation=None)
                self.Layers[stack].append(layer)
                self.TransferLayers.extend(layer.TransferLayers)
                self.TrainableLayers.extend(layer.TrainableLayers)
                if stack > 0 and res_block == 0:  
                    # first layer but not first stack
                    # linear projection residual shortcut connection to match
                    # changed dims
                    layer = ResNetLayer(num_filters=num_filters, kernel_size=1,
                                        strides=strides, activation=None,
                                        batch_normalization=False)
                    self.Layers[stack].append(layer)
                    self.TransferLayers.extend(layer.TransferLayers)
                    self.TrainableLayers.extend(layer.TrainableLayers)
                self.Layers[stack].append(tf.keras.layers.Activation('relu'))
            num_filters *= 2
        
        self.Layers[self.num_stack] = []
        self.Layers[self.num_stack].append(tf.keras.layers.AveragePooling2D(pool_size=8))
        self.Layers[self.num_stack].append(tf.keras.layers.Flatten())
        layer = PerturbedDense(num_classes)
        self.Layers[self.num_stack].append(layer)
        self.TransferLayers.append(layer)
        self.TrainableLayers.append(layer)
        self.optimizer = optimizer
    
    def build(self, input_shape):
        self.Layers[-1][0].build(input_shape)
        x = self.Layers[-1][0].compute_output_shape(input_shape)
        for stack in range(self.num_stack):
            layer = 0
            for res_block in range(self.num_res_blocks):
                self.Layers[stack][layer].build(x)
                y = self.Layers[stack][layer].compute_output_shape(x)
                layer += 1
                self.Layers[stack][layer].build(y)
                y = self.Layers[stack][layer].compute_output_shape(y)
                layer += 1
                if stack > 0 and res_block == 0:
                    self.Layers[stack][layer].build(x)
                    x = self.Layers[stack][layer].compute_output_shape(x)
                    layer += 1
                self.Layers[stack][layer].build(x)
                x = self.Layers[stack][layer].compute_output_shape(x)
                layer += 1

        stack = self.num_stack
        self.Layers[stack][0].build(x)
        x = self.Layers[stack][0].compute_output_shape(x)
        self.Layers[stack][1].build(x)
        y = self.Layers[stack][1].compute_output_shape(x)
        self.Layers[stack][2].build(y)
    
    def call(self, inputs, is_training=False):
        x = self.Layers[-1][0].call(inputs)
        for stack in range(self.num_stack):
            layer = 0
            for res_block in range(self.num_res_blocks):
                y = self.Layers[stack][layer].call(x)
                layer += 1
                y = self.Layers[stack][layer].call(y)
                layer += 1
                if stack > 0 and res_block == 0:
                    x = self.Layers[stack][layer].call(x)
                    layer += 1
                x = tf.keras.layers.Add()([x, y])
                x = self.Layers[stack][layer].call(x)
                layer += 1

        stack = self.num_stack
        x = self.Layers[stack][0].call(x)
        y = self.Layers[stack][1].call(x)
        outputs = self.Layers[stack][2].call(y)
        if not is_training:
            outputs = tf.nn.softmax(outputs)
        return outputs
    
    def mou(self, output_tensor):
        un = tf.reduce_mean(tf.reduce_sum(tf.math.reduce_std(output_tensor, 0)**2, 1))
        return un

    def get_loss(self, inputs, targets, samples, **kwargs):
        cse = tf.constant(0., dtype=tf.float32)
        outputs = []
        targets = tf.cast(targets, tf.int64)
        sigma_penalty = 0
        for _ in range(samples):
            output = self.call(inputs, is_training=True)
            outputs.append(tf.nn.softmax(output))
            cse += tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(targets, output))
        
        output_tensor = tf.convert_to_tensor(outputs, dtype=tf.float32)
        entropy = self.mou(output_tensor)
        for layer in self.TrainableLayers:
            sigma_penalty += kwargs.get('c', 0.01)*tf.reduce_sum(-tf.math.log(tf.math.softplus(layer.kernel_rho)))
        return (cse/samples)*kwargs.get('a', 1) + entropy*kwargs.get('b', 1) + sigma_penalty, cse/samples, entropy, sigma_penalty
    
    def compute_gradients(self, inputs, targets, **kwargs):
        _vars = []
        with tf.GradientTape(persistent=True) as tape:
            for layer in self.TrainableLayers:
                _vars.append(layer.kernel_rho)
            tape.watch(_vars)
            F, cse, entropy, sigma_penalty = self.get_loss(inputs, targets, 10, **kwargs)

        dF = tape.gradient(F, _vars)
        
        return dF, F, cse, entropy, sigma_penalty, _vars
    
    def fit(self, inputs, targets, **kwargs):
        start_time = time.time()
        grads, F, cse, entropy, sigma_penalty, vars = self.compute_gradients(inputs, targets, **kwargs)
        self.optimizer.apply_gradients(zip(grads, vars))
        end_time = time.time()
        return F, cse, entropy, sigma_penalty, end_time - start_time

    def getstate(self):
        np_kernel_mu = []
        np_kernel_rho = []
        for layer in self.TransferLayers:
            if isinstance(layer, (PerturbedConv2D, PerturbedDense)):
                np_kernel_mu.append(layer.kernel_mu.numpy())
                np_kernel_mu.append(layer.bias_mu.numpy())
                np_kernel_rho.append(layer.kernel_rho.numpy())
                np_kernel_rho.append(layer.bias_rho.numpy())
            elif isinstance(layer, tf.keras.layers.BatchNormalization):
                np_kernel_mu.append(layer.gamma.numpy())
                np_kernel_mu.append(layer.beta.numpy())
                np_kernel_mu.append(layer.moving_mean.numpy())
                np_kernel_mu.append(layer.moving_variance.numpy())
        return np.asarray(np_kernel_mu + np_kernel_rho)

    def setstate(self, kernels):
        itr = iter(self.TransferLayers)
        i = 0
        for layer in self.TransferLayers:
            if isinstance(layer, (PerturbedConv2D, PerturbedDense)):
                layer.kernel_mu.assign(tf.convert_to_tensor(kernels[i], tf.float32))
                layer.bias_mu.assign(tf.convert_to_tensor(kernels[i+1], tf.float32))
                i += 2
            elif isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.gamma.assign(tf.convert_to_tensor(kernels[i], tf.float32))
                layer.beta.assign(tf.convert_to_tensor(kernels[i+1], tf.float32))
                layer.moving_mean.assign(tf.convert_to_tensor(kernels[i+2], tf.float32))
                layer.moving_variance.assign(tf.convert_to_tensor(kernels[i+3], tf.float32))
                i += 4
        for layer in self.TransferLayers:
            if isinstance(layer, (PerturbedConv2D, PerturbedDense)):
                layer.kernel_rho.assign(tf.convert_to_tensor(kernels[i], tf.float32))
                layer.bias_rho.assign(tf.convert_to_tensor(kernels[i+1], tf.float32))
                i += 2
    
    def transferstate(self, kernel_mu):
        i = 0
        for layer in self.TransferLayers:
            if isinstance(layer, PerturbedConv2D):
                layer.kernel_mu.assign(tf.convert_to_tensor(kernel_mu[i], tf.float32))
                layer.bias_mu.assign(tf.convert_to_tensor(kernel_mu[i+1], tf.float32))
                i += 2
        for layer in self.TransferLayers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.gamma.assign(tf.convert_to_tensor(kernel_mu[i], tf.float32))
                layer.beta.assign(tf.convert_to_tensor(kernel_mu[i+1], tf.float32))
                layer.moving_mean.assign(tf.convert_to_tensor(kernel_mu[i+2], tf.float32))
                layer.moving_variance.assign(tf.convert_to_tensor(kernel_mu[i+3], tf.float32))
                i += 4
        for layer in self.TransferLayers:
            if isinstance(layer, PerturbedDense):
                layer.kernel_mu.assign(tf.convert_to_tensor(kernel_mu[i], tf.float32))
                layer.bias_mu.assign(tf.convert_to_tensor(kernel_mu[i+1], tf.float32))
                i += 2

# Cell
mean, std = 0, 255.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
y_train, y_test = np.squeeze(y_train), np.squeeze(y_test)
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
x_train, x_test = (x_train - mean) / std, (x_test - mean) / std
x_train_mean = np.mean(x_train, axis=0)
x_train -= x_train_mean
x_test -= x_train_mean

# Cell
(x_train_f, y_train_f), (x_test_f, y_test_f) = tf.keras.datasets.cifar100.load_data()
x_train_f, x_test_f = np.array(x_train_f, np.float32), np.array(x_test_f, np.float32)
mean_tr, std_tr, mean_te, std_te = 9., 255., 0., 255.
x_train_f, x_test_f = (x_train_f - mean_tr) / std_tr, (x_test_f - mean_te) / std_te
x_train_mean = np.mean(x_train_f, axis=0)
x_train_f -= x_train_mean
x_test_f -= x_train_mean

# Cell
x_test_c, y_test_c = tfds.as_numpy(tfds.load(name="cmaterdb", split='test', batch_size=-1, as_supervised=True))
x_test_c = np.array(x_test_c, np.float32)
mean_te, std_te = 0., 255.
x_test_c = (x_test_c - mean_te) / std_te
x_train_mean = np.mean(x_test_c, axis=0)
x_test_c -= x_train_mean

# Cell
prefix = ""

def load_config(file_path, model):
    train_config = None
    infile = open(file_path, 'rb')
    train_config = pickle.load(infile)
    model.setstate(np.load(file_path + ".npy", allow_pickle=True))
    return train_config

def load_config_new(file_path, model):
    model.setstate(np.load(file_path, allow_pickle=True))

class EvalVGG10:

    def __init__(self, x_in, x_out, weight_file=None, softmax_file=None, **kwargs):
        self.total_in, self.total_out = x_in.shape[0], x_out.shape[0]
        if softmax_file is None:
            self.model = PointCNN(3, 16, 10)
            self.model.build((None,) + x_in.shape[1:])
            load_config_new(weight_file, self.model)
            self.softmax_in = self._infer(x_in)
            self.softmax_out = self._infer(x_out)
            curr_date_time = (str(datetime.now()).replace(' ', '_')
                                                .replace(':', '_')
                                                .replace('-', '_')
                                                .replace('.', '_'))
            softmax_scores = np.asarray([self.softmax_in, self.softmax_out])
            np.save(prefix + "results/mnist/" + curr_date_time, softmax_scores)
        else:
            softmax_scores = np.load(entropy_file, allow_pickle=True)
            self.softmax_in, self.softmax_out = softmax_scores[0], softmax_scores[1]
        self.inc = 0.001
        self.start = 0.1
        self.end = 1
    
    def _infer(self, x):
        test_data = tf.data.Dataset.from_tensor_slices(x)
        batch_size = 128
        test_data = test_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)
        output = []
        for _, batch_x in enumerate(test_data.take(x.shape[0]//batch_size), 1):
            batch_output = list(self.model.call(batch_x).numpy())
            output.extend(batch_output)
        output_tensor = tf.convert_to_tensor(output)
        softmax_scores = tf.reduce_max(output_tensor, 1)
        return softmax_scores

    def fpr_at_95_tpr(self):
        # Positive = In Distribution (ID)
        # Negative = Out of Distribution (OOD)
        delta = self.start
        total_fpr = 0
        total_thresholds = 0
        while delta <= self.end:
            tp = tf.reduce_sum(tf.cast(tf.math.greater_equal(self.softmax_in, tf.constant(delta, dtype=tf.float32)), tf.int32))
            fn = self.total_in - tp
            fp = tf.reduce_sum(tf.cast(tf.math.greater_equal(self.softmax_out, tf.constant(delta, dtype=tf.float32)), tf.int32))
            tn = self.total_out - fp
            tpr = tp/(tp + fn)
            fpr = fp/(fp + tn)
            if tpr >= 0.94:
                print(delta, tpr, fpr)
            if tpr <= 0.9549 and tpr >= 0.9450:
                total_fpr += fpr
                total_thresholds += 1
            delta += self.inc
        return (total_fpr/total_thresholds).numpy()*100
    
    def auroc(self):
        # Positive = In Distribution (ID)
        # Negative = Out of Distribution (OOD)
        delta = self.start
        auroc, fpr_prev = 0, 1
        while delta <= self.end:
            tp = tf.reduce_sum(tf.cast(tf.math.greater_equal(self.softmax_in, tf.constant(delta, dtype=tf.float32)), tf.int32))
            fn = self.total_in - tp
            fp = tf.reduce_sum(tf.cast(tf.math.greater_equal(self.softmax_out, tf.constant(delta, dtype=tf.float32)), tf.int32))
            tn = self.total_out - fp
            tpr = tp/(tp + fn)
            fpr = fp/(fp + tn)
            auroc += (fpr_prev - fpr)*tpr
            fpr_prev = fpr
            delta += self.inc
        auroc += fpr*tpr;
        return auroc.numpy()*100
    
    def aupr_in(self):
        # Positive = In Distribution (ID)
        # Negative = Out of Distribution (OOD)
        aupr = 0
        recall_prev = 1
        recall, precision = None, None
        delta = self.start
        recalls = []
        while delta <= self.end:
            tp = tf.reduce_mean(tf.cast(tf.math.greater_equal(self.softmax_in, tf.constant(delta, dtype=tf.float32)), tf.float64))
            fp = tf.reduce_mean(tf.cast(tf.math.greater_equal(self.softmax_out, tf.constant(delta, dtype=tf.float32)), tf.float64))
            if tp + fp == 0:
                delta += self.inc
                continue
            precision = tp/(tp + fp)
            recall = tf.cast(tp, tf.float64)
            aupr += (recall_prev - recall)*precision
            recall_prev = recall
            delta += self.inc
        if recall is None or precision is None:
            return 0
        aupr += recall*precision
        return aupr.numpy()*100
    
    def detection(self):
        # Positive = In Distribution (ID)
        # Negative = Out of Distribution (OOD)
        total_thresholds, total_pe = 0, 0
        delta = self.start
        while delta <= self.end:
            tp = tf.reduce_sum(tf.cast(tf.math.greater_equal(self.softmax_in, tf.constant(delta, dtype=tf.float32)), tf.int32))
            fn = self.total_in - tp
            fp = tf.reduce_sum(tf.cast(tf.math.greater_equal(self.softmax_out, tf.constant(delta, dtype=tf.float32)), tf.int32))
            tn = self.total_out - fp
            tpr = tp/(tp + fn)
            fpr = fp/(fp + tn)
            if tpr <= 0.9549 and tpr >= 0.9450:
                total_pe += 0.5*(1 - tpr) + 0.5*fpr
                total_thresholds += 1                
            delta += self.inc
        return (total_pe/total_thresholds).numpy()*100

class EvalVGG10IndexOfDispersion(EvalVGG10):

    def __init__(self, x_in, x_out, weight_file=None, entropy_file=None, **kwargs):
        self.total_in, self.total_out = x_in.shape[0], x_out.shape[0] 
        if entropy_file is None:
            samples = kwargs.get('samples', 10)
            self.model = PerturbedNN(3, 16, 10)
            self.model.build((None,) + x_in.shape[1:])
            load_config(weight_file, self.model)
            self.entropy_in = self._infer(x_in, samples)
            self.entropy_out = self._infer(x_out, samples)
            curr_date_time = (str(datetime.now()).replace(' ', '_')
                                                .replace(':', '_')
                                                .replace('-', '_')
                                                .replace('.', '_'))
            entropies = np.asarray([self.entropy_in, self.entropy_out])
            np.save(prefix + "results/cifar10/" + curr_date_time, entropies)
        else:
            entropies = np.load(entropy_file, allow_pickle=True)
            self.entropy_in, self.entropy_out = entropies[0], entropies[1]
            print(tf.reduce_mean(self.entropy_in), tf.math.reduce_std(self.entropy_in), tf.reduce_max(self.entropy_in)) 
            print(tf.reduce_mean(self.entropy_out), tf.math.reduce_std(self.entropy_out), tf.reduce_min(self.entropy_out))
            print(tf.reduce_sum(tf.cast(tf.math.greater_equal(self.entropy_out, tf.constant(3.27, dtype=tf.float32)), tf.int32)))
        self.softmax_in, self.softmax_out = self.entropy_in, self.entropy_out
        self.inc = 100
        self.start = 0
        self.end = 260000

    def _infer(self, x, samples):
        batch_size = 100
        x_inp = tf.reshape(x, x.shape)
        outputs = []
        test_data = tf.data.Dataset.from_tensor_slices(x_inp)
        test_data = test_data.repeat().batch(batch_size).prefetch(1)
        for _ in range(samples):
            output = []
            for _, batch_x in enumerate(test_data.take(x_inp.shape[0]//batch_size), 1):
                batch_output = list(self.model.call(batch_x).numpy())
                output.extend(batch_output)
            outputs.append(output)
        output_tensor = tf.convert_to_tensor(outputs)
        avg_pred = tf.reduce_mean(output_tensor, 0)
        entropy = -tf.reduce_sum(avg_pred*tf.math.log(avg_pred + 1e-10), 1)
        iod = tf.reduce_sum(tf.math.reduce_std(output_tensor, 0)**2/tf.reduce_mean(output_tensor, 0), 1)
        un = 1/iod + 10000/entropy
        print(tf.reduce_mean(1/iod).numpy(), tf.math.reduce_std(1/iod).numpy(), tf.reduce_min(1/iod).numpy(), tf.reduce_max(1/iod).numpy())
        print(tf.reduce_mean(1/entropy).numpy(), tf.math.reduce_std(1/entropy).numpy(), tf.reduce_min(1/entropy).numpy(), tf.reduce_max(1/entropy).numpy())
        print(tf.reduce_mean(un).numpy(), tf.math.reduce_std(un).numpy(), tf.reduce_min(un).numpy(), tf.reduce_max(un).numpy())
        return un

class EvalSVHNMHB(EvalVGG10):

    def __init__(self, x_in, x_out, weight_file=None, entropy_file=None, **kwargs):
        self.total_in, self.total_out = x_in.shape[0], x_out.shape[0]
        if entropy_file is None:
            self.mhb_dist_in = self._infer(x_in, weight_file)
            self.mhb_dist_out = self._infer(x_out, weight_file)
            curr_date_time = (str(datetime.now()).replace(' ', '_')
                                                .replace(':', '_')
                                                .replace('-', '_')
                                                .replace('.', '_'))
            mhb_dist = np.asarray([self.mhb_dist_in, self.mhb_dist_out])
            np.save(prefix + "results/cifar10/" + curr_date_time, mhb_dist)
            print(prefix + "results/cifar10/" + curr_date_time + ".npy")
        else:
            mhb_dist = np.load(entropy_file, allow_pickle=True)
            self.mhb_dist_in, self.mhb_dist_out = mhb_dist[0], mhb_dist[1]
            print(tf.reduce_min(self.mhb_dist_in), tf.reduce_max(self.mhb_dist_in)) 
            print(tf.reduce_min(self.mhb_dist_out), tf.reduce_max(self.mhb_dist_out))
        self.softmax_in, self.softmax_out = self.mhb_dist_in, self.mhb_dist_out
        self.inc = 0.1
        self.start = -150.
        self.end = 0
    
    def _infer(self, x, weight_file):
        batch_size = 100
        x_test_re = tf.convert_to_tensor(x, dtype=tf.float32)
        dataset = "cifar10"
        sigma = np.load(prefix + "weights/" + dataset + "/sigma.npy")
        test_data = tf.data.Dataset.from_tensor_slices(x_test_re)
        test_data = test_data.repeat().batch(batch_size).prefetch(1)
        f_x = []
        for i, batch_x in enumerate(test_data.take(x_test_re.shape[0]//batch_size), 1):
            self.model = PointCNN(3, 16, 10)
            self.model.build(batch_x.shape)
            load_config_new(weight_file, self.model)
            self.model.call(batch_x, is_training=True)
            f_x.extend(list(self.model.LayerWiseOutputs[-1].numpy()))
            del self.model
        f_x_tensor = tf.convert_to_tensor(f_x)
        mhbs = []
        for _cls in range(10):
            mu = np.load(prefix + "weights/" + dataset + "/mean_" + str(_cls) + ".npy")
            mu = np.reshape(mu, (1,) + mu.shape)
            mhb_mat = -tf.matmul(tf.matmul(f_x_tensor - mu, sigma), f_x_tensor - mu, transpose_b=True)
            mhbs.append(tf.linalg.diag_part(mhb_mat))
        mhbs = tf.convert_to_tensor(mhbs)
        mhbs = tf.reduce_max(mhbs, 0)
        print(mhbs.shape)
        return mhbs

class EvalBayesAdapter(EvalVGG10):

    def __init__(self, x_in, x_out, weight_file=None, entropy_file=None, **kwargs):
        self.total_in, self.total_out = x_in.shape[0], x_out.shape[0] 
        if entropy_file is None:
            samples = kwargs.get('samples', 10)
            self.model = PerturbedNN(3, 16, 10)
            self.model.build((None,) + x_in.shape[1:])
            load_config(weight_file, self.model)
            self.entropy_in = self._infer(x_in, samples)
            self.entropy_out = self._infer(x_out, samples)
            curr_date_time = (str(datetime.now()).replace(' ', '_')
                                                .replace(':', '_')
                                                .replace('-', '_')
                                                .replace('.', '_'))
            entropies = np.asarray([self.entropy_in, self.entropy_out])
            np.save(prefix + "results/cifar10/" + curr_date_time, entropies)
            print(prefix + "results/cifar10/" + curr_date_time)
        else:
            entropies = np.load(entropy_file, allow_pickle=True)
            self.entropy_in, self.entropy_out = entropies[0], entropies[1]
            print(tf.reduce_mean(self.entropy_in), tf.math.reduce_std(self.entropy_in), tf.reduce_max(self.entropy_in)) 
            print(tf.reduce_mean(self.entropy_out), tf.math.reduce_std(self.entropy_out), tf.reduce_min(self.entropy_out))
            print(tf.reduce_sum(tf.cast(tf.math.greater_equal(self.entropy_out, tf.constant(3.27, dtype=tf.float32)), tf.int32)))
        self.softmax_in, self.softmax_out = self.entropy_in, self.entropy_out
        self.inc = 1e-6
        self.start = -0.002
        self.end = 6e-08

    def _infer(self, x, samples):
        batch_size = 100
        x_inp = tf.reshape(x, x.shape)
        outputs = []
        test_data = tf.data.Dataset.from_tensor_slices(x_inp)
        test_data = test_data.repeat().batch(batch_size).prefetch(1)
        for _ in range(samples):
            output = []
            for _, batch_x in enumerate(test_data.take(x_inp.shape[0]//batch_size), 1):
                batch_output = list(self.model.call(batch_x).numpy())
                output.extend(batch_output)
            outputs.append(output)
        output_tensor = tf.convert_to_tensor(outputs)
        avg_pred = tf.reduce_mean(output_tensor, 0)
        entropy_avg = -tf.reduce_sum(avg_pred*tf.math.log(avg_pred + 1e-10), 1)
        avg_entropy = tf.reduce_mean(-tf.reduce_sum(avg_pred*tf.math.log(output_tensor + 1e-10), 2), 0)
        un = entropy_avg - avg_entropy
        print(tf.reduce_mean(un).numpy(), tf.math.reduce_std(un).numpy(), tf.reduce_min(un).numpy(), tf.reduce_max(un).numpy())
        return un

class EvalVGG10IndexOfDispersion(EvalVGG10):

    def __init__(self, x_in, x_out, weight_file=None, entropy_file=None, **kwargs):
        self.total_in, self.total_out = x_in.shape[0], x_out.shape[0]
        measure_name = kwargs.get('measure', 'M1') 
        if entropy_file is None:
            samples = kwargs.get('samples', 10)
            self.model = PerturbedNN(3, 16, 10)
            self.model.build((None,) + x_in.shape[1:])
            load_config(weight_file, self.model)
            ein = self._infer(x_in, samples, measure_name)
            eout = self._infer(x_out, samples, measure_name)
            entropies = [ein, eout]
            if measure_name == 'M2':
                self.entropy_in = 1/entropies[0][0] + 1000/entropies[0][1]
                self.entropy_out = 1/entropies[1][0] + 1000/entropies[1][1]
            else:
                self.entropy_in, self.entropy_out = entropies[0], entropies[1]
            curr_date_time = (str(datetime.now()).replace(' ', '_')
                                                .replace(':', '_')
                                                .replace('-', '_')
                                                .replace('.', '_'))
            entropies = np.asarray(entropies)
            np.save(prefix + "results/cifar10/" + curr_date_time, entropies)
            print(prefix + "results/cifar10/" + curr_date_time + ".npy")
        else:
            entropies = np.load(entropy_file, allow_pickle=True)
            if measure_name == 'M2':
                self.entropy_in = 1/entropies[0][0] + 1000/entropies[0][1]
                self.entropy_out = 1/entropies[1][0] + 1000/entropies[1][1]
            else:
                self.entropy_in, self.entropy_out = entropies[0], entropies[1]
            print(tf.reduce_min(self.entropy_in), tf.reduce_max(self.entropy_in)) 
            print(tf.reduce_min(self.entropy_out), tf.reduce_max(self.entropy_out))
        self.softmax_in, self.softmax_out = self.entropy_in, self.entropy_out
        print(self.softmax_in.shape, self.softmax_out.shape)
        self.inc = 100
        self.start = 0
        self.end = 44016

    def _infer(self, x, samples, measure_name):
        batch_size = 1000
        x_inp = tf.reshape(x, x.shape)
        outputs = []
        test_data = tf.data.Dataset.from_tensor_slices(x_inp)
        test_data = test_data.repeat().batch(batch_size).prefetch(1)
        for _ in range(samples):
            output = []
            for _, batch_x in enumerate(test_data.take(x_inp.shape[0]//batch_size), 1):
                batch_output = list(self.model.call(batch_x).numpy())
                output.extend(batch_output)
            outputs.append(output)
        output_tensor = tf.convert_to_tensor(outputs)
        if measure_name == 'M1':
            un = tf.reduce_sum(tf.math.reduce_std(output_tensor, 0)**2/tf.reduce_mean(output_tensor, 0), 1)
            return -tf.math.log(un)
        elif measure_name == 'M2':
            avg_pred = tf.reduce_mean(output_tensor, 0)
            entropy = -tf.reduce_sum(avg_pred*tf.math.log(avg_pred + 1e-10), 1)
            iod = tf.reduce_sum((tf.math.reduce_std(output_tensor, 0)**2)/(tf.reduce_mean(output_tensor, 0) + 1e-16), 1)
            ret = np.asarray([iod, entropy])
            return ret

x_cifar10_c = np.load("/content/drive/MyDrive/bnn/data/cifar10_corrupted/CIFAR-10-C/speckle_noise.npy", allow_pickle=True)
x_cifar10_c = np.array(x_cifar10_c, np.float32)
mean_te, std_te = np.mean(x_cifar10_c), np.std(x_cifar10_c)
x_cifar10_c = (x_cifar10_c - mean_te) / std_te

# Cell
num_samples, min_samples = 3, 2
disp_files = [] # ["./drive/My Drive/bnn/results/cifar10/2021_02_05_18_33_06_735315.npy"] # ["./drive/My Drive/bnn/results/cifar10/2021_02_03_15_14_37_598569.npy"] # ["./drive/My Drive/bnn/results/cifar10/2021_02_03_15_11_16_716748.npy"]
disp_files = disp_files + [None for i in range(num_samples - min_samples - len(disp_files))]
scores = {"FPR at 95 % TPR": 0, "AUROC": 0, "AUPR-In": 0, "Detection Error": 0}
i = 0
valid_samples = 0
for samples in range(min_samples, num_samples):
    grader = EvalVGG10IndexOfDispersion(x_train, x_cifar10_c, weight_file=prefix + "weights/cifar10/2020_10_15_14_52_27_357637", entropy_file=disp_files[i], samples=samples, measure='M2')
#     try:
#         scores["FPR at 95 % TPR"] = round(grader.fpr_at_95_tpr(), 2)
#         scores["AUROC"] = round(grader.auroc(), 2)
#         scores["AUPR-In"] = round(grader.aupr_in(), 2)
#         scores["Detection Error"] = round(grader.detection(), 2)
#         valid_samples += 1
#     except ZeroDivisionError:
#         continue
#     i += 1
# for key in scores:
#     scores[key] = scores[key]/valid_samples
print(scores)

M = np.load("./drive/My Drive/bnn/results/cifar10/2021_02_13_20_33_04_916078.npy")
M.shape

# Cell
num_samples, min_samples = 3, 2
disp_files = [] # ["./drive/My Drive/bnn/results/cifar10/2021_01_29_14_26_14_516472.npy"] # ["./drive/My Drive/bnn/results/cifar10/2021_01_29_13_00_17_358657.npy"] # ["./drive/My Drive/bnn/results/cifar10/2021_01_29_12_53_19_348602.npy"]
disp_files = disp_files + [None for i in range(num_samples - min_samples - len(disp_files))]
scores = {"FPR at 95 % TPR": 0, "AUROC": 0, "AUPR-In": 0, "Detection Error": 0}
i = 0
valid_samples = 0
for samples in range(min_samples, num_samples):
    grader = EvalBayesAdapter(x_test, x_test_c, weight_file=prefix + "weights/cifar10/bayes_adapter_2021_01_28_14_58_05_698532", entropy_file=disp_files[i], samples=samples)
    try:
        scores["FPR at 95 % TPR"] = round(grader.fpr_at_95_tpr(), 2)
        scores["AUROC"] = round(grader.auroc(), 2)
        scores["AUPR-In"] = round(grader.aupr_in(), 2)
        scores["Detection Error"] = round(grader.detection(), 2)
        valid_samples += 1
    except ZeroDivisionError:
        continue
    i += 1
for key in scores:
    scores[key] = scores[key]/valid_samples
print(scores)

"""{'FPR at 95 % TPR': 71.1, 'AUROC': 81.79, 'AUPR-In': 91.99, 'Detection Error': 38.03}"""

class EvalEnsembleResNet(EvalVGG10):

    def __init__(self, x_in, x_out, weight_files, entropy_file=None, **kwargs):
        self.total_in, self.total_out = x_in.shape[0], x_out.shape[0]
        self.models = []
        if entropy_file is None:
            self.entropy_in = self._infer(x_in, weight_files)
            self.entropy_out = self._infer(x_out, weight_files)
            curr_date_time = (str(datetime.now()).replace(' ', '_')
                                                .replace(':', '_')
                                                .replace('-', '_')
                                                .replace('.', '_'))
            entropies = np.asarray([self.entropy_in, self.entropy_out])
            np.save(prefix + "results/cifar10/" + curr_date_time, entropies)
            print(prefix + "results/cifar10/" + curr_date_time + ".npy")
        else:
            entropies = np.load(entropy_file, allow_pickle=True)
            self.entropy_in, self.entropy_out = entropies[0], entropies[1]
            print(tf.reduce_mean(self.entropy_in), tf.math.reduce_std(self.entropy_in), tf.reduce_min(self.entropy_in), tf.math.reduce_max(self.entropy_in)) 
            print(tf.reduce_mean(self.entropy_out), tf.math.reduce_std(self.entropy_out), tf.reduce_min(self.entropy_out), tf.math.reduce_max(self.entropy_out))
        self.softmax_in, self.softmax_out = self.entropy_in, self.entropy_out
        print(self.softmax_in.shape, self.softmax_out.shape)
        self.inc = 0.001
        self.start = 0
        self.end = 1.5

    def _infer(self, x, weight_files):

        def call_model(weight_file, x_inp):
            batch_size = 1000
            test_data = tf.data.Dataset.from_tensor_slices(x_inp)
            test_data = test_data.repeat().batch(batch_size).prefetch(1)
            output = []
            for _, batch_x in enumerate(test_data.take(x_inp.shape[0]//batch_size), 1):
                model = PointCNN(3, 16, 10)
                model.build((None,) + x_inp.shape[1:])
                load_config(weight_file, model)
                batch_output = list(model.call(batch_x).numpy())
                output.extend(batch_output)
                del model
            return tf.convert_to_tensor(output)

        outputs = []
        for weight_file in weight_files:
            outputs.append(call_model(weight_file, x))
        avg_pred = tf.reduce_mean(tf.convert_to_tensor(outputs), 0)
        un = tf.constant(0, dtype=tf.float32)
        for output in outputs:
            print(output.shape, avg_pred.shape)
            kl = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE)
            un += kl(output, avg_pred)
        return un
    
    def fpr_at_95_tpr(self):
        # Positive = In Distribution (ID)
        # Negative = Out of Distribution (OOD)
        delta = self.start
        total_fpr = 0
        total_thresholds = 0
        tprs, fprs = [], []
        while delta <= self.end:
            tp = tf.math.floor(tf.reduce_sum(tf.cast(tf.math.less_equal(self.softmax_in, tf.constant(delta, dtype=tf.float32)), tf.int32))/self.total_in*100)
            fn = 100 - tp
            fp = tf.math.floor(tf.reduce_sum(tf.cast(tf.math.less_equal(self.softmax_out, tf.constant(delta, dtype=tf.float32)), tf.int32))/self.total_out*100)
            tn = 100 - fp
            tpr = tp/(tp + fn)
            fpr = fp/(fp + tn)
            tprs.append(tpr)
            fprs.append(fpr)
            if tpr <= 0.9549 and tpr >= 0.9450:
                total_fpr += fpr
                total_thresholds += 1
            delta += self.inc
        np.save(prefix + "results/mnist/tpr_fpr_mhb.npy", np.asarray([tprs, fprs]))
        return (total_fpr/total_thresholds).numpy()*100
    
    def auroc(self):
        # Positive = In Distribution (ID)
        # Negative = Out of Distribution (OOD)
        delta = self.start
        auroc, fpr_prev = 0, 1
        while delta <= self.end:
            tp = tf.math.floor(tf.reduce_sum(tf.cast(tf.math.less_equal(self.softmax_in, tf.constant(delta, dtype=tf.float32)), tf.int32))/self.total_in*100)
            fn = 100 - tp
            fp = tf.math.floor(tf.reduce_sum(tf.cast(tf.math.less_equal(self.softmax_out, tf.constant(delta, dtype=tf.float32)), tf.int32))/self.total_out*100)
            tn = 100 - fp
            tpr = tp/(tp + fn)
            fpr = fp/(fp + tn)
            auroc += (fpr_prev - fpr)*tpr
            fpr_prev = fpr
            delta += self.inc
        auroc += fpr*tpr
        return auroc.numpy()*100
    
    def aupr_in(self):
        # Positive = In Distribution (ID)
        # Negative = Out of Distribution (OOD)
        aupr = 0
        recall_prev = 1
        recall, precision = None, None
        recalls, precisions = [], []
        delta = self.start
        while delta <= self.end:
            tp = tf.reduce_mean(tf.cast(tf.math.less_equal(self.softmax_in, tf.constant(delta, dtype=tf.float32)), tf.float64))
            fp = tf.reduce_mean(tf.cast(tf.math.less_equal(self.softmax_out, tf.constant(delta, dtype=tf.float32)), tf.float64))
            if tp + fp == 0:
                delta += self.inc
                continue
            precision = tp/(tp + fp)
            recall = tf.cast(tp, tf.float64)
            recalls.append(recall)
            precisions.append(precision)
            aupr += (recall_prev - recall)*precision
            recall_prev = recall
            delta += self.inc
        np.save(prefix + "results/mnist/prc.npy", np.asarray([recalls, precisions]))
        if recall is None or precision is None:
            return 0
        aupr += recall*precision
        return aupr.numpy()*100
    
    def detection(self):
        # Positive = In Distribution (ID)
        # Negative = Out of Distribution (OOD)
        total_thresholds, total_pe = 0, 0
        delta = self.start
        while delta <= self.end:
            tp = tf.math.floor(tf.reduce_sum(tf.cast(tf.math.less_equal(self.softmax_in, tf.constant(delta, dtype=tf.float32)), tf.int32))/self.total_in*100)
            fn = 100 - tp
            fp = tf.math.floor(tf.reduce_sum(tf.cast(tf.math.less_equal(self.softmax_out, tf.constant(delta, dtype=tf.float32)), tf.int32))/self.total_out*100)
            tn = 100 - fp
            tpr = tp/(tp + fn)
            fpr = fp/(fp + tn)
            if tpr <= 0.9549 and tpr >= 0.9450:
                total_pe += 0.5*(1 - tpr) + 0.5*fpr
                total_thresholds += 1                
            delta += self.inc
        return (total_pe/total_thresholds).numpy()*100

# Cell 7
weight_files = [
    prefix + "weights/cifar10/2021_01_30_09_22_31_500250",
    prefix + "weights/cifar10/2021_01_30_09_27_34_146847",
    # prefix + "weights/mnist/2021_01_30_08_40_43_545708",
    # prefix + "weights/mnist/2021_01_30_08_41_50_115262",
    # prefix + "weights/mnist/2021_01_30_08_42_23_109625"
]
grader = EvalEnsembleResNet(x_test, x_test_c, 
    weight_files=weight_files, 
    entropy_file="./drive/My Drive/bnn/results/cifar10/2021_01_31_14_43_16_230398.npy")
scores = {"FPR at 95 % TPR": None, "AUROC": None, "AUPR-In": None, "Detection Error": None}
scores["FPR at 95 % TPR"] = grader.fpr_at_95_tpr()
scores["AUROC"] = grader.auroc()
scores["AUPR-In"] = grader.aupr_in()
scores["Detection Error"] = grader.detection()
print(scores)

# Cell
grader = EvalSVHNMHB(x_test, tf.random.normal(x_test.shape, 0., 1.), weight_file=prefix + "weights/cifar10/cifar10resnet20_mean.npy")
scores = {"FPR at 95 % TPR": None, "AUROC": None, "AUPR-In": None, "Detection Error": None}
scores["FPR at 95 % TPR"] = grader.fpr_at_95_tpr()
scores["AUROC"] = grader.auroc()
scores["AUPR-In"] = grader.aupr_in()
scores["Detection Error"] = grader.detection()
print(scores)

# Cell
grader = EvalVGG10(x_test, tf.random.normal(x_test.shape, 0, 1), weight_file=prefix + "weights/cifar10/cifar10resnet20_mean.npy")
scores = {"FPR at 95 % TPR": None, "AUROC": None, "AUPR-In": None, "Detection Error": None}
scores["FPR at 95 % TPR"] = round(grader.fpr_at_95_tpr(), 2)
scores["AUROC"] = round(grader.auroc(), 2)
scores["AUPR-In"] = round(grader.aupr_in(), 2)
scores["Detection Error"] = round(grader.detection(), 2)
print(scores)

# Cell
def odin_noise(x_test, eps, weight_file):
    x_test_inp = tf.reshape(x_test, x_test.shape)
    model = PointCNNTScaled(3, 16, 10)
    model.build(x_test_inp.shape)
    load_config_new(weight_file, model)
    test_data = tf.data.Dataset.from_tensor_slices(x_test_inp)
    batch_size = 100
    test_data = test_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)
    grads = []
    for _, batch_x in enumerate(test_data.take(x_test_inp.shape[0]//batch_size), 1):
        with tf.GradientTape() as tape:
            tape.watch(batch_x)
            batch_output = model.call(batch_x, T=1000)
            log_outputs = tf.math.log(tf.reduce_max(batch_output, 1))
        grad = tape.gradient(log_outputs, batch_x)
        grads.extend(list(grad.numpy()))
    grads = tf.convert_to_tensor(grads)
    x_test_new = x_test_inp - eps*tf.sign(-grads)
    return tf.reshape(x_test_new, x_test_new.shape).numpy()

x_test_new = odin_noise(x_test, 0.0012, prefix + "weights/cifar10/cifar10resnet20_mean.npy")
x_test_f_new = odin_noise(tf.random.normal(x_test.shape, 0., 1.), 0.0012, prefix + "weights/cifar10/cifar10resnet20_mean.npy")

grader = EvalVGG10(x_test_new, x_test_f_new, weight_file=prefix + "weights/cifar10/cifar10resnet20_mean.npy")
scores = {"FPR at 95 % TPR": None, "AUROC": None, "AUPR-In": None, "Detection Error": None}
scores["FPR at 95 % TPR"] = round(grader.fpr_at_95_tpr(), 2)
scores["AUROC"] = round(grader.auroc(), 2)
scores["AUPR-In"] = round(grader.aupr_in(), 2)
scores["Detection Error"] = round(grader.detection(), 2)
print(scores)
