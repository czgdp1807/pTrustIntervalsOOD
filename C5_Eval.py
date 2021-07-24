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
class PerturbedNN(tf.keras.Model):

    def __init__(self, num_dense_blocks, growth_rate, depth, compression_factor, num_classes=10, optimizer=None):
        super(PerturbedNN, self).__init__()
        self.num_dense_blocks, self.growth_rate, self.depth, self.compression_factor, self.num_classes = \
            num_dense_blocks, growth_rate, depth, compression_factor, num_classes
        num_bottleneck_layers = (depth - 4) // (2 * num_dense_blocks)
        num_filters_bef_dense_block = 2 * growth_rate
        # start model definition
        # densenet CNNs (composite function) are made of BN-ReLU-Conv2D
        self.Layers, self.TrainableLayers, self.TransferLayers = [], [], []
        self.Layers.append(tf.keras.layers.BatchNormalization())
        self.TransferLayers.append(self.Layers[-1])
        self.Layers.append(tf.keras.layers.Activation('relu'))
        self.Layers.append(PerturbedConv2D(num_filters_bef_dense_block, kernel_size=3, strides=1, padding='same'))
        self.TrainableLayers.append(self.Layers[-1])
        self.TransferLayers.append(self.Layers[-1])
        self.Layers.append(tf.keras.layers.Concatenate())

        # stack of dense blocks bridged by transition layers
        for i in range(num_dense_blocks):
            # a dense block is a stack of bottleneck layers
            for j in range(num_bottleneck_layers):
                self.Layers.append(tf.keras.layers.BatchNormalization())
                self.TransferLayers.append(self.Layers[-1])
                self.Layers.append(tf.keras.layers.Activation('relu'))
                self.Layers.append(PerturbedConv2D(4 * growth_rate, kernel_size=1, strides=1, padding='same'))
                self.TrainableLayers.append(self.Layers[-1])
                self.TransferLayers.append(self.Layers[-1])
                self.Layers.append(tf.keras.layers.Dropout(0.2))
                self.Layers.append(tf.keras.layers.BatchNormalization())
                self.TransferLayers.append(self.Layers[-1])
                self.Layers.append(tf.keras.layers.Activation('relu'))
                self.Layers.append(PerturbedConv2D(growth_rate, kernel_size=3, strides=1, padding='same'))
                self.TrainableLayers.append(self.Layers[-1])
                self.TransferLayers.append(self.Layers[-1])
                self.Layers.append(tf.keras.layers.Dropout(0.2))
                self.Layers.append(tf.keras.layers.Concatenate())

            # no transition layer after the last dense block
            if i == num_dense_blocks - 1:
                continue

            # transition layer compresses num of feature maps and reduces the size by 2
            num_filters_bef_dense_block += num_bottleneck_layers * growth_rate
            num_filters_bef_dense_block = int(num_filters_bef_dense_block * compression_factor)
            self.Layers.append(tf.keras.layers.BatchNormalization())
            self.TransferLayers.append(self.Layers[-1])
            self.Layers.append(PerturbedConv2D(num_filters_bef_dense_block, kernel_size=1, strides=1, padding='same'))
            self.TrainableLayers.append(self.Layers[-1])
            self.TransferLayers.append(self.Layers[-1])
            self.Layers.append(tf.keras.layers.Dropout(0.2))
            self.Layers.append(tf.keras.layers.AveragePooling2D())


        # add classifier on top
        # after average pooling, size of feature map is 1 x 1
        self.Layers.append(tf.keras.layers.AveragePooling2D(pool_size=8))
        self.Layers.append(tf.keras.layers.Flatten())
        self.Layers.append(PerturbedDense(num_classes))
        self.TrainableLayers.append(self.Layers[-1])
        self.TransferLayers.append(self.Layers[-1])
    
        self.optimizer = optimizer
    
    def build(self, input_shape):
        num_dense_blocks, growth_rate, depth, compression_factor, num_classes = \
            self.num_dense_blocks, self.growth_rate, self.depth, self.compression_factor, self.num_classes
        num_bottleneck_layers = (depth - 4) // (2 * num_dense_blocks)
        num_filters_bef_dense_block = 2 * growth_rate
        l = 0
        self.Layers[l].build(input_shape)
        x = self.Layers[l].compute_output_shape(input_shape)
        l += 1
        self.Layers[l].build(x)
        x = self.Layers[l].compute_output_shape(x)
        l += 1
        self.Layers[l].build(x)
        x = self.Layers[l].compute_output_shape(x)
        l += 1
        self.Layers[l].build([input_shape, x])
        x = self.Layers[l].compute_output_shape([input_shape, x])
        l += 1

        # stack of dense blocks bridged by transition layers
        for i in range(num_dense_blocks):
            # a dense block is a stack of bottleneck layers
            for j in range(num_bottleneck_layers):
                self.Layers[l].build(x)
                y = self.Layers[l].compute_output_shape(x)
                l += 1
                self.Layers[l].build(y)
                y = self.Layers[l].compute_output_shape(y)
                l += 1
                self.Layers[l].build(y)
                y = self.Layers[l].compute_output_shape(y)
                l += 1
                self.Layers[l].build(y)
                y = self.Layers[l].compute_output_shape(y)
                l += 1
                self.Layers[l].build(y)
                y = self.Layers[l].compute_output_shape(y)
                l += 1
                self.Layers[l].build(y)
                y = self.Layers[l].compute_output_shape(y)
                l += 1
                self.Layers[l].build(y)
                y = self.Layers[l].compute_output_shape(y)
                l += 1
                self.Layers[l].build(y)
                y = self.Layers[l].compute_output_shape(y)
                l += 1
                self.Layers[l].build([x, y])
                x = self.Layers[l].compute_output_shape([x, y])
                l += 1

            # no transition layer after the last dense block
            if i == num_dense_blocks - 1:
                continue

            # transition layer compresses num of feature maps and reduces the size by 2
            num_filters_bef_dense_block += num_bottleneck_layers * growth_rate
            num_filters_bef_dense_block = int(num_filters_bef_dense_block * compression_factor)
            self.Layers[l].build(x)
            y = self.Layers[l].compute_output_shape(x)
            l += 1
            self.Layers[l].build(y)
            y = self.Layers[l].compute_output_shape(y)
            l += 1
            self.Layers[l].build(y)
            y = self.Layers[l].compute_output_shape(y)
            l += 1
            self.Layers[l].build(y)
            x = self.Layers[l].compute_output_shape(y)
            l += 1

        # add classifier on top
        # after average pooling, size of feature map is 1 x 1
        self.Layers[l].build(x)
        x = self.Layers[l].compute_output_shape(x)
        l += 1
        self.Layers[l].build(x)
        y = self.Layers[l].compute_output_shape(x)
        l += 1
        self.Layers[l].build(y)
    
    def call(self, inputs, is_training=False):
        num_dense_blocks, growth_rate, depth, compression_factor, num_classes = \
            self.num_dense_blocks, self.growth_rate, self.depth, self.compression_factor, self.num_classes
        num_bottleneck_layers = (depth - 4) // (2 * num_dense_blocks)
        num_filters_bef_dense_block = 2 * growth_rate
        # start model definition
        # densenet CNNs (composite function) are made of BN-ReLU-Conv2D
        l = 0
        x = self.Layers[l].call(inputs)
        l += 1
        x = self.Layers[l].call(x)
        l += 1
        x = self.Layers[l].call(x)
        l += 1
        x = self.Layers[l].call([inputs, x])
        l += 1

        # stack of dense blocks bridged by transition layers
        for i in range(num_dense_blocks):
            # a dense block is a stack of bottleneck layers
            for j in range(num_bottleneck_layers):
                y = self.Layers[l].call(x)
                l += 1
                y = self.Layers[l].call(y)
                l += 1
                y = self.Layers[l].call(y)
                l += 1
                y = self.Layers[l].call(y)
                l += 1
                y = self.Layers[l].call(y)
                l += 1
                y = self.Layers[l].call(y)
                l += 1
                y = self.Layers[l].call(y)
                l += 1
                y = self.Layers[l].call(y)
                l += 1
                x = self.Layers[l].call([x, y])
                l += 1

            # no transition layer after the last dense block
            if i == num_dense_blocks - 1:
                continue

            # transition layer compresses num of feature maps and reduces the size by 2
            num_filters_bef_dense_block += num_bottleneck_layers * growth_rate
            num_filters_bef_dense_block = int(num_filters_bef_dense_block * compression_factor)
            y = self.Layers[l].call(x)
            l += 1
            y = self.Layers[l].call(y)
            l += 1
            y = self.Layers[l].call(y)
            l += 1
            x = self.Layers[l].call(y)
            l += 1


        # add classifier on top
        # after average pooling, size of feature map is 1 x 1
        x = self.Layers[l].call(x)
        l += 1
        y = self.Layers[l].call(x)
        l += 1
        outputs = self.Layers[l].call(y)
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
            sigma_penalty += kwargs.get('c', 0.01)*tf.reduce_sum(-tf.math.log(tf.math.exp(layer.kernel_rho)))
        return (cse/samples)*kwargs.get('a', 1) + entropy*kwargs.get('b', 1) + sigma_penalty, cse/samples, entropy, sigma_penalty
    
    def compute_gradients(self, inputs, targets, **kwargs):
        _vars = []
        with tf.GradientTape(persistent=True) as tape:
            for layer in self.TrainableLayers:
                _vars.append(layer.kernel_rho)
            tape.watch(_vars)
            F, cse, entropy, sigma_penalty = self.get_loss(inputs, targets, 5, **kwargs)

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
class PointCNN(tf.keras.Model):

    def __init__(self, num_dense_blocks, growth_rate, depth, compression_factor, num_classes=10, optimizer=None):
        super(PointCNN, self).__init__()
        self.num_dense_blocks, self.growth_rate, self.depth, self.compression_factor, self.num_classes = \
            num_dense_blocks, growth_rate, depth, compression_factor, num_classes
        num_bottleneck_layers = (depth - 4) // (2 * num_dense_blocks)
        num_filters_bef_dense_block = 2 * growth_rate
        # start model definition
        # densenet CNNs (composite function) are made of BN-ReLU-Conv2D
        self.Layers, self.TrainableLayers, self.TransferLayers = [], [], []
        self.LayerWiseOutputs = []
        self.Layers.append(tf.keras.layers.BatchNormalization())
        self.TransferLayers.append(self.Layers[-1])
        self.Layers.append(tf.keras.layers.Activation('relu'))
        self.Layers.append(PointConv2D(num_filters_bef_dense_block, kernel_size=3, strides=1, padding='same'))
        self.TrainableLayers.append(self.Layers[-1])
        self.TransferLayers.append(self.Layers[-1])
        self.Layers.append(tf.keras.layers.Concatenate())

        # stack of dense blocks bridged by transition layers
        for i in range(num_dense_blocks):
            # a dense block is a stack of bottleneck layers
            for j in range(num_bottleneck_layers):
                self.Layers.append(tf.keras.layers.BatchNormalization())
                self.TransferLayers.append(self.Layers[-1])
                self.Layers.append(tf.keras.layers.Activation('relu'))
                self.Layers.append(PointConv2D(4 * growth_rate, kernel_size=1, strides=1, padding='same'))
                self.TrainableLayers.append(self.Layers[-1])
                self.TransferLayers.append(self.Layers[-1])
                self.Layers.append(tf.keras.layers.Dropout(0.2))
                self.Layers.append(tf.keras.layers.BatchNormalization())
                self.TransferLayers.append(self.Layers[-1])
                self.Layers.append(tf.keras.layers.Activation('relu'))
                self.Layers.append(PointConv2D(growth_rate, kernel_size=3, strides=1, padding='same'))
                self.TrainableLayers.append(self.Layers[-1])
                self.TransferLayers.append(self.Layers[-1])
                self.Layers.append(tf.keras.layers.Dropout(0.2))
                self.Layers.append(tf.keras.layers.Concatenate())

            # no transition layer after the last dense block
            if i == num_dense_blocks - 1:
                continue

            # transition layer compresses num of feature maps and reduces the size by 2
            num_filters_bef_dense_block += num_bottleneck_layers * growth_rate
            num_filters_bef_dense_block = int(num_filters_bef_dense_block * compression_factor)
            self.Layers.append(tf.keras.layers.BatchNormalization())
            self.TransferLayers.append(self.Layers[-1])
            self.Layers.append(PointConv2D(num_filters_bef_dense_block, kernel_size=1, strides=1, padding='same'))
            self.TrainableLayers.append(self.Layers[-1])
            self.TransferLayers.append(self.Layers[-1])
            self.Layers.append(tf.keras.layers.Dropout(0.2))
            self.Layers.append(tf.keras.layers.AveragePooling2D())


        # add classifier on top
        # after average pooling, size of feature map is 1 x 1
        self.Layers.append(tf.keras.layers.AveragePooling2D(pool_size=8))
        self.Layers.append(tf.keras.layers.Flatten())
        self.Layers.append(PointDense(num_classes))
        self.TrainableLayers.append(self.Layers[-1])
        self.TransferLayers.append(self.Layers[-1])
    
        self.optimizer = optimizer
    
    def build(self, input_shape):
        num_dense_blocks, growth_rate, depth, compression_factor, num_classes = \
            self.num_dense_blocks, self.growth_rate, self.depth, self.compression_factor, self.num_classes
        num_bottleneck_layers = (depth - 4) // (2 * num_dense_blocks)
        num_filters_bef_dense_block = 2 * growth_rate
        l = 0
        self.Layers[l].build(input_shape)
        x = self.Layers[l].compute_output_shape(input_shape)
        l += 1
        self.Layers[l].build(x)
        x = self.Layers[l].compute_output_shape(x)
        l += 1
        self.Layers[l].build(x)
        x = self.Layers[l].compute_output_shape(x)
        l += 1
        self.Layers[l].build([input_shape, x])
        x = self.Layers[l].compute_output_shape([input_shape, x])
        l += 1

        # stack of dense blocks bridged by transition layers
        for i in range(num_dense_blocks):
            # a dense block is a stack of bottleneck layers
            for j in range(num_bottleneck_layers):
                self.Layers[l].build(x)
                y = self.Layers[l].compute_output_shape(x)
                l += 1
                self.Layers[l].build(y)
                y = self.Layers[l].compute_output_shape(y)
                l += 1
                self.Layers[l].build(y)
                y = self.Layers[l].compute_output_shape(y)
                l += 1
                self.Layers[l].build(y)
                y = self.Layers[l].compute_output_shape(y)
                l += 1
                self.Layers[l].build(y)
                y = self.Layers[l].compute_output_shape(y)
                l += 1
                self.Layers[l].build(y)
                y = self.Layers[l].compute_output_shape(y)
                l += 1
                self.Layers[l].build(y)
                y = self.Layers[l].compute_output_shape(y)
                l += 1
                self.Layers[l].build(y)
                y = self.Layers[l].compute_output_shape(y)
                l += 1
                self.Layers[l].build([x, y])
                x = self.Layers[l].compute_output_shape([x, y])
                l += 1

            # no transition layer after the last dense block
            if i == num_dense_blocks - 1:
                continue

            # transition layer compresses num of feature maps and reduces the size by 2
            num_filters_bef_dense_block += num_bottleneck_layers * growth_rate
            num_filters_bef_dense_block = int(num_filters_bef_dense_block * compression_factor)
            self.Layers[l].build(x)
            y = self.Layers[l].compute_output_shape(x)
            l += 1
            self.Layers[l].build(y)
            y = self.Layers[l].compute_output_shape(y)
            l += 1
            self.Layers[l].build(y)
            y = self.Layers[l].compute_output_shape(y)
            l += 1
            self.Layers[l].build(y)
            x = self.Layers[l].compute_output_shape(y)
            l += 1

        # add classifier on top
        # after average pooling, size of feature map is 1 x 1
        self.Layers[l].build(x)
        x = self.Layers[l].compute_output_shape(x)
        l += 1
        self.Layers[l].build(x)
        y = self.Layers[l].compute_output_shape(x)
        l += 1
        self.Layers[l].build(y)
    
    def call(self, inputs, is_training=False):
        num_dense_blocks, growth_rate, depth, compression_factor, num_classes = \
            self.num_dense_blocks, self.growth_rate, self.depth, self.compression_factor, self.num_classes
        num_bottleneck_layers = (depth - 4) // (2 * num_dense_blocks)
        num_filters_bef_dense_block = 2 * growth_rate
        # start model definition
        # densenet CNNs (composite function) are made of BN-ReLU-Conv2D
        l = 0
        x = self.Layers[l].call(inputs)
        l += 1
        x = self.Layers[l].call(x)
        l += 1
        x = self.Layers[l].call(x)
        l += 1
        x = self.Layers[l].call([inputs, x])
        l += 1

        # stack of dense blocks bridged by transition layers
        for i in range(num_dense_blocks):
            # a dense block is a stack of bottleneck layers
            for j in range(num_bottleneck_layers):
                y = self.Layers[l].call(x)
                l += 1
                y = self.Layers[l].call(y)
                l += 1
                y = self.Layers[l].call(y)
                l += 1
                y = self.Layers[l].call(y)
                l += 1
                y = self.Layers[l].call(y)
                l += 1
                y = self.Layers[l].call(y)
                l += 1
                y = self.Layers[l].call(y)
                l += 1
                y = self.Layers[l].call(y)
                l += 1
                x = self.Layers[l].call([x, y])
                l += 1

            # no transition layer after the last dense block
            if i == num_dense_blocks - 1:
                continue

            # transition layer compresses num of feature maps and reduces the size by 2
            num_filters_bef_dense_block += num_bottleneck_layers * growth_rate
            num_filters_bef_dense_block = int(num_filters_bef_dense_block * compression_factor)
            y = self.Layers[l].call(x)
            l += 1
            y = self.Layers[l].call(y)
            l += 1
            y = self.Layers[l].call(y)
            l += 1
            x = self.Layers[l].call(y)
            l += 1


        # add classifier on top
        # after average pooling, size of feature map is 1 x 1
        x = self.Layers[l].call(x)
        l += 1
        y = self.Layers[l].call(x)
        l += 1
        outputs = self.Layers[l].call(y)
        self.LayerWiseOutputs.append(outputs)
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
            sigma_penalty += kwargs.get('c', 0.01)*tf.reduce_sum(-tf.math.log(tf.math.exp(layer.kernel_rho)))
        return (cse/samples)*kwargs.get('a', 1) + entropy*kwargs.get('b', 1) + sigma_penalty, cse/samples, entropy, sigma_penalty
    
    def compute_gradients(self, inputs, targets, **kwargs):
        _vars = []
        with tf.GradientTape(persistent=True) as tape:
            for layer in self.TrainableLayers:
                _vars.append(layer.kernel_rho)
            tape.watch(_vars)
            F, cse, entropy, sigma_penalty = self.get_loss(inputs, targets, 5, **kwargs)

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
        for layer in self.TransferLayers:
            if isinstance(layer, (PointConv2D, PointDense)):
                np_kernel_mu.append(layer.kernel_mu.numpy())
                np_kernel_mu.append(layer.bias_mu.numpy())
            elif isinstance(layer, tf.keras.layers.BatchNormalization):
                np_kernel_mu.append(layer.gamma.numpy())
                np_kernel_mu.append(layer.beta.numpy())
                np_kernel_mu.append(layer.moving_mean.numpy())
                np_kernel_mu.append(layer.moving_variance.numpy())
        return np.asarray(np_kernel_mu)
 
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
        num_dense_blocks, growth_rate, depth, compression_factor, num_classes = \
            self.num_dense_blocks, self.growth_rate, self.depth, self.compression_factor, self.num_classes
        num_bottleneck_layers = (depth - 4) // (2 * num_dense_blocks)
        num_filters_bef_dense_block = 2 * growth_rate
        # start model definition
        # densenet CNNs (composite function) are made of BN-ReLU-Conv2D
        l = 0
        x = self.Layers[l].call(inputs)
        l += 1
        x = self.Layers[l].call(x)
        l += 1
        x = self.Layers[l].call(x)
        l += 1
        x = self.Layers[l].call([inputs, x])
        l += 1

        # stack of dense blocks bridged by transition layers
        for i in range(num_dense_blocks):
            # a dense block is a stack of bottleneck layers
            for j in range(num_bottleneck_layers):
                y = self.Layers[l].call(x)
                l += 1
                y = self.Layers[l].call(y)
                l += 1
                y = self.Layers[l].call(y)
                l += 1
                y = self.Layers[l].call(y)
                l += 1
                y = self.Layers[l].call(y)
                l += 1
                y = self.Layers[l].call(y)
                l += 1
                y = self.Layers[l].call(y)
                l += 1
                y = self.Layers[l].call(y)
                l += 1
                x = self.Layers[l].call([x, y])
                l += 1

            # no transition layer after the last dense block
            if i == num_dense_blocks - 1:
                continue

            # transition layer compresses num of feature maps and reduces the size by 2
            num_filters_bef_dense_block += num_bottleneck_layers * growth_rate
            num_filters_bef_dense_block = int(num_filters_bef_dense_block * compression_factor)
            y = self.Layers[l].call(x)
            l += 1
            y = self.Layers[l].call(y)
            l += 1
            y = self.Layers[l].call(y)
            l += 1
            x = self.Layers[l].call(y)
            l += 1


        # add classifier on top
        # after average pooling, size of feature map is 1 x 1
        x = self.Layers[l].call(x)
        l += 1
        y = self.Layers[l].call(x)
        l += 1
        outputs = self.Layers[l].call(y)
        if not is_training:
            outputs = tf.nn.softmax(outputs/T)
        return outputs

# Cell
mean, std = 0, 255.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
y_train, y_test = np.squeeze(y_train), np.squeeze(y_test)
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
x_train, x_test = (x_train - mean)/std, (x_test - mean)/std

# Cell
mean, std = 0, 255.
(x_train_f, y_train_f), (x_test_f, y_test_f) = tf.keras.datasets.cifar100.load_data()
y_train_f, y_test_f = np.squeeze(y_train_f), np.squeeze(y_test_f)
x_train_f, x_test_f = np.array(x_train_f, np.float32), np.array(x_test_f, np.float32)
x_train_f, x_test_f = (x_train_f - mean)/std, (x_test_f - mean)/std

# Cell
x_test_c, y_test_c = tfds.as_numpy(tfds.load(name="cmaterdb", split='test', batch_size=-1, as_supervised=True))
x_test_c = np.array(x_test_c, np.float32)
mean_te, std_te = 0., 255.
x_test_c = (x_test_c - mean_te) / std_te

x_cifar10_c = np.load("/content/drive/MyDrive/bnn/data/cifar10_corrupted/CIFAR-10-C/speckle_noise.npy", allow_pickle=True)
x_cifar10_c = np.array(x_cifar10_c, np.float32)
mean_te, std_te = 0, 255.
x_cifar10_c = (x_cifar10_c - mean_te) / std_te

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
            self.model = PointCNN(3, 12, 100, 0.5)
            self.model.build((None,) + x_in.shape[1:])
            load_config_new(weight_file, self.model)
            self.softmax_in = self._infer(x_in)
            self.softmax_out = self._infer(x_out)
            del self.model
            curr_date_time = (str(datetime.now()).replace(' ', '_')
                                                .replace(':', '_')
                                                .replace('-', '_')
                                                .replace('.', '_'))
            softmax_scores = np.asarray([self.softmax_in, self.softmax_out])
            np.save(prefix + "results/cifar100/" + curr_date_time, softmax_scores)
        else:
            softmax_scores = np.load(entropy_file, allow_pickle=True)
            self.softmax_in, self.softmax_out = softmax_scores[0], softmax_scores[1]
        self.inc = 0.001
        self.start = 0
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
            if tpr > 0.9:
                print(tpr.numpy(), fpr.numpy())
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
        self.inc = 0.01
        self.start = -110
        self.end = 0
    
    def _infer(self, x, weight_file):
        batch_size = 100
        x_test_re = tf.convert_to_tensor(x, dtype=tf.float32)
        dataset = "cifar10"
        sigma = np.load(prefix + "weights/" + dataset + "/sigma.npy")
        test_data = tf.data.Dataset.from_tensor_slices(x_test_re)
        test_data = test_data.repeat().batch(batch_size).prefetch(1)
        f_x = []
        self.model = PointCNN(3, 12, 100, 0.5)
        self.model.build((None, ) + (32, 32, 3))
        load_config_new(weight_file, self.model)
        for i, batch_x in enumerate(test_data.take(x_test_re.shape[0]//batch_size), 1):
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
            self.model = PerturbedNN(3, 12, 100, 0.5)
            self.model.build((None,) + x_in.shape[1:])
            load_config(weight_file, self.model)
            self.entropy_in = self._infer(x_in, samples)
            self.entropy_out = self._infer(x_out, samples)
            del self.model
            curr_date_time = (str(datetime.now()).replace(' ', '_')
                                                .replace(':', '_')
                                                .replace('-', '_')
                                                .replace('.', '_'))
            entropies = np.asarray([self.entropy_in, self.entropy_out])
            np.save(prefix + "results/cifar100/" + curr_date_time, entropies)
            print(prefix + "results/cifar100/" + curr_date_time)
        else:
            entropies = np.load(entropy_file, allow_pickle=True)
            self.entropy_in, self.entropy_out = entropies[0], entropies[1]
            print(tf.reduce_min(self.entropy_in), tf.reduce_max(self.entropy_in)) 
            print(tf.reduce_min(self.entropy_out), tf.reduce_max(self.entropy_out))
        self.softmax_in, self.softmax_out = self.entropy_in, self.entropy_out
        self.inc = 0.00001
        self.start = -0.0007
        self.end = 0.0001

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
            self.model = PerturbedNN(3, 12, 100, 0.5)
            self.model.build((None,) + x_in.shape[1:])
            load_config(weight_file, self.model)
            ein = self._infer(x_in, samples, measure_name)
            eout = self._infer(x_out, samples, measure_name)
            del self.model
            entropies = [ein, eout]
            print(ein.shape, eout.shape)
            if measure_name == 'M2':
                self.entropy_in = 1/(entropies[0][0] + 1e-16) + 1000/(entropies[0][1] + 1e-16)
                self.entropy_out = 1/(entropies[1][0] + 1e-16) + 1000/(entropies[1][1] + 1e-16)
            else:
                self.entropy_in, self.entropy_out = entropies[0], entropies[1]
            curr_date_time = (str(datetime.now()).replace(' ', '_')
                                                .replace(':', '_')
                                                .replace('-', '_')
                                                .replace('.', '_'))
            entropies = np.asarray(entropies)
            np.save(prefix + "results/cifar100/" + curr_date_time, entropies)
            print(prefix + "results/cifar100/" + curr_date_time + ".npy")
        else:
            e_in = np.load(entropy_file + "_in.npy", allow_pickle=True)
            e_out = np.load(entropy_file + "_out.npy", allow_pickle=True)
            entropies = [e_in, e_out]
            if measure_name == 'M2':
                self.entropy_in = 1/(entropies[0][0] + 1e-16) + 1e-6/(entropies[0][1] + 1e-16)
                self.entropy_out = 1/(entropies[1][0] + 1e-16) + 1e-6/(entropies[1][1] + 1e-16)
            else:
                self.entropy_in, self.entropy_out = entropies[0], entropies[1]
            print(tf.reduce_min(self.entropy_in), tf.reduce_max(self.entropy_in)) 
            print(tf.reduce_min(self.entropy_out), tf.reduce_max(self.entropy_out))
        self.softmax_in, self.softmax_out = self.entropy_in, self.entropy_out
        print(self.softmax_in.shape, self.softmax_out.shape)
        self.inc = 10
        self.start = 13000
        self.end = 40000

    def _infer(self, x, samples, measure_name):
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
        if measure_name == 'M1':
            un = tf.reduce_sum(tf.math.reduce_std(output_tensor, 0)**2/(tf.reduce_mean(output_tensor, 0) + 1e-16), 1)
            return -tf.math.log(un)
        elif measure_name == 'M2':
            avg_pred = tf.reduce_mean(output_tensor, 0)
            entropy = -tf.reduce_sum(avg_pred*tf.math.log(avg_pred + 1e-10), 1)
            iod = tf.reduce_sum((tf.math.reduce_std(output_tensor, 0)**2)/(tf.reduce_mean(output_tensor, 0) + 1e-16), 1)
            ret = np.asarray([iod, entropy])
            return ret

# Cell
num_samples, min_samples = 3, 2
disp_files = [] #["./drive/My Drive/bnn/results/cifar100/2021_03_14_12_39_52_174998"] # ["./drive/My Drive/bnn/results/cifar100/2021_03_14_12_17_50_499521.npy"] # ["./drive/My Drive/bnn/results/cifar100/2021_02_03_15_44_56_745753.npy"] # ["/content/drive/My Drive/bnn/results/cifar100/2020_10_25_13_36_55_962363.npy"] # ["/content/drive/My Drive/bnn/results/cifar100/2020_10_25_13_32_01_903890.npy"]
disp_files = disp_files + [None for i in range(num_samples - min_samples - len(disp_files))]
scores = {"FPR at 95 % TPR": 0, "AUROC": 0, "AUPR-In": 0, "Detection Error": 0}
i = 0
valid_samples = 0
print(x_test.shape, x_cifar10_c.shape)
for samples in range(min_samples, num_samples):
    grader = EvalVGG10IndexOfDispersion(x_test, x_cifar10_c[40000:50000], weight_file=prefix + "weights/cifar10/2021_03_14_12_11_09_446850", entropy_file=disp_files[i], samples=samples, measure='M2')
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

class EvalEnsembleDenseNet(EvalVGG10):

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
        self.inc = 0.01
        self.start = -1
        self.end = 4

    def _infer(self, x, weight_files):

        def call_model(weight_file, x_inp):
            batch_size = 1000
            test_data = tf.data.Dataset.from_tensor_slices(x_inp)
            test_data = test_data.repeat().batch(batch_size).prefetch(1)
            output = []
            for _, batch_x in enumerate(test_data.take(x_inp.shape[0]//batch_size), 1):
                model = PointCNN(3, 12, 100, 0.5)
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
    prefix + "weights/cifar100/2021_03_14_08_45_35_585737",
    prefix + "weights/cifar100/2021_03_14_08_49_22_364746",
    prefix + "weights/cifar100/2021_03_14_08_53_18_372364",
    prefix + "weights/cifar100/2021_03_14_08_59_41_276215",
    prefix + "weights/cifar100/2021_03_14_09_03_04_368781"
]
grader = EvalEnsembleDenseNet(x_test, x_test_f, 
    weight_files=weight_files, entropy_file="./drive/My Drive/bnn/results/cifar10/2021_03_14_09_22_56_794617.npy")
scores = {"FPR at 95 % TPR": None, "AUROC": None, "AUPR-In": None, "Detection Error": None}
scores["FPR at 95 % TPR"] = grader.fpr_at_95_tpr()
scores["AUROC"] = grader.auroc()
scores["AUPR-In"] = grader.aupr_in()
scores["Detection Error"] = grader.detection()
print(scores)

# Cell
grader = EvalVGG10(x_test, x_test_c, weight_file=prefix + "weights/cifar10/cifar10_densenet_mean.npy")
scores = {"FPR at 95 % TPR": None, "AUROC": None, "AUPR-In": None, "Detection Error": None}
scores["FPR at 95 % TPR"] = round(grader.fpr_at_95_tpr(), 2)
scores["AUROC"] = round(grader.auroc(), 2)
scores["AUPR-In"] = round(grader.aupr_in(), 2)
scores["Detection Error"] = round(grader.detection(), 2)
print(scores)

# Cell
grader = EvalSVHNMHB(x_test, x_test_c, weight_file=prefix + "weights/cifar10/cifar10_densenet_mean.npy", entropy_file="./drive/My Drive/bnn/results/cifar10/2021_03_13_12_32_59_129344.npy")
scores = {"FPR at 95 % TPR": None, "AUROC": None, "AUPR-In": None, "Detection Error": None}
scores["FPR at 95 % TPR"] = grader.fpr_at_95_tpr()
scores["AUROC"] = grader.auroc()
scores["AUPR-In"] = grader.aupr_in()
scores["Detection Error"] = grader.detection()
print(scores)

# Cell
num_samples, min_samples = 3, 2
disp_files = [] # ["./drive/My Drive/bnn/results/cifar100/2021_03_14_09_35_25_113243.npy"] # ["./drive/My Drive/bnn/results/cifar100/2021_01_29_16_35_11_386719.npy"] # ["/content/drive/My Drive/bnn/results/cifar100/2020_10_25_13_36_55_962363.npy"] # ["/content/drive/My Drive/bnn/results/cifar100/2020_10_25_13_32_01_903890.npy"]
disp_files = disp_files + [None for i in range(num_samples - min_samples - len(disp_files))]
scores = {"FPR at 95 % TPR": 0, "AUROC": 0, "AUPR-In": 0, "Detection Error": 0}
i = 0
valid_samples = 0
for samples in range(min_samples, num_samples):
    grader = EvalBayesAdapter(x_test, x_test_c, weight_file=prefix + "weights/cifar10/bayes_adapter_2021_03_14_09_31_46_218224", entropy_file=disp_files[i], samples=samples)
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

# Cell
def odin_noise(x_test, eps, weight_file):
    x_test_inp = tf.reshape(x_test, x_test.shape)
    model = PointCNNTScaled(3, 12, 100, 0.5)
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
    del model
    return tf.reshape(x_test_new, x_test_new.shape).numpy()

x_test_new = odin_noise(x_test, 0.0012, prefix + "weights/cifar10/cifar10_densenet_mean.npy")
x_test_f_new = odin_noise(x_test_c, 0.0012, prefix + "weights/cifar10/cifar10_densenet_mean.npy")

grader = EvalVGG10(x_test_new, x_test_f_new, weight_file=prefix + "weights/cifar10/cifar10_densenet_mean.npy")
scores = {"FPR at 95 % TPR": None, "AUROC": None, "AUPR-In": None, "Detection Error": None}
scores["FPR at 95 % TPR"] = round(grader.fpr_at_95_tpr(), 2)
scores["AUROC"] = round(grader.auroc(), 2)
scores["AUPR-In"] = round(grader.aupr_in(), 2)
scores["Detection Error"] = round(grader.detection(), 2)
print(scores)

