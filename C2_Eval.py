# -*- coding: utf-8 -*-

# Cell 1
import tensorflow as tf, tensorflow_datasets as tfds, numpy as np
import math
from datetime import datetime
import time, pickle

# Cell 2
class StandardDeviationInit(tf.keras.initializers.Initializer):

    def __init__(self, minval=0, maxval=1):
        self.minval = minval
        self.maxval = maxval
    
    def __call__(self, shape, dtype=None):
        return tf.random.uniform(shape, self.minval, self.maxval, dtype=dtype)

# Cell 3
class PointConv2D(tf.keras.layers.Conv2D):
    
    def __init__(self, filters, kernel_size, stride, padding, activation, dtype, **kwargs):
        super(PointConv2D, self).__init__(filters, kernel_size, stride, activation=activation, 
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
        
class PerturbedConv2D(tf.keras.layers.Conv2D):
    
    def __init__(self, filters, kernel_size, stride, padding, activation=None, dtype=tf.float32, **kwargs):
        super(PerturbedConv2D, self).__init__(filters, kernel_size, stride, activation=activation, 
                                          dtype=dtype, padding=padding, use_bias=True, **kwargs)
    
    def build(self, input_shape):
        super(PerturbedConv2D, self).build(input_shape)
        self.kernel_mu = self.add_weight(name='kernel_mu', shape=self.kernel.shape,
                                         initializer=self.kernel_initializer,
                                         trainable=True,
                                         dtype=self.dtype)
        self.kernel_rho = self.add_weight(name='kernel_rho', shape=self.kernel.shape,
                                          initializer=StandardDeviationInit(-8, -9),
                                          trainable=True,
                                          dtype=self.dtype)
        self.bias_mu = self.add_weight(name='bias_mu', shape=self.bias.shape,
                                         initializer=self.bias_initializer,
                                         trainable=True,
                                         dtype=self.dtype)
        self.bias_rho = self.add_weight(name='bias_rho', shape=self.bias.shape,
                                          initializer=StandardDeviationInit(-8, -9),
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
    
    def build(self, input_shape):
        super(PerturbedDense, self).build(input_shape)
        self.kernel_mu = self.add_weight(name='kernel_mu', shape=self.kernel.shape,
                                         initializer=self.kernel_initializer,
                                         trainable=True,
                                         dtype=self.dtype)
        self.kernel_rho = self.add_weight(name='kernel_rho', shape=self.kernel.shape,
                                          initializer=StandardDeviationInit(-8, -9),
                                          trainable=True,
                                          dtype=self.dtype)
        self.bias_mu = self.add_weight(name='bias_mu', shape=self.bias.shape,
                                         initializer=self.bias_initializer,
                                         trainable=True,
                                         dtype=self.dtype)
        self.bias_rho = self.add_weight(name='bias_rho', shape=self.bias.shape,
                                          initializer=StandardDeviationInit(-8, -9),
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

# Cell 4
mean, std = 120.707, 64.15
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
x_train, x_test = (x_train - mean) / std, (x_test - mean) / std

# Cell 5
(x_train_f, y_train_f), (x_test_f, y_test_f) = tf.keras.datasets.cifar100.load_data()
x_train_f, x_test_f = np.array(x_train_f, np.float32), np.array(x_test_f, np.float32)
mean_tr, std_tr, mean_te, std_te = np.mean(x_train_f), np.std(x_train_f), np.mean(x_test_f), np.std(x_test_f)
x_train_f, x_test_f = (x_train_f - mean_tr) / std_tr, (x_test_f - mean_te) / std_te

# Cell 6
class PerturbedNN(tf.keras.Model):

    def __init__(self, optimizer=None):
        super(PerturbedNN, self).__init__()
        self.Conv_1 = PerturbedConv2D(64, (3, 3), 1, 'same', tf.nn.relu, tf.float32)
        self.BatchNorm_1 = tf.keras.layers.BatchNormalization()
        self.Conv_2 = PerturbedConv2D(64, (3, 3), 1, 'same', tf.nn.relu, tf.float32)
        self.BatchNorm_2 = tf.keras.layers.BatchNormalization()
        self.MaxPool_1 = tf.keras.layers.MaxPooling2D((2, 2))
        self.Conv_3 = PerturbedConv2D(128, (3, 3), 1, 'same', tf.nn.relu, tf.float32)
        self.BatchNorm_3 = tf.keras.layers.BatchNormalization()
        self.Conv_4 = PerturbedConv2D(128, (3, 3), 1, 'same', tf.nn.relu, tf.float32)
        self.BatchNorm_4 = tf.keras.layers.BatchNormalization()
        self.MaxPool_2 = tf.keras.layers.MaxPooling2D((2, 2))
        self.Conv_5 = PerturbedConv2D(256, (3, 3), 1, 'same', tf.nn.relu, tf.float32)
        self.BatchNorm_5 = tf.keras.layers.BatchNormalization()
        self.Conv_6 = PerturbedConv2D(256, (3, 3), 1, 'same', tf.nn.relu, tf.float32)
        self.BatchNorm_6 = tf.keras.layers.BatchNormalization()
        self.Conv_7 = PerturbedConv2D(256, (3, 3), 1, 'same', tf.nn.relu, tf.float32)
        self.BatchNorm_7 = tf.keras.layers.BatchNormalization()
        self.MaxPool_3 = tf.keras.layers.MaxPooling2D((2, 2))
        self.Conv_8 = PerturbedConv2D(512, (3, 3), 1, 'same', tf.nn.relu, tf.float32)
        self.BatchNorm_8 = tf.keras.layers.BatchNormalization()
        self.Conv_9 = PerturbedConv2D(512, (3, 3), 1, 'same', tf.nn.relu, tf.float32)
        self.BatchNorm_9 = tf.keras.layers.BatchNormalization()
        self.Conv_10 = PerturbedConv2D(512, (3, 3), 1, 'same', tf.nn.relu, tf.float32)
        self.BatchNorm_10 = tf.keras.layers.BatchNormalization()
        self.MaxPool_4 = tf.keras.layers.MaxPooling2D((2, 2))
        self.Conv_11 = PerturbedConv2D(512, (3, 3), 1, 'same', tf.nn.relu, tf.float32)
        self.BatchNorm_11 = tf.keras.layers.BatchNormalization()
        self.Conv_12 = PerturbedConv2D(512, (3, 3), 1, 'same', tf.nn.relu, tf.float32)
        self.BatchNorm_12 = tf.keras.layers.BatchNormalization()
        self.Conv_13 = PerturbedConv2D(512, (3, 3), 1, 'same', tf.nn.relu, tf.float32)
        self.BatchNorm_13 = tf.keras.layers.BatchNormalization()
        self.MaxPool_5 = tf.keras.layers.MaxPooling2D((2, 2))
        self.Flatten_1 = tf.keras.layers.Flatten()
        self.Dense_1 = PerturbedDense(512, tf.nn.relu)
        self.BatchNorm_14 = tf.keras.layers.BatchNormalization()
        self.Dense_2 = PerturbedDense(10)
        self.Layers = [self.Conv_1, self.BatchNorm_1, self.Conv_2, self.BatchNorm_2, self.MaxPool_1, 
                        self.Conv_3, self.BatchNorm_3, self.Conv_4, self.BatchNorm_4, self.MaxPool_2,
                        self.Conv_5, self.BatchNorm_5, self.Conv_6, self.BatchNorm_6, self.Conv_7, self.BatchNorm_7, self.MaxPool_3,
                        self.Conv_8, self.BatchNorm_8, self.Conv_9, self.BatchNorm_9, self.Conv_10, self.BatchNorm_10, self.MaxPool_4,
                        self.Conv_11, self.BatchNorm_11, self.Conv_12, self.BatchNorm_12, self.Conv_13, self.BatchNorm_13, self.MaxPool_5,
                        self.Flatten_1, self.Dense_1, self.BatchNorm_14, self.Dense_2]
        self.TransferLayers = [self.Conv_1, self.BatchNorm_1, self.Conv_2, self.BatchNorm_2, 
                                self.Conv_3, self.BatchNorm_3, self.Conv_4, self.BatchNorm_4,
                                self.Conv_5, self.BatchNorm_5, self.Conv_6, self.BatchNorm_6, self.Conv_7, self.BatchNorm_7,
                                self.Conv_8, self.BatchNorm_8, self.Conv_9, self.BatchNorm_9, self.Conv_10, self.BatchNorm_10,
                                self.Conv_11, self.BatchNorm_11, self.Conv_12, self.BatchNorm_12, self.Conv_13, self.BatchNorm_13,
                                self.Dense_1, self.BatchNorm_14, self.Dense_2]
        self.TrainableLayers = [self.Conv_1, self.Conv_2, self.Conv_3, self.Conv_4,
                                self.Conv_5, self.Conv_6, self.Conv_7, self.Conv_8, 
                                self.Conv_9, self.Conv_10, self.Conv_11, self.Conv_12, 
                                self.Conv_13, self.Dense_1, self.Dense_2]
        self.optimizer = optimizer
    
    def build(self, input_shape):
        for layer in self.Layers:
            layer.build(input_shape)
            input_shape = layer.compute_output_shape(input_shape)
    
    def call(self, inputs, is_training=False):
        for layer in self.Layers:
            inputs = layer.call(inputs)
        if not is_training:
            inputs = tf.nn.softmax(inputs)
        return inputs
    
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
        for layer in self.Layers:
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
        itr = iter(self.Layers)
        i = 0
        while i < kernels.shape[0]:
            try:
                layer = next(itr)
            except StopIteration:
                break
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
        itr = iter(self.TrainableLayers)
        while i < kernels.shape[0]:
            layer = next(itr)
            layer.kernel_rho.assign(tf.convert_to_tensor(kernels[i], tf.float32))
            layer.bias_rho.assign(tf.convert_to_tensor(kernels[i+1], tf.float32))
            i += 2
    
    def transferstate(self, kernel_mu):
        itr = iter(self.TransferLayers)
        i = 0
        while i < kernel_mu.shape[0]:
            layer = next(itr)
            if isinstance(layer, (PerturbedConv2D, PerturbedDense)):
                layer.kernel_mu.assign(tf.convert_to_tensor(kernel_mu[i], tf.float32))
                layer.bias_mu.assign(tf.convert_to_tensor(kernel_mu[i+1], tf.float32))
                i += 2
            elif isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.gamma.assign(tf.convert_to_tensor(kernel_mu[i], tf.float32))
                layer.beta.assign(tf.convert_to_tensor(kernel_mu[i+1], tf.float32))
                layer.moving_mean.assign(tf.convert_to_tensor(kernel_mu[i+2], tf.float32))
                layer.moving_variance.assign(tf.convert_to_tensor(kernel_mu[i+3], tf.float32))
                i += 4

# Cell 4
class PointCNN(tf.keras.Model):

    def __init__(self, optimizer=None):
        super(PointCNN, self).__init__()
        self.Conv_1 = PointConv2D(64, (3, 3), 1, 'same', tf.nn.relu, tf.float32)
        self.BatchNorm_1 = tf.keras.layers.BatchNormalization()
        self.Conv_2 = PointConv2D(64, (3, 3), 1, 'same', tf.nn.relu, tf.float32)
        self.BatchNorm_2 = tf.keras.layers.BatchNormalization()
        self.MaxPool_1 = tf.keras.layers.MaxPooling2D((2, 2))
        self.Conv_3 = PointConv2D(128, (3, 3), 1, 'same', tf.nn.relu, tf.float32)
        self.BatchNorm_3 = tf.keras.layers.BatchNormalization()
        self.Conv_4 = PointConv2D(128, (3, 3), 1, 'same', tf.nn.relu, tf.float32)
        self.BatchNorm_4 = tf.keras.layers.BatchNormalization()
        self.MaxPool_2 = tf.keras.layers.MaxPooling2D((2, 2))
        self.Conv_5 = PointConv2D(256, (3, 3), 1, 'same', tf.nn.relu, tf.float32)
        self.BatchNorm_5 = tf.keras.layers.BatchNormalization()
        self.Conv_6 = PointConv2D(256, (3, 3), 1, 'same', tf.nn.relu, tf.float32)
        self.BatchNorm_6 = tf.keras.layers.BatchNormalization()
        self.Conv_7 = PointConv2D(256, (3, 3), 1, 'same', tf.nn.relu, tf.float32)
        self.BatchNorm_7 = tf.keras.layers.BatchNormalization()
        self.MaxPool_3 = tf.keras.layers.MaxPooling2D((2, 2))
        self.Conv_8 = PointConv2D(512, (3, 3), 1, 'same', tf.nn.relu, tf.float32)
        self.BatchNorm_8 = tf.keras.layers.BatchNormalization()
        self.Conv_9 = PointConv2D(512, (3, 3), 1, 'same', tf.nn.relu, tf.float32)
        self.BatchNorm_9 = tf.keras.layers.BatchNormalization()
        self.Conv_10 = PointConv2D(512, (3, 3), 1, 'same', tf.nn.relu, tf.float32)
        self.BatchNorm_10 = tf.keras.layers.BatchNormalization()
        self.MaxPool_4 = tf.keras.layers.MaxPooling2D((2, 2))
        self.Conv_11 = PointConv2D(512, (3, 3), 1, 'same', tf.nn.relu, tf.float32)
        self.BatchNorm_11 = tf.keras.layers.BatchNormalization()
        self.Conv_12 = PointConv2D(512, (3, 3), 1, 'same', tf.nn.relu, tf.float32)
        self.BatchNorm_12 = tf.keras.layers.BatchNormalization()
        self.Conv_13 = PointConv2D(512, (3, 3), 1, 'same', tf.nn.relu, tf.float32)
        self.BatchNorm_13 = tf.keras.layers.BatchNormalization()
        self.MaxPool_5 = tf.keras.layers.MaxPooling2D((2, 2))
        self.Flatten_1 = tf.keras.layers.Flatten()
        self.Dense_1 = PointDense(512, tf.nn.relu)
        self.BatchNorm_14 = tf.keras.layers.BatchNormalization()
        self.Dense_2 = PointDense(10)
        self.Layers = [self.Conv_1, self.BatchNorm_1, self.Conv_2, self.BatchNorm_2, self.MaxPool_1, 
                        self.Conv_3, self.BatchNorm_3, self.Conv_4, self.BatchNorm_4, self.MaxPool_2,
                        self.Conv_5, self.BatchNorm_5, self.Conv_6, self.BatchNorm_6, self.Conv_7, self.BatchNorm_7, self.MaxPool_3,
                        self.Conv_8, self.BatchNorm_8, self.Conv_9, self.BatchNorm_9, self.Conv_10, self.BatchNorm_10, self.MaxPool_4,
                        self.Conv_11, self.BatchNorm_11, self.Conv_12, self.BatchNorm_12, self.Conv_13, self.BatchNorm_13, self.MaxPool_5,
                        self.Flatten_1, self.Dense_1, self.BatchNorm_14, self.Dense_2]
        self.TrainableLayers = [self.Conv_9, self.Conv_10, self.Conv_11, self.Conv_12, 
                                self.Conv_13, self.Dense_1, self.Dense_2]
        self.optimizer = optimizer
    
    def build(self, input_shape):
        for layer in self.Layers:
            layer.build(input_shape)
            input_shape = layer.compute_output_shape(input_shape)
    
    def call(self, inputs, is_training=False):
        for layer in self.Layers:
            inputs = layer.call(inputs)
        if not is_training:
            inputs = tf.nn.softmax(inputs)
        return inputs
    
    def mou(self, output_tensor):
        un = tf.reduce_mean(tf.reduce_sum(tf.math.reduce_std(output_tensor, 0)**2, 1))
        return un

    def get_loss(self, inputs, targets, **kwargs):
        targets = tf.cast(targets, tf.int64)
        output = self.call(inputs, is_training=True)
        cse = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(targets, output))
        
        return cse
    
    def compute_gradients(self, inputs, targets, **kwargs):
        _vars = []
        with tf.GradientTape(persistent=True) as tape:
            for layer in self.TrainableLayers:
                _vars.append(layer.kernel)
            tape.watch(_vars)
            F = self.get_loss(inputs, targets, **kwargs)

        dF = tape.gradient(F, _vars)
        
        return dF, F, _vars
    
    def fit(self, inputs, targets, **kwargs):
        start_time = time.time()
        grads, F, vars = self.compute_gradients(inputs, targets, **kwargs)
        self.optimizer.apply_gradients(zip(grads, vars))
        end_time = time.time()
        return F, end_time - start_time

    def getstate(self):
        np_kernel_mu = []
        for layer in self.Layers:
            if isinstance(layer, (PointConv2D, PointDense)):
                np_kernel_mu.append(layer.kernel.numpy())
                np_kernel_mu.append(layer.bias.numpy())
            elif isinstance(layer, tf.keras.layers.BatchNormalization):
                np_kernel_mu.append(layer.gamma.numpy())
                np_kernel_mu.append(layer.beta.numpy())
                np_kernel_mu.append(layer.moving_mean.numpy())
                np_kernel_mu.append(layer.moving_variance.numpy())
        return np.asarray(np_kernel_mu)

    def setstate(self, kernels):
        itr = iter(self.Layers)
        i = 0
        while i < kernels.shape[0]:
            try:
                layer = next(itr)
            except StopIteration:
                break
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
    
    def transferstate(self, kernels):
        itr = iter(self.Layers)
        i = 0
        while i < kernels.shape[0]:
            try:
                layer = next(itr)
            except StopIteration:
                break
            if isinstance(layer, (PointConv2D, PointDense)):
                if layer not in self.TrainableLayers:
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
        for layer in self.Layers:
            inputs = layer.call(inputs)
        if not is_training:
            inputs = tf.nn.softmax(inputs/T)
        return inputs

# Cell 7
x_test_c, y_test_c = tfds.as_numpy(tfds.load(name="cmaterdb", split='test', batch_size=-1, as_supervised=True))
x_test_c = np.array(x_test_c, np.float32)
mean_te, std_te = np.mean(x_test_c), np.std(x_test_c)
x_test_c = (x_test_c - mean_te) / std_te

# Cell 8
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
            self.model = PointCNN()
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
            # if tpr >= 0.94 or tpr <= 0.0001:
            #     print(delta, tpr, fpr)
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


class EvalVGG10MHB(EvalVGG10):

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
        self.start = -120.
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
            self.model = PointCNN()
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
            self.model = PerturbedNN()
            self.model.build((None,) + x_in.shape[1:])
            load_config(weight_file, self.model)
            samples = kwargs.get('samples', 10)
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
        self.inc = 0.001
        self.start = -2
        self.end = 0.

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
            self.model = PerturbedNN()
            self.model.build((None,) + x_in.shape[1:])
            load_config(weight_file, self.model)
            samples = kwargs.get('samples', 10)
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
        self.inc = 1000
        self.start = 520
        self.end = 145940

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

# Cell 10
num_samples, min_samples = 3, 2
disp_files = [] # ["./drive/My Drive/bnn/results/cifar10/2021_02_05_18_12_46_085123.npy"] # ["./drive/My Drive/bnn/results/cifar10/2021_02_05_17_43_39_227505.npy"] # ["./drive/My Drive/bnn/results/cifar10/2021_02_05_17_39_47_340834.npy"] # ["./drive/My Drive/bnn/results/cifar10/2021_02_05_17_33_01_522325.npy"] # ["./drive/My Drive/bnn/results/cifar10/2021_02_05_17_22_42_332220.npy"] # ["./drive/My Drive/bnn/results/cifar10/2021_02_05_17_18_15_639676.npy"] # ["./drive/My Drive/bnn/results/cifar10/2021_02_03_15_01_23_996662.npy"] # ["./drive/My Drive/bnn/results/cifar10/2021_02_03_14_47_36_072811.npy"] # ["/content/drive/My Drive/bnn/results/cifar10/2020_09_30_16_03_12_296354.npy"] # ["/content/drive/My Drive/bnn/results/cifar10/2020_09_30_09_40_59_660654.npy"] # ["/content/drive/My Drive/bnn/results/cifar10/2020_09_30_09_22_19_320812.npy"] # ["/content/drive/My Drive/bnn/results/cifar10/2020_09_30_07_54_46_588716.npy"] #["/content/drive/My Drive/bnn/results/cifar10/2020_09_29_14_53_11_820554.npy"] # ["/content/drive/My Drive/bnn/results/cifar10/2020_09_29_14_48_21_652316.npy"] # ["/content/drive/My Drive/bnn/results/cifar10/2020_09_29_14_34_01_301732.npy"] # ["/content/drive/My Drive/bnn/results/cifar10/2020_09_29_13_00_23_612488.npy"] # ["/content/drive/My Drive/bnn/results/cifar10/2020_09_29_11_14_36_502672.npy"]
disp_files = disp_files + [None for i in range(num_samples - min_samples - len(disp_files))]
scores = {"FPR at 95 % TPR": 0, "AUROC": 0, "AUPR-In": 0, "Detection Error": 0}
i = 0
valid_samples = 0
for samples in range(min_samples, num_samples):
    grader = EvalVGG10IndexOfDispersion(x_train, x_cifar10_c, weight_file=prefix + "weights/cifar10/2020_09_29_10_40_38_794710", entropy_file=disp_files[i], samples=samples, measure='M2')
#     try:
#         scores["FPR at 95 % TPR"] += round(grader.fpr_at_95_tpr(), 2)
#         scores["AUROC"] += round(grader.auroc(), 2)
#         scores["AUPR-In"] += round(grader.aupr_in(), 2)
#         scores["Detection Error"] += round(grader.detection(), 2)
#         valid_samples += 1
#     except ZeroDivisionError:
#         continue
#     i += 1
# for key in scores:
#     scores[key] = scores[key]/valid_samples
# print(scores)

class EvalEnsembleVGG10(EvalVGG10):

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
                model = PointCNN()
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
    prefix + "weights/cifar10/2021_01_30_08_16_06_947676",
    prefix + "weights/cifar10/2021_01_30_08_18_05_805711",
    # prefix + "weights/mnist/2021_01_30_08_40_43_545708",
    # prefix + "weights/mnist/2021_01_30_08_41_50_115262",
    # prefix + "weights/mnist/2021_01_30_08_42_23_109625"
]
grader = EvalEnsembleVGG10(x_test, x_test_c, 
    weight_files=weight_files, 
    entropy_file="./drive/My Drive/bnn/results/cifar10/2021_01_31_14_04_08_003494.npy")
scores = {"FPR at 95 % TPR": None, "AUROC": None, "AUPR-In": None, "Detection Error": None}
scores["FPR at 95 % TPR"] = grader.fpr_at_95_tpr()
scores["AUROC"] = grader.auroc()
scores["AUPR-In"] = grader.aupr_in()
scores["Detection Error"] = grader.detection()
print(scores)

# Cell 10
num_samples, min_samples = 3, 2
disp_files = ["./drive/My Drive/bnn/results/cifar10/2021_01_29_14_43_04_496076.npy"]# ["./drive/My Drive/bnn/results/cifar10/2021_01_29_14_40_00_948858.npy"] # ["/content/drive/My Drive/bnn/results/cifar10/2020_09_30_16_03_12_296354.npy"] # ["/content/drive/My Drive/bnn/results/cifar10/2020_09_30_09_40_59_660654.npy"] # ["/content/drive/My Drive/bnn/results/cifar10/2020_09_30_09_22_19_320812.npy"] # ["/content/drive/My Drive/bnn/results/cifar10/2020_09_30_07_54_46_588716.npy"] #["/content/drive/My Drive/bnn/results/cifar10/2020_09_29_14_53_11_820554.npy"] # ["/content/drive/My Drive/bnn/results/cifar10/2020_09_29_14_48_21_652316.npy"] # ["/content/drive/My Drive/bnn/results/cifar10/2020_09_29_14_34_01_301732.npy"] # ["/content/drive/My Drive/bnn/results/cifar10/2020_09_29_13_00_23_612488.npy"] # ["/content/drive/My Drive/bnn/results/cifar10/2020_09_29_11_14_36_502672.npy"]
disp_files = disp_files + [None for i in range(num_samples - min_samples - len(disp_files))]
scores = {"FPR at 95 % TPR": 0, "AUROC": 0, "AUPR-In": 0, "Detection Error": 0}
i = 0
valid_samples = 0
for samples in range(min_samples, num_samples):
    grader = EvalBayesAdapter(x_test, x_test_f, weight_file=prefix + "weights/cifar10/bayes_adapter_2021_01_29_11_33_53_031602", entropy_file=disp_files[i], samples=samples)
    try:
        scores["FPR at 95 % TPR"] += grader.fpr_at_95_tpr()
        scores["AUROC"] += grader.auroc()
        scores["AUPR-In"] += grader.aupr_in()
        scores["Detection Error"] += grader.detection()
        valid_samples += 1
    except ZeroDivisionError:
        continue
    i += 1
for key in scores:
    scores[key] = scores[key]/valid_samples
print(scores)

# Cell 9
grader = EvalVGG10MHB(x_test, x_test_f, weight_file=prefix + "weights/cifar10/cifar10vgg_mean.npy")
scores = {"FPR at 95 % TPR": None, "AUROC": None, "AUPR-In": None, "Detection Error": None}
scores["FPR at 95 % TPR"] = grader.fpr_at_95_tpr()
scores["AUROC"] = grader.auroc()
scores["AUPR-In"] = grader.aupr_in()
scores["Detection Error"] = grader.detection()
print(scores)

# Cell 11
grader = EvalVGG10(x_test, x_test_f, weight_file=prefix + "weights/cifar10/cifar10vgg_mean.npy")
scores = {"FPR at 95 % TPR": None, "AUROC": None, "AUPR-In": None, "Detection Error": None}
scores["FPR at 95 % TPR"] = grader.fpr_at_95_tpr()
scores["AUROC"] = grader.auroc()
scores["AUPR-In"] = grader.aupr_in()
scores["Detection Error"] = grader.detection()
print(scores)

# Cell 12
def odin_noise(x_test, eps, weight_file):
    x_test_inp = tf.reshape(x_test, x_test.shape)
    model = PointCNNTScaled()
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

x_test_new = odin_noise(x_test, 0.0012, prefix + "weights/cifar10/cifar10vgg_mean.npy")
x_test_f_new = odin_noise(x_test_f, 0.0012, prefix + "weights/cifar10/cifar10vgg_mean.npy")

grader = EvalVGG10(x_test_new, x_test_f_new, weight_file=prefix + "weights/cifar10/cifar10vgg_mean.npy")
scores = {"FPR at 95 % TPR": None, "AUROC": None, "AUPR-In": None, "Detection Error": None}
scores["FPR at 95 % TPR"] = grader.fpr_at_95_tpr()
scores["AUROC"] = grader.auroc()
scores["AUPR-In"] = grader.aupr_in()
scores["Detection Error"] = grader.detection()
print(scores)
