# -*- coding: utf-8 -*-

# Cell 1
import tensorflow as tf, tensorflow_datasets as tfds, numpy as np
import matplotlib.pyplot as plt
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

# Cell 3
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
        term_w = tf.math.multiply(eps_w, tf.math.softplus(self.kernel_rho))
        kernel = tf.math.add(self.kernel_mu, term_w)
        eps_b_shape = self.bias_mu.shape
        eps_b = tf.random.normal(eps_b_shape, 0, 1, dtype=tf.float32)
        term_b = tf.math.multiply(eps_b, tf.math.softplus(self.bias_rho))
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
        term_w = tf.math.multiply(eps_w, tf.math.softplus(self.kernel_rho))
        kernel = tf.math.add(self.kernel_mu, term_w)
        eps_b_shape = self.bias_mu.shape
        eps_b = tf.random.normal(eps_b_shape, 0, 1, dtype=tf.float32)
        term_b = tf.math.multiply(eps_b, tf.math.softplus(self.bias_rho))
        bias = tf.math.add(self.bias_mu, term_b)
        return kernel, bias
    
    def call(self, inputs, avoid_sampling=False):
        k, b = self._reparametrize(avoid_sampling)
        self.kernel, self.bias = k + 0, b + 0
        return super(PerturbedDense, self).call(inputs)

# Cell 4
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

# Cell 5
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

# Cell 6
mean, std = 0, 255.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
y_train, y_test = np.squeeze(y_train), np.squeeze(y_test)
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
x_train, x_test = (x_train - mean) / std, (x_test - mean) / std
x_train_mean = np.mean(x_train, axis=0)
x_train -= x_train_mean
x_test -= x_train_mean

# Cell 7
prefix = ""

def transfer_weights(model):
    model.transferstate(np.load(prefix + "weights/cifar10/cifar10resnet20_mean.npy", allow_pickle=True))

def load_config(file_path, model):
    train_config = None
    infile = open(file_path, 'rb')
    train_config = pickle.load(infile)
    model.setstate(np.load(file_path + ".npy", allow_pickle=True))
    return train_config

def save_config(train_config, model, file_path):
    outfile = open(file_path, 'wb')
    pickle.dump(train_config, outfile, pickle.HIGHEST_PROTOCOL)
    np.save(file_path, model.getstate())

def train(train_images, train_labels, alpha, iterations, batch_size, checkpoint=None, **kwargs):
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)
    curr_date_time = str(datetime.now()).replace(' ', '_').replace(':', '_').replace('-', '_').replace('.', '_')
    size = train_images.shape[0]
    model = kwargs.get('model_type', PerturbedNN)(3, 16, 10, tf.optimizers.RMSprop(alpha))
    model.build((None,) + train_images.shape[1:])
    train_config = None
    weight_file = None
    if checkpoint is None:
        train_config = {'learning_rates': [alpha], 'batch_sizes': [batch_size], 
                        'iterations': 0, 'times_per_batch': [], 'dataset_sizes': [size],
                        'total_losses': []}
        weight_file = "weights/cifar10/" + curr_date_time
        transfer_weights(model)
    else:
        weight_file = checkpoint
        train_config = load_config(prefix + checkpoint, model)
        train_config['learning_rates'].append(alpha)
        train_config['batch_sizes'].append(batch_size)
        train_config['dataset_sizes'].append(size)
        print("Configuration History")
        print("=====================")
        print("Learning rates: ", train_config['learning_rates'])
        print("Batch sizes: ", train_config['batch_sizes'])
        print("Iterations: ", train_config['iterations'])
        print("Dataset sizes", train_config['dataset_sizes'])
        print("Training times per batch", train_config['times_per_batch'])
        print("\n")

    try:
        nll = None
        for itr, (batch_x, batch_y) in enumerate(train_data.take(iterations), 1):
            M = size//batch_size
            total_loss, cse, entropy, sigma_penalty, time_per_batch = model.fit(batch_x, batch_y, **kwargs)
            print("Total Loss: ", total_loss.numpy())
            print("Cross Entropy: ", cse.numpy())
            print("MOU: ", entropy.numpy())
            print("L2 Regularisation on Standard Deviations: ", sigma_penalty.numpy())
            print("Iteration %s over"%(itr))
            train_config['times_per_batch'].append(time_per_batch)
            train_config['total_losses'].append(total_loss.numpy())
            train_config['iterations'] += 1
        
        save_config(train_config, model, prefix + weight_file)
        return weight_file
    except KeyboardInterrupt:
        save_config(train_config, model, prefix + weight_file)
        return weight_file

def test(test_images, test_labels, samples, weight_file, batch_size=100, **kwargs):
    test_data = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    test_data = test_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)
    model = kwargs.get('model_type', PerturbedNN)(3, 16, 10)
    model.build((None,) + test_images.shape[1:])
    test_config = load_config(prefix + weight_file, model)
    total_score = 0
    batches = 0
    for _ in range(samples):
        batches = 0
        for _, (batch_x, batch_y) in enumerate(test_data.take(test_images.shape[0]//batch_size), 1):
            outputs = model.call(batch_x)
            correct_prediction = tf.equal(tf.argmax(outputs, 1), tf.cast(batch_y, tf.int64))
            total_score += tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)*100
            batches += 1
    return total_score/(samples*batches)

# Cell 8
train(x_train, y_train, 0.01, 400, 128, checkpoint="weights/cifar10/2020_10_15_14_52_27_357637", c=1, b=1e11, a=1, model_type=PerturbedNN)

x_cifar10_c_path = "/content/drive/MyDrive/bnn/data/cifar10_corrupted/CIFAR-10-C/zoom_blur.npy"
x_cifar10_c = np.load(x_cifar10_c_path, allow_pickle=True)
x_cifar10_c = np.array(x_cifar10_c, np.float32)
mean_te, std_te = np.mean(x_cifar10_c), np.std(x_cifar10_c)
x_cifar10_c = (x_cifar10_c - mean_te) / std_te
y_cifar10_c = np.load("/content/drive/MyDrive/bnn/data/cifar10_corrupted/CIFAR-10-C/labels.npy", allow_pickle=True)

# Cell 9
test(x_cifar10_c, y_cifar10_c, 1, "weights/cifar10/2020_10_15_14_52_27_357637", batch_size=1000)
