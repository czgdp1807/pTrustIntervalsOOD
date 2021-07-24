# -*- coding: utf-8 -*-

# Cell 1
import tensorflow as tf, tensorflow_datasets as tfds, numpy as np
import matplotlib.pyplot as plt
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
minrho, maxrho = -7, -6
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
        avoid_sampling=True
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
        avoid_sampling=True
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

# Cell 5
mean, std = 120.707, 64.15
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
y_train, y_test = np.squeeze(y_train), np.squeeze(y_test)
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
x_train, x_test = (x_train - mean) / std, (x_test - mean) / std

# Cell 6
(x_train_f, y_train_f), (x_test_f, y_test_f) = tf.keras.datasets.cifar100.load_data()
y_train_f, y_test_f = np.squeeze(y_train_f), np.squeeze(y_test_f)
x_train_f, x_test_f = np.array(x_train_f, np.float32), np.array(x_test_f, np.float32)
mean_tr, std_tr, mean_te, std_te = np.mean(x_train_f), np.std(x_train_f), np.mean(x_test_f), np.std(x_test_f)
x_train_f, x_test_f = (x_train_f - mean_tr) / std_tr, (x_test_f - mean_te) / std_te

# Cell 7
prefix = ""

def transfer_weights(model):
    model.transferstate(np.load(prefix + "weights/cifar10/cifar10vgg_mean.npy", allow_pickle=True))

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
    model = kwargs.get('model_type', PerturbedNN)(tf.optimizers.RMSprop(alpha))
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
            print("Entropy: ", entropy.numpy())
            print("L2 Regularisation on Standard Deviations: ", sigma_penalty.numpy())
            print("Iteration %s over"%(itr))
            train_config['times_per_batch'].append(time_per_batch)
            train_config['total_losses'].append(total_loss.numpy())
            train_config['iterations'] += 1
            if cse.numpy()*100 >= 1:
                break
        
        save_config(train_config, model, prefix + weight_file)
        return weight_file
    except KeyboardInterrupt:
        save_config(train_config, model, prefix + weight_file)
        return weight_file

def test(test_images, test_labels, samples, weight_file, batch_size=100, **kwargs):
    test_data = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    test_data = test_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)
    print(test_labels[0])
    model = kwargs.get('model_type', PerturbedNN)()
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
train(x_train, y_train, 0.1, 1, 64, checkpoint=None, c=1, b=1e13, a=1)

x_cifar10_c_path = "/content/drive/MyDrive/bnn/data/cifar10_corrupted/CIFAR-10-C/zoom_blur.npy"
x_cifar10_c = np.load(x_cifar10_c_path, allow_pickle=True)
x_cifar10_c = np.array(x_cifar10_c, np.float32)
mean_te, std_te = np.mean(x_cifar10_c), np.std(x_cifar10_c)
x_cifar10_c = (x_cifar10_c - mean_te) / std_te
y_cifar10_c = np.load("/content/drive/MyDrive/bnn/data/cifar10_corrupted/CIFAR-10-C/labels.npy", allow_pickle=True)

# Cell 9
test(x_cifar10_c, y_cifar10_c, 1, "weights/cifar10/2020_09_29_10_40_38_794710", batch_size=1000)
