# -*- coding: utf-8 -*-

# Cell 1
import tensorflow as tf, numpy as np
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
class PerturbedConv2D(tf.keras.layers.Conv2D):
    
    def __init__(self, filters, kernel_size, stride, padding, activation=None, dtype=tf.float32, **kwargs):
        super(PerturbedConv2D, self).__init__(filters, kernel_size, stride, activation=activation, 
                                          dtype=dtype, padding=padding, use_bias=False, **kwargs)
    
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
    
    def _reparametrize(self, avoid_sampling=False):
        if avoid_sampling:
            return self.kernel_mu
        eps_w_shape = self.kernel_mu.shape
        eps_w = tf.random.normal(eps_w_shape, 0, 0.01, dtype=tf.float32)
        # term_w = tf.math.multiply(eps_w, tf.math.log(tf.math.add(
        #                           tf.math.exp(tf.clip_by_value(self.kernel_rho, -87.315, 88.722)),
        #                           tf.constant(1., shape=eps_w_shape, dtype=tf.float32))))
        term_w = eps_w
        return tf.math.add(self.kernel_mu, term_w)
    
    def call(self, inputs, avoid_sampling=False):
        self.kernel = self._reparametrize(avoid_sampling) + 0
        return super(PerturbedConv2D, self).call(inputs)

class PerturbedDense(tf.keras.layers.Dense):

    def __init__(self, units, activation=None, **kwargs):
        super(PerturbedDense, self).__init__(units, activation=activation, use_bias=False, **kwargs)
    
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
    
    def _reparametrize(self, avoid_sampling=False):
        if avoid_sampling:
            return self.kernel_mu
        eps_w_shape = self.kernel_mu.shape
        eps_w = tf.random.normal(eps_w_shape, 0, 0.01, dtype=tf.float32)
        # term_w = tf.math.multiply(eps_w, tf.math.log(tf.math.add(
        #                           tf.math.exp(tf.clip_by_value(self.kernel_rho, -87.315, 88.722)),
        #                           tf.constant(1., shape=eps_w_shape, dtype=tf.float32))))
        term_w = eps_w
        return tf.math.add(self.kernel_mu, term_w)
    
    def call(self, inputs, avoid_sampling=False):
        self.kernel = self._reparametrize(avoid_sampling) + 0
        return super(PerturbedDense, self).call(inputs)

class PointConv2D(tf.keras.layers.Conv2D):
    
    def __init__(self, filters, kernel_size, stride, padding, activation, dtype, **kwargs):
        super(PointConv2D, self).__init__(filters, kernel_size, stride, activation=activation, 
                                          dtype=dtype, padding=padding, use_bias=False, **kwargs)
    
    def build(self, input_shape):
        super(PointConv2D, self).build(input_shape)
    
    def call(self, inputs):
        return super(PointConv2D, self).call(inputs)

class PointDense(tf.keras.layers.Dense):

    def __init__(self, units, activation=None, **kwargs):
        super(PointDense, self).__init__(units, activation=activation, use_bias=False, **kwargs)
    
    def build(self, input_shape):
        super(PointDense, self).build(input_shape)
    
    def call(self, inputs):
        return super(PointDense, self).call(inputs)

# Cell 4
class PointCNN(tf.keras.Model):

    def __init__(self, optimizer=None):
        super(PointCNN, self).__init__()
        self.Conv_1 = PointConv2D(32, (5, 5), 1, 'same', tf.nn.relu, tf.float32)
        self.MaxPool_1 = tf.keras.layers.MaxPooling2D((2, 2), strides=2)
        self.Conv_2 = PointConv2D(64, (5, 5), 1,  'same', tf.nn.relu, tf.float32)
        self.MaxPool_2 = tf.keras.layers.MaxPooling2D((2, 2), strides=2)
        self.Flatten_1 = tf.keras.layers.Flatten()
        self.Dense_1 = PointDense(1024, tf.nn.relu)
        self.Dense_2 = PointDense(10)
        self.Layers = [self.Conv_1, self.MaxPool_1, self.Conv_2, self.MaxPool_2, 
                       self.Flatten_1, self.Dense_1, self.Dense_2]
        self.TrainableLayers = [self.Conv_1, self.Conv_2, self.Dense_1, self.Dense_2]
        self.optimizer = optimizer
    
    def build(self, input_shape):
        for layer in self.Layers:
            layer.build(input_shape)
            input_shape = layer.compute_output_shape(input_shape)

    def getstate(self):
        np_kernels = []
        for layer in self.TrainableLayers:
            np_kernels.append(layer.kernel.numpy())
        return np.asarray(np_kernels)

    def setstate(self, kernels):
        i = 0
        while i < kernels.shape[0]:
            layer = self.TrainableLayers[i]
            layer.kernel.assign(tf.convert_to_tensor(kernels[i], tf.float32))
            i += 1

class PerturbedNN(tf.keras.Model):

    def __init__(self, optimizer=None):
        super(PerturbedNN, self).__init__()
        self.Conv_1 = PerturbedConv2D(32, (5, 5), 1, 'same', tf.nn.relu, tf.float32)
        self.MaxPool_1 = tf.keras.layers.MaxPooling2D((2, 2), strides=2)
        self.Conv_2 = PerturbedConv2D(64, (5, 5), 1,  'same', tf.nn.relu, tf.float32)
        self.MaxPool_2 = tf.keras.layers.MaxPooling2D((2, 2), strides=2)
        self.Flatten_1 = tf.keras.layers.Flatten()
        self.Dense_1 = PerturbedDense(1024, tf.nn.relu)
        self.Dense_2 = PerturbedDense(10)
        self.Layers = [self.Conv_1, self.MaxPool_1, self.Conv_2, self.MaxPool_2, 
                       self.Flatten_1, self.Dense_1, self.Dense_2]
        self.TrainableLayers = [self.Conv_1, self.Conv_2, self.Dense_1, self.Dense_2]
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
        avg_pred = tf.reduce_mean(output_tensor, 0)
        entropy = tf.reduce_mean(-tf.reduce_sum(avg_pred*tf.math.log(avg_pred + 1e-10), 1))
        return entropy

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
        for layer in self.TrainableLayers:
            np_kernel_mu.append(layer.kernel_mu.numpy())
            np_kernel_rho.append(layer.kernel_rho.numpy())
        return np.asarray(np_kernel_mu + np_kernel_rho)

    def setstate(self, kernels):
        itr = iter(self.TrainableLayers)
        i = 0
        while i < kernels.shape[0]//2:
            layer = next(itr)
            layer.kernel_mu.assign(tf.convert_to_tensor(kernels[i], tf.float32))
            i += 1
        itr = iter(self.TrainableLayers)
        i = kernels.shape[0]//2
        while i < kernels.shape[0]:
            layer = next(itr)
            layer.kernel_rho.assign(tf.convert_to_tensor(kernels[i], tf.float32))
            i += 1
    
    def transferstate(self, kernel_mu):
        itr = iter(self.TrainableLayers)
        i = 0
        while i < kernel_mu.shape[0]:
            layer = next(itr)
            layer.kernel_mu.assign(tf.convert_to_tensor(kernel_mu[i], tf.float32))
            i += 1
    
class LeNetIOD(PerturbedNN):

    def mou(self, output_tensor):
        un = tf.reduce_mean(tf.reduce_sum(tf.math.reduce_std(output_tensor, 0)**2, 1))
        return un

# Cell 5
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
x_train, x_test = x_train / 255., x_test / 255.
x_train, x_test = np.reshape(x_train, x_train.shape + (1,)), np.reshape(x_test, x_test.shape + (1,))

# Cell 6
(x_train_f, y_train_f), (x_test_f, y_test_f) = tf.keras.datasets.fashion_mnist.load_data()
x_train_f, x_test_f = np.array(x_train_f, np.float32), np.array(x_test_f, np.float32)
x_train_f, x_test_f = x_train_f / 255., x_test_f / 255.

# Cell 7
prefix = ""

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

def train(train_images, train_labels, alpha, iterations, batch_size, checkpoint=None, mean_file=None, **kwargs):
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
        weight_file = "weights/mnist/" + curr_date_time
        if mean_file is not None:
            point_model = PointCNN()
            point_model.build((None,) + train_images.shape[1:])
            load_config(prefix + mean_file, point_model)
            model.transferstate(point_model.getstate())
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
        
        save_config(train_config, model, prefix + weight_file)
        return weight_file
    except KeyboardInterrupt:
        save_config(train_config, model, prefix + weight_file)
        return weight_file

def test(test_images, test_labels, samples, weight_file, batch_size=128, **kwargs):
    test_data = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    test_data = test_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)
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
train(x_train, y_train, 0.1, 100, 256, checkpoint="weights/mnist/2020_09_27_06_50_38_475439", c=1, b=1e10, a=1, model_type=LeNetIOD)

# Cell 9
test(x_test, y_test, 1, "weights/mnist/2020_09_27_06_50_38_475439")

def gen_adv(ann_path, x, eps):
    test_data = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    test_data = test_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)
    model = kwargs.get('model_type', PointCNN)()
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
