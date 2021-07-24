# -*- coding: utf-8 -*-

# Cell 1
import tensorflow as tf, numpy as np
import math
from datetime import datetime
import time, pickle

# Cell 2
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
                                          initializer=tf.initializers.glorot_normal,
                                          trainable=True,
                                          dtype=self.dtype)
    
    def _reparametrize(self, avoid_sampling=False):
        if avoid_sampling:
            return self.kernel_mu
        eps_w_shape = self.kernel_mu.shape
        eps_w = tf.random.normal(eps_w_shape, 0, 1, dtype=tf.float32)
        term_w = tf.math.multiply(eps_w,
                                  tf.math.exp(tf.clip_by_value(self.kernel_rho, -87.315, 88.722)))
        return tf.math.add(self.kernel_mu, term_w)
    
    def call(self, inputs):
        self.kernel = self._reparametrize() + 0
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
                                          initializer=tf.initializers.glorot_uniform,
                                          trainable=True,
                                          dtype=self.dtype)
    
    def _reparametrize(self, avoid_sampling=False):
        if avoid_sampling:
            return self.kernel_mu
        eps_w_shape = self.kernel_mu.shape
        eps_w = tf.random.normal(eps_w_shape, 0, 1, dtype=tf.float32)
        term_w = tf.math.multiply(eps_w,
                                  tf.math.exp(tf.clip_by_value(self.kernel_rho, -87.315, 88.722)))
        return tf.math.add(self.kernel_mu, term_w)
    
    def call(self, inputs):
        self.kernel = self._reparametrize() + 0
        return super(PerturbedDense, self).call(inputs)

# Cell 3
class PerturbedNN(tf.keras.Model):

    def __init__(self, optimizer=None):
        super(PerturbedNN, self).__init__()
        self.Conv_1 = PerturbedConv2D(32, (5, 5), 1, 'same', tf.nn.relu, tf.float32)
        self.MaxPool_1 = tf.keras.layers.MaxPooling2D((2, 2), strides=2)
        self.Conv_2 = PerturbedConv2D(64, (5, 5), 1, 'same', tf.nn.relu, tf.float32)
        self.MaxPool_2 = tf.keras.layers.MaxPooling2D((2, 2), strides=2)
        self.Flatten_1 = tf.keras.layers.Flatten()
        self.Dense_1 = PerturbedDense(1024, tf.nn.relu)
        self.Dense_2 = PerturbedDense(10)
        self.Layers = [self.Conv_1, self.MaxPool_1, self.Conv_2, self.MaxPool_2, 
                       self.Flatten_1, self.Dense_1, self.Dense_2]
        self.TrainableLayers = [self.Conv_1, self.Conv_2, self.Dense_1, self.Dense_2]
        self.LayerWiseOutputs = []
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
    
    def log_prior(self, weights):
        """
        Computes the natural logarithm of scale
        mixture prior of weights.

        Parameters
        ==========

        weights: tf.Tensor

        Returns
        =======

        tf.Tensor

        Note
        ====

        The two standard deviations of the scale mixture are,
        exp(0) and exp(-6). The weight of both normal distributions
        is 0.5.
        """
        shape = weights.shape
        sigma_1 = tf.constant(0.005, shape=shape, dtype=tf.float32)
        sigma_2 = tf.constant(0.00005, shape=shape, dtype=tf.float32)
        def pdf(w, sigma):
            res1 = tf.math.divide(tf.math.square(w), tf.math.square(sigma)*2)
            return tf.math.divide(tf.math.exp(tf.clip_by_value(-res1, -87.315, 88.722)), sigma*(2*math.pi)**0.5)
        part_1 = tf.clip_by_value(0.25*pdf(weights, sigma_1), tf.float32.min//2, tf.float32.max//2)
        part_2 = tf.clip_by_value(0.75*pdf(weights, sigma_2), tf.float32.min//2, tf.float32.max//2)
        return tf.math.reduce_sum(tf.math.log(part_1 + part_2))

    def log_posterior(self, weights, mu, rho):
        """
        Computes the natural logarithm of Gaussian
        posterior on weights.

        Parameters
        ==========

        weights: tf.Tensor
        mu: tf.Tensor
          The mean of the posterior Gaussian distribution.
        rho: tf.Tensor
          Used to compute the variance of the posterior Gaussian distribution.
        
        Returns
        =======

        tf.Tensor
        """
        def pdf(w, mu, sigma):
            res1 = tf.math.divide(tf.math.square(w - mu), tf.math.square(sigma)*2)
            return tf.math.divide(tf.math.exp(tf.clip_by_value(-res1, -87.315, 88.722)), sigma*(2*math.pi)**0.5)
        sigma = tf.math.log(tf.math.add(
                                  tf.math.exp(tf.clip_by_value(rho, -87.315, 88.722)),
                                  tf.constant(1., shape=rho.shape, dtype=tf.float32)))
        log_q = tf.math.log(tf.clip_by_value(pdf(weights, mu, sigma), tf.float32.min//2, tf.float32.max//2))
        return tf.math.reduce_sum(log_q)

    def get_loss(self, inputs, targets, samples, weight=1., inference=False):
        """
        Computes the total training loss.

        Parameters
        ==========

        inputs: tf.Tensor/np.array
            Input to the layers.
        targets: tf.Tensor/np.array
            True targets that the model wants to learn from.
        samples: int
            The number of samples to be drawn for weights.
        weight: tf.float32 or equivalent.
            Weight given to loss of each batch. By default, 1.
        inference: bool
            Used to determine the order of the outputs in the tuple
            being returned.
        
        Returns
        =======

        tuple
          Containing loss and output of neural network for each sample.
        """
        loss = tf.constant(0., dtype=tf.float32)
        avg_cse = tf.constant(0., dtype=tf.float32)
        outputs_list = []
        targets = tf.cast(targets, tf.int64)
        for _ in range(samples):
            outputs = self.call(inputs, is_training=True)
            outputs_list.append(outputs)
            pw, qw = tf.constant(0, dtype=tf.float32), tf.constant(0, dtype=tf.float32)
            for layer in self.TrainableLayers:
                pw += self.log_prior(layer.kernel)
                qw += self.log_posterior(layer.kernel, layer.kernel_mu, layer.kernel_rho)

            cse = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(targets, outputs))
            avg_cse += cse
            if inference:
                loss += cse
            else:
                loss += (qw - pw)*weight + tf.cast(cse, tf.float32)

        if inference:
            return outputs_list, loss/samples, avg_cse/samples
        return loss/samples, outputs_list, avg_cse/samples
    
    def compute_gradients(self, inputs, targets, weight, ignore_mean=False):
        """
        Computes gradients of cost function for given inputs, 
        targets.

        Parameters
        ==========

        inputs: tf.Tensor
        targets: tf.Tensor
        weight: tf.float32 or equivalent
          The weight given to each batch of input data.
        
        Returns
        =======

        list
          Containing gradients, tf.Tensor, w.r.t each variable.
        """
        _vars = []
        with tf.GradientTape(persistent=True) as tape:
            for layer in self.TrainableLayers:
                if not ignore_mean:
                    _vars.append(layer.kernel_mu)
                _vars.append(layer.kernel_rho)
            tape.watch(_vars)
            F, _, avg_cse = self.get_loss(inputs, targets, 10, weight)

        dF = tape.gradient(F, _vars)
        
        return dF, F, avg_cse, _vars
    
    def fit(self, inputs, targets, weight=1., ignore_mean=False):
        """
        Performs parameter updates.

        Parameters
        ==========

        inputs: tf.Tensor
        targets: tf.Tensor
        alpha: tf.float32 or equivalent
          The learning rate.
        weight: tf.float32 or equivalent
          The weight given to each batch of input data.
        
        Returns
        =======

        None
        """
        start_time = time.time()
        grads, F, cse, vars = self.compute_gradients(inputs, targets, weight, ignore_mean=ignore_mean)
        self.optimizer.apply_gradients(zip(grads, vars))
        end_time = time.time()
        return F, cse, end_time - start_time

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

class PointCNN(tf.keras.Model):

    def __init__(self, optimizer=None):
        super(PointCNN, self).__init__()
        self.Conv_1 = PointConv2D(32, (5, 5), 1, 'same', tf.nn.relu, tf.float32)
        self.MaxPool_1 = tf.keras.layers.MaxPooling2D((2, 2), strides=2)
        self.Conv_2 = PointConv2D(64, (5, 5), 1, 'same', tf.nn.relu, tf.float32)
        self.MaxPool_2 = tf.keras.layers.MaxPooling2D((2, 2), strides=2)
        self.Flatten_1 = tf.keras.layers.Flatten()
        self.Dense_1 = PointDense(1024, tf.nn.relu)
        self.Dense_2 = PointDense(10)
        self.Layers = [self.Conv_1, self.MaxPool_1, self.Conv_2, self.MaxPool_2, 
                       self.Flatten_1, self.Dense_1, self.Dense_2]
        self.TrainableLayers = [self.Conv_1, self.Conv_2, self.Dense_1, self.Dense_2]
        self.LayerWiseOutputs = []
        self.optimizer = optimizer
    
    def build(self, input_shape):
        for layer in self.Layers:
            layer.build(input_shape)
            input_shape = layer.compute_output_shape(input_shape)
    
    def call(self, inputs, is_training=False):
        for layer in self.Layers:
            inputs = layer.call(inputs)
            self.LayerWiseOutputs.append(inputs)
        if not is_training:
            inputs = tf.nn.softmax(inputs)
        return inputs

    def get_loss(self, inputs, targets, inference=False):
        targets = tf.cast(targets, tf.int64)
        outputs = self.call(inputs, is_training=True)
        cse = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(targets, outputs))
        
        if inference:
            return outputs, cse
        return cse, outputs
    
    def compute_gradients(self, inputs, targets):
        _vars = []
        with tf.GradientTape(persistent=True) as tape:
            for layer in self.TrainableLayers:
                _vars.append(layer.kernel)
            F, _ = self.get_loss(inputs, targets)
        dF = tape.gradient(F, _vars)
        
        return dF, F, _vars
    
    def fit(self, inputs, targets):
        """
        Performs parameter updates.

        Parameters
        ==========

        inputs: tf.Tensor
        targets: tf.Tensor
        alpha: tf.float32 or equivalent
          The learning rate.
        weight: tf.float32 or equivalent
          The weight given to each batch of input data.
        
        Returns
        =======

        None
        """
        start_time = time.time()
        grads, F, vars = self.compute_gradients(inputs, targets)
        self.optimizer.apply_gradients(zip(grads, vars))
        end_time = time.time()
        return F, end_time - start_time

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

class PointCNNTScaled(PointCNN):

    def call(self, inputs, T=1, is_training=False):
        for layer in self.Layers:
            inputs = layer.call(inputs)
        if not is_training:
            inputs = tf.nn.softmax(inputs/T)
        return inputs

# Cell 6
prefix = ""

def load_config(file_path, model):
    train_config = None
    infile = open(file_path, 'rb')
    train_config = pickle.load(infile)
    model.setstate(np.load(file_path + ".npy", allow_pickle=True))
    return train_config

# Cell 4
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
x_train, x_test = x_train / 255., x_test / 255.
x_train, x_test = np.reshape(x_train, x_train.shape + (1,)), np.reshape(x_test, x_test.shape + (1,))

# Cell 5
(x_train_f, y_train_f), (x_test_f, y_test_f) = tf.keras.datasets.fashion_mnist.load_data()
x_train_f, x_test_f = np.array(x_train_f, np.float32), np.array(x_test_f, np.float32)
x_train_f, x_test_f = x_train_f / 255., x_test_f / 255.
x_train_f, x_test_f = np.reshape(x_train_f, x_train_f.shape + (1,)), np.reshape(x_test_f, x_test_f.shape + (1,))

class EvalBayesLeNet:

    def __init__(self, x_in, x_out, weight_file=None, entropy_file=None, **kwargs):
        self.total = x_in.shape[0]
        if entropy_file is None:
            self.model = PerturbedNN()
            self.model.build((None,) + x_in.shape[1:] + (1,))
            load_config(weight_file, self.model)
            samples = kwargs.get('samples', 10)
            self.entropy_in = self._infer(x_in, samples)
            self.entropy_out = self._infer(x_out, samples)
            curr_date_time = (str(datetime.now()).replace(' ', '_')
                                                .replace(':', '_')
                                                .replace('-', '_')
                                                .replace('.', '_'))
            entropies = np.asarray([self.entropy_in, self.entropy_out])
            np.save(prefix + "results/mnist/" + curr_date_time, entropies)
        else:
            entropies = np.load(entropy_file, allow_pickle=True)
            self.entropy_in, self.entropy_out = entropies[0], entropies[1]
            print(tf.reduce_mean(self.entropy_in), tf.math.reduce_std(self.entropy_in), tf.reduce_max(self.entropy_in)) 
            print(tf.reduce_mean(self.entropy_out), tf.math.reduce_std(self.entropy_out), tf.reduce_min(self.entropy_out))
        self.inc = 0.1
        self.start = 0
        self.end = 10
    
    def _infer(self, x, samples):
        batch_size = 1000
        x_inp = tf.reshape(x, x.shape + (1,))
        outputs = []
        for _ in range(samples):
            output = []
            test_data = tf.data.Dataset.from_tensor_slices(x_inp)
            test_data = test_data.repeat().batch(batch_size).prefetch(1)
            for _, batch_x in enumerate(test_data.take(x_inp.shape[0]//batch_size), 1):
                batch_output = list(self.model.call(batch_x).numpy())
                output.extend(batch_output)
            outputs.append(tf.convert_to_tensor(output, dtype=tf.float32))
        output_tensor = tf.convert_to_tensor(outputs)
        print(output_tensor.shape)
        avg_pred = tf.reduce_mean(output_tensor, 0)
        entropy = -tf.reduce_sum(avg_pred*tf.math.log(avg_pred + 1e-10), 1)
        print(tf.reduce_min(entropy), tf.reduce_max(entropy))
        return entropy

    def fpr_at_95_tpr(self):
        # Positive = In Distribution (ID)
        # Negative = Out of Distribution (OOD)
        delta = self.start
        total_fpr = 0
        total_thresholds = 0
        while delta <= self.end:
            tp = tf.reduce_sum(tf.cast(tf.math.less(self.entropy_in, tf.constant(delta, dtype=tf.float32)), tf.int32))
            fn = self.total - tp
            fp = tf.reduce_sum(tf.cast(tf.math.less(self.entropy_out, tf.constant(delta, dtype=tf.float32)), tf.int32))
            tn = self.total - fp
            tpr = tp/(tp + fn)
            fpr = fp/(fp + tn)
            print(fpr, tpr)
            if tpr <= 0.9505 and tpr >= 0.9495:
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
            tp = tf.reduce_sum(tf.cast(tf.math.less(self.entropy_in, tf.constant(delta, dtype=tf.float32)), tf.int32))
            fn = self.total - tp
            fp = tf.reduce_sum(tf.cast(tf.math.less(self.entropy_out, tf.constant(delta, dtype=tf.float32)), tf.int32))
            tn = self.total - fp
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
        delta = self.start
        while delta <= self.end:
            tp = tf.reduce_mean(tf.cast(tf.math.less(self.entropy_in, tf.constant(delta, dtype=tf.float32)), tf.float64))
            fp = tf.reduce_mean(tf.cast(tf.math.less(self.entropy_out, tf.constant(delta, dtype=tf.float32)), tf.float64))
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
        delta = self.start
        total_thresholds, total_pe = 0, 0
        while delta <= self.end:
            tp = tf.reduce_sum(tf.cast(tf.math.less(self.entropy_in, tf.constant(delta, dtype=tf.float32)), tf.int32))
            fn = self.total - tp
            fp = tf.reduce_sum(tf.cast(tf.math.less(self.entropy_out, tf.constant(delta, dtype=tf.float32)), tf.int32))
            tn = self.total - fp
            tpr = tp/(tp + fn)
            fpr = fp/(fp + tn)
            if tpr <= 0.9505 and tpr >= 0.9495:
                total_pe += 0.5*(1 - tpr) + 0.5*fpr
                total_thresholds += 1 
            delta += self.inc
        return (total_pe/total_thresholds).numpy()*100

class EvalLeNetEntropy:
    
    def __init__(self, x_in, x_out, weight_file=None, entropy_file=None, **kwargs):
        self.total = x_in.shape[0]
        if entropy_file is None:
            self.model = PointCNN()
            self.model.build((None,) + x_in.shape[1:] + (1,))
            load_config(weight_file, self.model)
            samples = kwargs.get('samples', 1)
            self.entropy_in = self._infer(x_in, samples)
            self.entropy_out = self._infer(x_out, samples)
            curr_date_time = (str(datetime.now()).replace(' ', '_')
                                                .replace(':', '_')
                                                .replace('-', '_')
                                                .replace('.', '_'))
            entropies = np.asarray([self.entropy_in, self.entropy_out])
            np.save(prefix + "results/mnist/" + curr_date_time, entropies)
        else:
            entropies = np.load(entropy_file, allow_pickle=True)
            self.entropy_in, self.entropy_out = entropies[0], entropies[1]
            print(tf.reduce_mean(self.entropy_in), tf.math.reduce_std(self.entropy_in), tf.reduce_max(self.entropy_in)) 
            print(tf.reduce_mean(self.entropy_out), tf.math.reduce_std(self.entropy_out), tf.reduce_min(self.entropy_out))
        self.inc = 0.0001
        self.start = 0.4
        self.end = 0.7
    
    def _infer(self, x, samples):
        batch_size = 1000
        x_inp = tf.reshape(x, x.shape + (1,))
        outputs = []
        for _ in range(samples):
            output = []
            test_data = tf.data.Dataset.from_tensor_slices(x_inp)
            test_data = test_data.repeat().batch(batch_size).prefetch(1)
            for _, batch_x in enumerate(test_data.take(x_inp.shape[0]//batch_size), 1):
                batch_output = list(self.model.call(batch_x).numpy())
                output.extend(batch_output)
            outputs.append(tf.convert_to_tensor(output, dtype=tf.float32))
        output_tensor = tf.convert_to_tensor(outputs)
        print(output_tensor.shape)
        avg_pred = tf.reduce_mean(output_tensor, 0)
        entropy = -tf.reduce_sum(avg_pred*tf.math.log(avg_pred + 1e-10), 1)
        print(tf.reduce_min(entropy), tf.reduce_max(entropy))
        return entropy

    def fpr_at_95_tpr(self):
        # Positive = In Distribution (ID)
        # Negative = Out of Distribution (OOD)
        delta = self.start
        total_fpr = 0
        total_thresholds = 0
        while delta <= self.end:
            tp = tf.reduce_sum(tf.cast(tf.math.less(self.entropy_in, tf.constant(delta, dtype=tf.float32)), tf.int32))
            fn = self.total - tp
            fp = tf.reduce_sum(tf.cast(tf.math.less(self.entropy_out, tf.constant(delta, dtype=tf.float32)), tf.int32))
            tn = self.total - fp
            tpr = tp/(tp + fn)
            fpr = fp/(fp + tn)
            if tpr <= 0.9505 and tpr >= 0.9495:
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
            tp = tf.reduce_sum(tf.cast(tf.math.less(self.entropy_in, tf.constant(delta, dtype=tf.float32)), tf.int32))
            fn = self.total - tp
            fp = tf.reduce_sum(tf.cast(tf.math.less(self.entropy_out, tf.constant(delta, dtype=tf.float32)), tf.int32))
            tn = self.total - fp
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
        delta = self.start
        while delta <= self.end:
            tp = tf.reduce_mean(tf.cast(tf.math.less(self.entropy_in, tf.constant(delta, dtype=tf.float32)), tf.float64))
            fp = tf.reduce_mean(tf.cast(tf.math.less(self.entropy_out, tf.constant(delta, dtype=tf.float32)), tf.float64))
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
        delta = self.start
        total_thresholds, total_pe = 0, 0
        while delta <= self.end:
            tp = tf.reduce_sum(tf.cast(tf.math.less(self.entropy_in, tf.constant(delta, dtype=tf.float32)), tf.int32))
            fn = self.total - tp
            fp = tf.reduce_sum(tf.cast(tf.math.less(self.entropy_out, tf.constant(delta, dtype=tf.float32)), tf.int32))
            tn = self.total - fp
            tpr = tp/(tp + fn)
            fpr = fp/(fp + tn)
            if tpr <= 0.9505 and tpr >= 0.9495:
                total_pe += 0.5*(1 - tpr) + 0.5*fpr
                total_thresholds += 1 
            delta += self.inc
        return (total_pe/total_thresholds).numpy()*100

class EvalLeNet:

    def __init__(self, x_in, x_out, weight_file=None, softmax_file=None, **kwargs):
        self.total = x_in.shape[0]
        if softmax_file is None:
            self.model = PointCNN()
            self.model.build((None,) + x_in.shape[1:] + (1,))
            load_config(weight_file, self.model)
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
        self.inc = 0.1
        self.start = 0
        self.end = 100
    
    def _infer(self, x):
        x = x.reshape(x.shape + (1,))
        outputs = self.model.call(x)
        softmax_scores = tf.reduce_max(outputs, 1)
        return softmax_scores

    def fpr_at_95_tpr(self):
        # Positive = In Distribution (ID)
        # Negative = Out of Distribution (OOD)
        delta = self.start
        total_fpr = 0
        total_thresholds = 0
        tprs, fprs = [], []
        while delta <= self.end:
            tp = tf.reduce_sum(tf.cast(tf.math.greater_equal(self.softmax_in, tf.constant(delta, dtype=tf.float32)), tf.int32))
            fn = self.total - tp
            fp = tf.reduce_sum(tf.cast(tf.math.greater_equal(self.softmax_out, tf.constant(delta, dtype=tf.float32)), tf.int32))
            tn = self.total - fp
            tpr = tp/(tp + fn)
            fpr = fp/(fp + tn)
            tprs.append(tpr)
            fprs.append(fpr)
            if tpr >= 0.94 or tpr <= 0.001:
                print(delta, tpr, fpr)
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
            tp = tf.reduce_sum(tf.cast(tf.math.greater_equal(self.softmax_in, tf.constant(delta, dtype=tf.float32)), tf.int32))
            fn = self.total - tp
            fp = tf.reduce_sum(tf.cast(tf.math.greater_equal(self.softmax_out, tf.constant(delta, dtype=tf.float32)), tf.int32))
            tn = self.total - fp
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
            tp = tf.reduce_mean(tf.cast(tf.math.greater_equal(self.softmax_in, tf.constant(delta, dtype=tf.float32)), tf.float64))
            fp = tf.reduce_mean(tf.cast(tf.math.greater_equal(self.softmax_out, tf.constant(delta, dtype=tf.float32)), tf.float64))
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
            tp = tf.reduce_sum(tf.cast(tf.math.greater_equal(self.softmax_in, tf.constant(delta, dtype=tf.float32)), tf.int32))
            fn = self.total - tp
            fp = tf.reduce_sum(tf.cast(tf.math.greater_equal(self.softmax_out, tf.constant(delta, dtype=tf.float32)), tf.int32))
            tn = self.total - fp
            tpr = tp/(tp + fn)
            fpr = fp/(fp + tn)
            if tpr <= 0.9549 and tpr >= 0.9450:
                total_pe += 0.5*(1 - tpr) + 0.5*fpr
                total_thresholds += 1                
            delta += self.inc
        return (total_pe/total_thresholds).numpy()*100
        

class EvalLeNetMHB(EvalLeNet):

    def __init__(self, x_in, x_out, weight_file=None, entropy_file=None, **kwargs):
        self.total = x_in.shape[0]
        if entropy_file is None:
            self.model = PointCNN()
            self.model.build((None,) + x_in.shape[1:])
            load_config(weight_file, self.model)
            self.mhb_dist_in = self._infer(x_in)
            self.mhb_dist_out = self._infer(x_out)
            curr_date_time = (str(datetime.now()).replace(' ', '_')
                                                .replace(':', '_')
                                                .replace('-', '_')
                                                .replace('.', '_'))
            mhb_dist = np.asarray([self.mhb_dist_in, self.mhb_dist_out])
            np.save(prefix + "results/mnist/" + curr_date_time, mhb_dist)
            print(prefix + "results/mnist/" + curr_date_time + ".npy")
        else:
            mhb_dist = np.load(entropy_file, allow_pickle=True)
            self.mhb_dist_in, self.mhb_dist_out = mhb_dist[0], mhb_dist[1]
            print(tf.reduce_min(self.mhb_dist_in), tf.reduce_max(self.mhb_dist_in)) 
            print(tf.reduce_min(self.mhb_dist_out), tf.reduce_max(self.mhb_dist_out))
        self.softmax_in, self.softmax_out = self.mhb_dist_in, self.mhb_dist_out
        self.inc = -2.0e-10
        self.start = -2.0e-9
        self.end = 2.0e7
    
    def _infer(self, x):
        x_test_re = tf.convert_to_tensor(x, dtype=tf.float32)
        dataset = "mnist"
        sigma = np.load(prefix + "weights/" + dataset + "/sigma.npy")
        self.model.call(x_test_re, is_training=True)
        f_x = self.model.LayerWiseOutputs[-1]
        mhbs = []
        for _cls in range(10):
            mu = np.load(prefix + "weights/" + dataset + "/mean_" + str(_cls) + ".npy")
            mu = np.reshape(mu, (1,) + mu.shape)
            mhb_mat = -tf.matmul(tf.matmul(f_x - mu, sigma), f_x - mu, transpose_b=True)
            mhbs.append(tf.linalg.diag_part(mhb_mat))
        mhbs = tf.convert_to_tensor(mhbs)
        mhbs = tf.reduce_max(mhbs, 0)
        print(mhbs.shape)
        return mhbs

class EvalBayesAdapter(EvalLeNet):

    def __init__(self, x_in, x_out, weight_file=None, entropy_file=None, **kwargs):
        self.total = x_in.shape[0]
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
            np.save(prefix + "results/mnist/" + curr_date_time, entropies)
            print(prefix + "results/mnist/" + curr_date_time)
        else:
            entropies = np.load(entropy_file, allow_pickle=True)
            self.entropy_in, self.entropy_out = entropies[0], entropies[1]
            print(tf.reduce_mean(self.entropy_in), tf.math.reduce_std(self.entropy_in), tf.reduce_min(self.entropy_in), tf.reduce_max(self.entropy_in)) 
            print(tf.reduce_mean(self.entropy_out), tf.math.reduce_std(self.entropy_out), tf.reduce_min(self.entropy_in), tf.reduce_max(self.entropy_out))
        self.softmax_in, self.softmax_out = self.entropy_in, self.entropy_out
        print(self.softmax_in.shape, self.softmax_out.shape)
        self.inc = 1.0e-9
        self.start = -3.0e-7
        self.end = 5.0e-7

    def _infer(self, x, samples):
        batch_size = 1000
        x_inp = x
        outputs = []
        for _ in range(samples):
            output = []
            test_data = tf.data.Dataset.from_tensor_slices(x_inp)
            test_data = test_data.repeat().batch(batch_size).prefetch(1)
            for _, batch_x in enumerate(test_data.take(x_inp.shape[0]//batch_size), 1):
                batch_output = list(self.model.call(batch_x).numpy())
                output.extend(batch_output)
            outputs.append(tf.convert_to_tensor(output, dtype=tf.float32))
        output_tensor = tf.convert_to_tensor(outputs)
        print(output_tensor.shape)
        avg_pred = tf.reduce_mean(output_tensor, 0)
        entropy_avg = -tf.reduce_sum(avg_pred*tf.math.log(avg_pred + 1e-10), 1)
        avg_entropy = tf.reduce_mean(-tf.reduce_sum(avg_pred*tf.math.log(output_tensor + 1e-10), 2), 0)
        return entropy_avg - avg_entropy

class EvalEnsembleLeNet(EvalLeNet):

    def __init__(self, x_in, x_out, weight_files, entropy_file=None, **kwargs):
        self.total = x_in.shape[0]
        self.models = []
        if entropy_file is None:
            for weight_file in weight_files:
                model = PointCNN()
                model.build((None,) + x_in.shape[1:])
                load_config(weight_file, model)
                self.models.append(model)
            self.entropy_in = self._infer(x_in)
            self.entropy_out = self._infer(x_out)
            curr_date_time = (str(datetime.now()).replace(' ', '_')
                                                .replace(':', '_')
                                                .replace('-', '_')
                                                .replace('.', '_'))
            entropies = np.asarray([self.entropy_in, self.entropy_out])
            np.save(prefix + "results/mnist/" + curr_date_time, entropies)
            print(prefix + "results/mnist/" + curr_date_time + ".npy")
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

    def _infer(self, x):

        def call_model(model, x_inp):
            batch_size = 100
            test_data = tf.data.Dataset.from_tensor_slices(x_inp)
            test_data = test_data.repeat().batch(batch_size).prefetch(1)
            output = []
            for _, batch_x in enumerate(test_data.take(x_inp.shape[0]//batch_size), 1):
                batch_output = list(model.call(batch_x).numpy())
                output.extend(batch_output)
            return tf.convert_to_tensor(output)

        outputs = []
        i = 0
        for model in self.models:
            outputs.append(call_model(model, x))
            i += 1
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
            tp = tf.reduce_sum(tf.cast(tf.math.less_equal(self.softmax_in, tf.constant(delta, dtype=tf.float32)), tf.int32))
            fn = self.total - tp
            fp = tf.reduce_sum(tf.cast(tf.math.less_equal(self.softmax_out, tf.constant(delta, dtype=tf.float32)), tf.int32))
            tn = self.total - fp
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
            tp = tf.reduce_sum(tf.cast(tf.math.less_equal(self.softmax_in, tf.constant(delta, dtype=tf.float32)), tf.int32))
            fn = self.total - tp
            fp = tf.reduce_sum(tf.cast(tf.math.less_equal(self.softmax_out, tf.constant(delta, dtype=tf.float32)), tf.int32))
            tn = self.total - fp
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
            tp = tf.reduce_sum(tf.cast(tf.math.less_equal(self.softmax_in, tf.constant(delta, dtype=tf.float32)), tf.int32))
            fn = self.total - tp
            fp = tf.reduce_sum(tf.cast(tf.math.less_equal(self.softmax_out, tf.constant(delta, dtype=tf.float32)), tf.int32))
            tn = self.total - fp
            tpr = tp/(tp + fn)
            fpr = fp/(fp + tn)
            if tpr <= 0.9549 and tpr >= 0.9450:
                total_pe += 0.5*(1 - tpr) + 0.5*fpr
                total_thresholds += 1                
            delta += self.inc
        return (total_pe/total_thresholds).numpy()*100

class EvalLeNetIndexOfDispersion(EvalLeNet):

    def __init__(self, x_in, x_out, weight_file=None, entropy_file=None, **kwargs):
        self.total = x_in.shape[0]
        measure_name = kwargs.get('measure', 'M1')
        print(measure_name)
        if entropy_file is None:
            self.model = PerturbedNN()
            self.model.build((None,) + x_in.shape[1:])
            load_config(weight_file, self.model)
            samples = kwargs.get('samples', 10)
            ein = self._infer(x_in, samples, measure_name)
            eout = self._infer(x_out, samples, measure_name)
            entropies = [ein, eout]
            print(ein.shape, eout.shape)
            if measure_name == 'M2':
                self.entropy_in = 1/entropies[0][0] + 100/entropies[0][1]
                self.entropy_out = 1/entropies[1][0] + 100/entropies[1][1]
            else:
                self.entropy_in, self.entropy_out = entropies[0], entropies[1]
            curr_date_time = (str(datetime.now()).replace(' ', '_')
                                                .replace(':', '_')
                                                .replace('-', '_')
                                                .replace('.', '_'))
            entropies = np.asarray(entropies)
            np.save(prefix + "results/mnist/" + curr_date_time, entropies)
            print(prefix + "results/mnist/" + curr_date_time)
        else:
            entropies = np.load(entropy_file, allow_pickle=True)
            if measure_name == 'M2':
                self.entropy_in = 1/entropies[0][0] + 10000/entropies[0][1]
                self.entropy_out = 1/entropies[1][0] + 10000/entropies[1][1]
            else:
                self.entropy_in, self.entropy_out = entropies[0], entropies[1]
            print(tf.reduce_min(self.entropy_in), tf.reduce_max(self.entropy_in)) 
            print(tf.reduce_min(self.entropy_out), tf.reduce_max(self.entropy_out))
        self.softmax_in, self.softmax_out = self.entropy_in, self.entropy_out
        print(self.softmax_in.shape, self.softmax_out.shape)
        self.inc = 1000
        self.start = 0
        self.end = 1000000

    def _infer(self, x, samples, measure_name):
        batch_size = 1000
        x_inp = x
        outputs = []
        for _ in range(samples):
            output = []
            test_data = tf.data.Dataset.from_tensor_slices(x_inp)
            test_data = test_data.repeat().batch(batch_size).prefetch(1)
            for _, batch_x in enumerate(test_data.take(x_inp.shape[0]//batch_size), 1):
                batch_output = list(self.model.call(batch_x).numpy())
                output.extend(batch_output)
            outputs.append(tf.convert_to_tensor(output, dtype=tf.float32))
        output_tensor = tf.convert_to_tensor(outputs)
        if measure_name == 'M1':
            un = tf.reduce_sum(tf.math.reduce_std(output_tensor, 0)**2/(tf.reduce_mean(output_tensor, 0) + 1e-16), 1)
            return -tf.math.log(un)
        elif measure_name == 'M2':
            avg_pred = tf.reduce_mean(output_tensor, 0)
            entropy = -tf.reduce_sum(avg_pred*tf.math.log(avg_pred + 1e-10), 1)
            iod = tf.reduce_sum((tf.math.reduce_std(output_tensor, 0)**2)/(tf.reduce_mean(output_tensor, 0) + 1e-16), 1)
            ret = np.asarray([iod, entropy])
            print(ret.shape)
            return ret

# adv_x_test = np.load("/content/drive/MyDrive/bnn/data/mnist/adv_" + str(0.2) + ".npy")

# Cell 8
num_samples, min_samples = 11, 10
disp_files = [] # ["./drive/My Drive/bnn/results/mnist/2021_02_06_19_32_41_768507.npy"] # ["./drive/My Drive/bnn/results/mnist/2021_02_06_19_20_45_171292.npy"] # ["./drive/My Drive/bnn/results/mnist/2021_02_06_19_03_21_668195.npy"] # ["./drive/My Drive/bnn/results/mnist/2021_02_06_18_52_15_031155.npy"] # ["./drive/My Drive/bnn/results/mnist/2021_02_06_18_42_06_947089.npy"] # ["./drive/My Drive/bnn/results/mnist/2021_02_06_18_36_22_435190.npy"] # ["./drive/My Drive/bnn/results/mnist/2021_02_06_18_33_39_875803.npy"] # ["./drive/My Drive/bnn/results/mnist/2021_02_02_08_37_40_905801.npy"] # ["./drive/My Drive/bnn/results/mnist/2021_02_02_08_28_00_603781.npy"] # ["./drive/My Drive/bnn/results/mnist/2021_02_02_08_26_52_431152.npy"] # ["./drive/My Drive/bnn/results/mnist/2021_02_02_08_16_25_767823.npy"] # ["./drive/My Drive/bnn/results/mnist/2021_02_02_08_13_08_412481.npy"] # ["./drive/My Drive/bnn/results/mnist/2021_02_02_08_04_43_748597.npy"] #["./drive/My Drive/bnn/results/mnist/2021_02_02_07_55_16_091590.npy"] #["/content/drive/My Drive/bnn/results/mnist/2020_09_27_06_58_23_878458.npy"] # ["/content/drive/My Drive/bnn/results/mnist/2020_10_17_07_17_38_926697.npy"] # ["/content/drive/My Drive/bnn/results/mnist/2020_09_25_10_48_59_232588.npy"]
disp_files = disp_files + [None for i in range(num_samples - min_samples - len(disp_files))]
scores = {"FPR at 95 % TPR": 0, "AUROC": 0, "AUPR-In": 0, "Detection Error": 0}
i = 0
for samples in range(min_samples, num_samples):
    grader = EvalLeNetIndexOfDispersion(x_test, x_test_f, weight_file=prefix + "weights/mnist/2020_09_27_06_50_38_475439", entropy_file=disp_files[i], samples=samples, measure='M2')
#     scores["FPR at 95 % TPR"] += grader.fpr_at_95_tpr()
#     scores["AUROC"] += round(grader.auroc(), 2)
#     scores["AUPR-In"] += grader.aupr_in()
#     scores["Detection Error"] += grader.detection()
#     i += 1
# for key in scores:
#     scores[key] = scores[key]/(num_samples - min_samples)
# print(scores)

"""2 - ./drive/My Drive/bnn/results/mnist/2021_03_15_14_36_20_964885

5 - ./drive/My Drive/bnn/results/mnist/2021_03_15_14_37_02_998355

10 - ./drive/My Drive/bnn/results/mnist/2021_03_15_14_37_42_506637
"""

adv_model = PointCNN()
adv_model.build((None,) + x_test.shape[1:])
load_config(prefix + "weights/mnist/2020_09_19_14_53_28_250809", adv_model)
adv_x_test = []
eps = 0.3
i = 0
test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_data = test_data.repeat().batch(1).prefetch(1)
for _, (batch_x, batch_y) in enumerate(test_data.take(x_test.shape[0]//1), 1):
    with tf.GradientTape() as tape:
        tape.watch(batch_x)
        F = adv_model.get_loss(batch_x, batch_y)
    dF = tf.math.sign(tape.gradient(F, batch_x))
    adv_x_test.append((batch_x + eps*dF).numpy()[0])
    print(str(i) + " done...")
    i += 1
adv_x_test = np.asarray(adv_x_test)
print(adv_x_test.shape)
np.save("/content/drive/MyDrive/bnn/data/mnist/adv_" + str(eps) + ".py", adv_x_test)

def test_adv(test_images, test_labels, weight_file, batch_size=100, **kwargs):
    test_data = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    test_data = test_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)
    model = PointCNN()
    model.build((None,) + test_images.shape[1:])
    test_config = load_config(prefix + weight_file, model)
    total_score = 0
    batches = 0
    for _, (batch_x, batch_y) in enumerate(test_data.take(test_images.shape[0]//batch_size), 1):
        outputs = model.call(batch_x)
        correct_prediction = tf.equal(tf.argmax(outputs, 1), tf.cast(batch_y, tf.int64))
        total_score += tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)*100
        batches += 1
    return total_score/batches

test_adv(adv_x_test, y_test, "weights/mnist/2020_09_19_14_53_28_250809", 100)

"""{'FPR at 95 % TPR': 12.463333333333333, 'AUROC': 96.74600100000002, 'AUPR-In': 97.00985284404713, 'Detection Error': 8.738333333333333}"""

# Cell 7
weight_files = [
    prefix + "weights/mnist/2021_01_30_08_39_23_493169",
    prefix + "weights/mnist/2021_01_30_08_40_15_110618",
    # prefix + "weights/mnist/2021_01_30_08_40_43_545708",
    # prefix + "weights/mnist/2021_01_30_08_41_50_115262",
    # prefix + "weights/mnist/2021_01_30_08_42_23_109625"
]
grader = EvalEnsembleLeNet(x_test, x_test_f, 
    weight_files=weight_files, 
    entropy_file="./drive/My Drive/bnn/results/mnist/2021_01_30_15_13_44_606797.npy")
scores = {"FPR at 95 % TPR": None, "AUROC": None, "AUPR-In": None, "Detection Error": None}
scores["FPR at 95 % TPR"] = grader.fpr_at_95_tpr()
scores["AUROC"] = grader.auroc()
scores["AUPR-In"] = grader.aupr_in()
scores["Detection Error"] = grader.detection()
print(scores)

num_samples, min_samples = 3, 2
disp_files = ["./drive/My Drive/bnn/results/mnist/2021_01_29_14_07_48_783249.npy"] # ["./drive/My Drive/bnn/results/mnist/2021_01_29_14_02_35_922317.npy"] # ["./drive/My Drive/bnn/results/mnist/2021_01_29_12_28_14_569533.npy"] #["/content/drive/My Drive/bnn/results/mnist/2020_09_27_06_58_23_878458.npy"] # ["/content/drive/My Drive/bnn/results/mnist/2020_10_17_07_17_38_926697.npy"] # ["/content/drive/My Drive/bnn/results/mnist/2020_09_25_10_48_59_232588.npy"]
disp_files = disp_files + [None for i in range(num_samples - min_samples - len(disp_files))]
scores = {"FPR at 95 % TPR": 0, "AUROC": 0, "AUPR-In": 0, "Detection Error": 0}
i = 0
for samples in range(min_samples, num_samples):
    grader = EvalBayesAdapter(x_test, x_test_f, weight_file=prefix + "weights/mnist/bayes_adapter_2021_01_29_11_51_15_108159", entropy_file=disp_files[i], samples=samples)
    scores["FPR at 95 % TPR"] += grader.fpr_at_95_tpr()
    scores["AUROC"] += grader.auroc()
    scores["AUPR-In"] += grader.aupr_in()
    scores["Detection Error"] += grader.detection()
    i += 1
for key in scores:
    scores[key] = scores[key]/(num_samples - min_samples)
print(scores)

# Cell 7
grader = EvalLeNetMHB(x_test, x_test_f, weight_file=prefix + "weights/mnist/2020_09_19_14_53_28_250809", entropy_file="./drive/My Drive/bnn/results/mnist/2020_10_23_11_36_51_472869.npy")
scores = {"FPR at 95 % TPR": None, "AUROC": None, "AUPR-In": None, "Detection Error": None}
scores["FPR at 95 % TPR"] = grader.fpr_at_95_tpr()
scores["AUROC"] = grader.auroc()
scores["AUPR-In"] = grader.aupr_in()
scores["Detection Error"] = grader.detection()
print(scores)

# Cell 9
grader = EvalLeNetEntropy(x_test, x_test_f, weight_file=prefix + "weights/mnist/2020_09_19_14_53_28_250809", entropy_file="/content/drive/My Drive/bnn/results/mnist/2020_09_25_12_27_08_695127.npy")
scores = {"FPR at 95 % TPR": None, "AUROC": None, "AUPR-In": None, "Detection Error": None}
scores["FPR at 95 % TPR"] = grader.fpr_at_95_tpr()
scores["AUROC"] = grader.auroc()
scores["AUPR-In"] = grader.aupr_in()
scores["Detection Error"] = grader.detection()
print(scores)

# Cell 10
grader = EvalLeNet(x_test, x_test_f, weight_file=prefix + "weights/mnist/2020_09_19_14_53_28_250809")
scores = {"FPR at 95 % TPR": None, "AUROC": None, "AUPR-In": None, "Detection Error": None}
scores["FPR at 95 % TPR"] = grader.fpr_at_95_tpr()
scores["AUROC"] = grader.auroc()
scores["AUPR-In"] = grader.aupr_in()
scores["Detection Error"] = grader.detection()
print(scores)

# Cell 11
def odin_noise(x_test, eps, weight_file):
    x_test_inp = tf.convert_to_tensor(x_test, dtype=tf.float32)
    model = PointCNNTScaled()
    model.build(x_test_inp.shape)
    load_config(weight_file, model)
    with tf.GradientTape() as tape:
        tape.watch(x_test_inp)
        outputs = model.call(x_test_inp, T=10)
        log_outputs = tf.math.log(tf.reduce_max(outputs, 1))
    grad = tape.gradient(log_outputs, x_test_inp)
    x_test_new = x_test_inp - eps*tf.sign(-grad)
    return tf.reshape(x_test_new, x_test_new.shape[:-1]).numpy()

x_test_new = odin_noise(x_test, 0.0001, prefix + "weights/mnist/2020_09_19_14_53_28_250809")
x_test_f_new = odin_noise(x_test_f, 0.0001, prefix + "weights/mnist/2020_09_19_14_53_28_250809")

grader = EvalLeNet(x_test_new, x_test_f_new, weight_file=prefix + "weights/mnist/2020_09_19_14_53_28_250809")
scores = {"FPR at 95 % TPR": None, "AUROC": None, "AUPR-In": None, "Detection Error": None}
scores["FPR at 95 % TPR"] = grader.fpr_at_95_tpr()
scores["AUROC"] = grader.auroc()
scores["AUPR-In"] = grader.aupr_in()
scores["Detection Error"] = grader.detection()
print(scores)

# Cell 12
scores = {"FPR at 95 % TPR": 0, "AUROC": 0, "AUPR-In": 0, "Detection Error": 0}
grader = EvalEnsembleLeNet(x_test, x_test_f, weight_file1=prefix + "weights/mnist/2020_10_04_12_26_26_960315", weight_file2=prefix + "weights/mnist/2020_09_19_14_53_28_250809", entropy_file=None)
scores["FPR at 95 % TPR"] += grader.fpr_at_95_tpr()
scores["AUROC"] += grader.auroc()
scores["AUPR-In"] += grader.aupr_in()
scores["Detection Error"] += grader.detection()
print(scores)
