from abc import abstractmethod
import tensorflow as tf
import numpy as np
import scipy
import operator
from scipy.stats.distributions import chi2

from GeodesicRelaxationDCCA.algorithms.correlation import CCA

ops = {'>': operator.gt, '<': operator.lt, '>=': operator.ge, '<=': operator.le, '==': operator.eq}

def get_cca_loss(fy_1, fy_2, num_shared_dim, rx=0, ry=0):
    B1, B2, epsilon, omega, ccor, _, _ = CCA(fy_1, fy_2, num_shared_dim, rx, ry)
    return tf.reduce_mean(tf.square(tf.norm(tf.subtract(epsilon, omega), axis=0))) / num_shared_dim

def compute_l1(weights):
    return tf.math.reduce_sum(abs(weights))

def compute_l2(weights):
    return tf.math.reduce_sum(tf.math.square(weights))

def get_rec_loss(y_1, y_2, yhat_1, yhat_2):
    rec_loss_1 = tf.square(tf.norm(y_1 - yhat_1, axis=0))
    rec_loss_2 = tf.square(tf.norm(y_2 - yhat_2, axis=0))
    return tf.add(tf.reduce_mean(rec_loss_1), tf.reduce_mean(rec_loss_2))/2

def get_mv_rec_loss(list_of_tuples):
    rec_loss = []
    for tup in list_of_tuples:
        rec_loss.append(tf.square(tf.norm(tup[0] - tup[1], axis=0)))
    return tf.reduce_mean(rec_loss)

def get_distance_metric(S, U):
    Ps = np.eye(S.shape[1]) - tf.transpose(S)@np.linalg.inv(S@tf.transpose(S))@S
    Q = scipy.linalg.orth(tf.transpose(U))
    dist = np.linalg.norm(Ps@Q, ord=2)
    return dist

def get_similarity_metric_v1(S, U, dims):
    _, _, _, _, ccor, _, _ = CCA(tf.transpose(S), tf.transpose(U), dims)
    return 1-ccor[0]

def get_mv_similarity_metric(S, U, dims):
    _, _, _, _, ccor, _, _ = CCA(tf.transpose(S), tf.transpose(U), dims)
    return 1 - (sum(ccor)/dims)

def get_similarity_metric_v2(S1, U1, S2, U2, dims):
    _, _, _, _, ccor_1, _, _ = CCA(tf.transpose(S1), tf.transpose(U1), dims)
    _, _, _, _, ccor_2, _, _ = CCA(tf.transpose(S2), tf.transpose(U2), dims)
    return np.mean(ccor_1+ccor_2)

class MetricDict():
    def __init__(self):
        self._keys = None
        self._dict = None

    def update(self, update_dict):
        if self._keys is None:
            self._keys = update_dict.keys()
            self._dict = {key: list() for key in self._keys}
            
        for key in self._keys:
            self._dict[key].append(update_dict[key])

    def output(self):
        return {key: np.concatenate(self._dict[key], axis=0) for key in self._keys}

class MovingMetric():
    def __init__(self, window_length, history_length, fun):
        self.window = list()
        self.window_length = window_length
        self.history = list()
        self.history_length = history_length
        self.fun = fun

    def update_window(self, value):
        if len(self.window) == self.window_length:
            self.window.pop(0)
        self.window.append(value)

        if len(self.history) == self.history_length:
            self.history.pop(0)
        self.history.append(self.fun(self.window))

    def get_metric(self):
        return self.history[-1]


class EmptyWatchdog():
    def decrease_counter(self):
        pass

    def is_active(self):
        return True

    def compute(self):
        return 0

    def check(self):
        return False

    def reset(self):
        pass


class EpochWatchdog():
    def __init__(self, num_epochs):
        self.num_epochs = num_epochs
        self.counter = self.num_epochs

    def decrease_counter(self):
        if self.counter > 0:
            self.counter -= 1

    def is_active(self):
        return True

    def compute(self):
        return self.counter

    def check(self):
        return self.counter == 0

    def reset(self):
        self.counter = self.num_epochs
        return True


class WindowedWatchdog():
    def __init__(self, moving_metric_dict, keys, threshold, metric_fun, metric_op, dead_time=1000):
        self.moving_metric_dict = moving_metric_dict
        self.keys = keys
        self.moving_metric = self.moving_metric_dict[self.keys.pop(0)]
        self.threshold = threshold
        self.metric_fun = metric_fun
        self.metric_op = metric_op
        self.dead_time = dead_time
        self.counter = self.dead_time

    def decrease_counter(self):
        if self.counter > 0:
            self.counter -= 1

    def is_active(self):
        return self.counter == 0

    def compute(self):
        if self.is_active():
            return self.metric_fun(self.moving_metric.history)
        else:
            return 0
    
    def check(self):
        if (len(self.moving_metric.history) == self.moving_metric.history_length) and self.is_active():
            return ops[self.metric_op](self.compute(), self.threshold)
        else:
            return False

    def reset(self):
        if len(self.keys) > 0:
            self.moving_metric = self.moving_metric_dict[self.keys.pop(0)]
            self.counter = self.dead_time
            return True
        else:
            return False


class CorrelationConvergenceWatchdog(WindowedWatchdog):
    def __init__(self, moving_metric, keys, dead_time=1000):
        super(CorrelationConvergenceWatchdog, self).__init__(
            moving_metric_dict=moving_metric,
            keys=keys,
            threshold=0.005,
            metric_fun=lambda window: (
                tf.math.reduce_mean(window[-int(len(window)/4):]) - tf.math.reduce_mean(window[:int(len(window)/4)])
            ),
            metric_op='<',
            dead_time=dead_time
        )


def est_sig_shared_dim(K,dim_num,N,det_type,Pfa=0.001):

    # K: estimated canonical correlations, type = numpy array 
    # dim_num: shared dimension to test for significance
    # N: number of samples
    # det_type: detector_1 or detector_2
    # Pfa: probability of false alarm

    tot_dim = K.shape[0] # assuming both data sets are of same size, change if they are of unequal size
    s = dim_num-1 # counter, checking hypothesis if num_corr_comp = s or num_corr_comp > s 
    if det_type == 'detector 1':
        BL_statistic = (-1 * (N - s - (tot_dim + tot_dim + 1) / 2 + np.sum(K[:s] ** (-2))) * 
            np.log(np.prod(1 - K[s:] ** 2)))

        dof = (tot_dim-s)*(tot_dim-s)

        T = chi2.ppf(1 - Pfa, df=dof)

        if BL_statistic < T:
            return False
        else:
            return True
    elif det_type == 'detector 2':
        
        GLR_statistic = N / 2 * np.log(np.prod(1 - K[s:] ** 2))
        T = - np.log(N) / 2 * (tot_dim - s) * (tot_dim - s)
        
        if GLR_statistic > T:
            return False
        else:
            return True