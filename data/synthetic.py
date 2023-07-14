import os
from tkinter import scrolledtext
import numpy as np
import tensorflow as tf
import pickle as pkl


def _sigmoid(x):
    return 1/(1 + np.exp(-x))

def _generate_Gaussian_data(num_samples, distinct_correlations, num_channels,non_lin_type):
    # Mixing matrices
    A_1 = np.random.normal(size=(num_channels, num_channels))
    A_2 = np.random.normal(size=(num_channels, num_channels))

    z_1 = np.empty(shape=(num_channels, num_samples))
    z_2 = np.empty(shape=(num_channels, num_samples))


    for index, rho in enumerate(distinct_correlations):
        mean = np.array([0, 0])

        cov = np.array([[1, rho],
                        [rho, 1]])

        bivariate_sample = np.random.multivariate_normal(mean, cov, size=num_samples,
                                                        check_valid='warn',
                                                        tol=1e-10).T

        z_1[index] = bivariate_sample[0]
        z_2[index] = bivariate_sample[1]

    Az_1 = np.matmul(A_1, z_1)
    Az_2 = np.matmul(A_2, z_2)
    
    # Add non-linearities
    y_1 = np.zeros_like(Az_1)
    y_2 = np.zeros_like(Az_2)
    
    if non_lin_type=='channel_wise':
        for i in range(num_channels):
            # Channel-wise nonlinearity
            y_1[i] = 0.2*_sigmoid(Az_1[i]) + (Az_1[i])**3
            y_2[i] = np.tanh(Az_2[i]) + 0.2*np.exp(Az_2[i])
    elif non_lin_type=='not_channel_wise':
            # Not channel-wise nonlinearity
            y_1[0] =  _sigmoid(Az_1[0])* Az_1[1] 
            y_1[1] =  Az_1[2] * _sigmoid(Az_1[3]) 
            y_1[2] =  _sigmoid(Az_1[4]) * Az_1[1]
            y_1[3] =  Az_1[0]*Az_1[1]*Az_1[2] 
            y_1[4] =  (Az_1[4])**1 * Az_1[1] #**3

            y_2[0] =  (Az_2[2])**1 * Az_2[3] #**3
            y_2[1] =  np.tanh(Az_2[4]) * Az_2[1] 
            y_2[2] =  Az_2[0] * (Az_2[2])**1 #**3
            y_2[3] =  Az_2[3] * Az_2[4] 
            y_2[4] =  Az_2[0]* Az_2[2] * Az_2[3] 
        
    return tf.convert_to_tensor(y_1.T, dtype=tf.float32), tf.convert_to_tensor(y_2.T, dtype=tf.float32), \
        tf.convert_to_tensor(Az_1.T, dtype=tf.float32), tf.convert_to_tensor(Az_2.T, dtype=tf.float32), \
        tf.convert_to_tensor(z_1.T, dtype=tf.float32), tf.convert_to_tensor(z_2.T, dtype=tf.float32)

class SyntheticData:
    def __init__(
            self, 
            num_samples, 
            batch_size,
            num_batches,
            correlations,
            num_channels,
            non_lin_type,
            y_0, y_1, Az_0, Az_1, z_0, z_1,
            save_path=None):
        
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.correlations = correlations
        self.num_channels = num_channels
        self.non_lin_type = non_lin_type
        self.dim_samples = num_channels
        self.true_dim = np.sum([corr > 0 for corr in correlations])
        self.y_0 = y_0
        self.y_1 = y_1
        self.Az_0 = Az_0
        self.Az_1 = Az_1
        self.z_0 = z_0
        self.z_1 = z_1
        self.path = save_path

        self.dataset = self.get_dataset()

        self.training_data = self.dataset.take(self.num_batches)

    @classmethod
    def generate(cls, num_samples, batch_size, correlations, num_channels,non_lin_type):
        num_batches = np.floor(num_samples/batch_size).astype(int)
        num_samples = num_batches * batch_size

        y_0, y_1, Az_0, Az_1, z_0, z_1 = _generate_Gaussian_data(
            num_samples=num_samples,
            distinct_correlations=correlations,
            num_channels=num_channels,
            non_lin_type=non_lin_type
        )

        return cls(
            num_samples=num_samples, 
            batch_size=batch_size,
            num_batches=num_batches,
            correlations=correlations,
            num_channels=num_channels,
            non_lin_type=non_lin_type,
            y_0=y_0, y_1=y_1, Az_0=Az_0, Az_1=Az_1, z_0=z_0, z_1=z_1
        )

    @classmethod
    def from_saved_data(cls, save_path):
        with open(os.path.join(save_path, 'data.pkl'), 'rb') as f:
            save_dict = pkl.load(f)

        return cls(
            num_samples=save_dict['num_samples'], 
            batch_size=save_dict['batch_size'],
            num_batches=save_dict['num_batches'],
            correlations=save_dict['correlations'],
            num_channels=save_dict['num_channels'],
            non_lin_type=save_dict['non_lin_type'],
            y_0=save_dict['y_0'], y_1=save_dict['y_1'], 
            Az_0=save_dict['Az_0'], Az_1=save_dict['Az_1'], 
            z_0=save_dict['z_0'], z_1=save_dict['z_1']
        )
        
    def save(self, dir):
        save_dict = {
            'num_samples' : self.num_samples,
            'num_batches' :  self.num_batches,
            'batch_size' : self.batch_size,
            'correlations' : self.correlations,
            'num_channels' : self.num_channels,
            'non_lin_type' : self.non_lin_type,
            'y_0' : self.y_0,
            'y_1' : self.y_1,
            'Az_0' : self.Az_0,
            'Az_1' : self.Az_1,
            'z_0' : self.z_0,
            'z_1' : self.z_1
        }

        with open(os.path.join(dir, 'data.pkl'), 'wb') as f:
            pkl.dump(save_dict, f)

    def get_dataset(self):
        # Create dataset from numpy array in dict
        dataset = tf.data.Dataset.from_tensor_slices(
            {
                'nn_input_0': self.y_0,
                'nn_input_1': self.y_1,
                'Az_0': self.Az_0,
                'Az_1': self.Az_1,
                'z_0': self.z_0,
                'z_1': self.z_1
            }
        )

        # Batch
        dataset = dataset.batch(self.batch_size)

        return dataset

