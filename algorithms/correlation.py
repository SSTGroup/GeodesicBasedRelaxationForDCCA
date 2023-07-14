import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def CCA(view1, view2, num_shared_dim, r1=0, r2=0):
    V1 = tf.cast(view1, dtype=tf.float32)
    V2 = tf.cast(view2, dtype=tf.float32)

    assert V1.shape[0] == V2.shape[0]
    M = tf.cast(tf.shape(V1)[0], dtype=tf.float32)
    ddim_1 = tf.constant(V1.shape[1], dtype=tf.int16)
    ddim_2 = tf.constant(V2.shape[1], dtype=tf.int16)

    # check mean and variance
    mean_V1 = tf.reduce_mean(V1, 0)
    mean_V2 = tf.reduce_mean(V2, 0)

    V1_bar = tf.subtract(V1, tf.tile(tf.convert_to_tensor(mean_V1)[None], [M, 1]))
    V2_bar = tf.subtract(V2, tf.tile(tf.convert_to_tensor(mean_V2)[None], [M, 1]))

    Sigma12 = tf.linalg.matmul(tf.transpose(V1_bar), V2_bar) / (M - 1)
    Sigma11 = tf.linalg.matmul(tf.transpose(V1_bar), V1_bar) / (M - 1) + r1 * tf.eye(ddim_1)
    Sigma22 = tf.linalg.matmul(tf.transpose(V2_bar), V2_bar) / (M - 1) + r2 * tf.eye(ddim_2)

    Sigma11_root_inv = tf.linalg.sqrtm(tf.linalg.inv(Sigma11))
    Sigma22_root_inv = tf.linalg.sqrtm(tf.linalg.inv(Sigma22))
    Sigma22_root_inv_T = tf.transpose(Sigma22_root_inv)

    C = tf.linalg.matmul(tf.linalg.matmul(Sigma11_root_inv, Sigma12), Sigma22_root_inv_T)
    D, U, V = tf.linalg.svd(C, full_matrices=False)

    A = tf.matmul(tf.transpose(U)[:num_shared_dim], Sigma11_root_inv)
    B = tf.matmul(tf.transpose(V)[:num_shared_dim], Sigma22_root_inv)

    epsilon = tf.matmul(A, tf.transpose(V1_bar))
    omega = tf.matmul(B, tf.transpose(V2_bar))

    return A, B, epsilon, omega, D[:num_shared_dim], mean_V1, mean_V2
