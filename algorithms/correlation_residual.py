import numpy as np
import tensorflow as tf
from scipy.optimize import minimize
from tensorflow.python.framework import function
from tqdm.auto import tqdm


# Correlation operations
def canonical_correlations(view1, view2, num_shared_dim, regularization=0.0):
    V1 = tf.cast(view1, dtype=tf.float32)
    V2 = tf.cast(view2, dtype=tf.float32)

    assert V1.shape[0] == V2.shape[0]
    M = tf.cast(tf.shape(V1)[0], dtype=tf.float32)
    ddim_1 = tf.constant(V1.shape[1], dtype=tf.int16)
    ddim_2 = tf.constant(V2.shape[1], dtype=tf.int16)

    mean_V1 = tf.reduce_mean(V1, 0)
    mean_V2 = tf.reduce_mean(V2, 0)

    V1_bar = tf.subtract(V1, tf.tile(tf.convert_to_tensor(mean_V1)[None], [M, 1]))
    V2_bar = tf.subtract(V2, tf.tile(tf.convert_to_tensor(mean_V2)[None], [M, 1]))

    Sigma12 = tf.linalg.matmul(tf.transpose(V1_bar), V2_bar) / (M - 1)
    Sigma11 = tf.linalg.matmul(tf.transpose(V1_bar), V1_bar) / (M - 1) + regularization * tf.eye(ddim_1)
    Sigma22 = tf.linalg.matmul(tf.transpose(V2_bar), V2_bar) / (M - 1) + regularization * tf.eye(ddim_2)

    Sigma11_root_inv = tf.linalg.sqrtm(tf.linalg.inv(Sigma11))
    Sigma22_root_inv = tf.linalg.sqrtm(tf.linalg.inv(Sigma22))
    Sigma22_root_inv_T = tf.transpose(Sigma22_root_inv)

    # assert not tf.reduce_any(tf.math.is_nan(Sigma11_root_inv))
    # assert not tf.reduce_any(tf.math.is_nan(Sigma22_root_inv))

    C = tf.linalg.matmul(tf.linalg.matmul(Sigma11_root_inv, Sigma12), Sigma22_root_inv_T)
    D, U, V = tf.linalg.svd(C, full_matrices=False)

    return D[:num_shared_dim]


def principal_angles(X, Y, corr_reg):
    r = min(X.shape[0], Y.shape[0])
    ccor = canonical_correlations(tf.transpose(X), tf.transpose(Y), r, corr_reg)

    # Ensure values are not smaller than 0 or greater than 1
    # This can happen due to floating point inaccuracy
    ccor = tf.clip_by_value(ccor, clip_value_min=0.00001, clip_value_max=0.99999)
    assert not tf.reduce_any(ccor > 1)

    princip_ang = tf.math.acos(ccor)
    return princip_ang


@tf.custom_gradient
def custom_norm(x, axis=None):
    y = tf.norm(x, axis=axis)

    def grad(dy):
        return dy * (x / (y + 1e-19))

    return y, grad


def chordal_distance(X, Y, corr_reg):
    sin_values = tf.math.sin(principal_angles(X, Y, corr_reg))
    dimensionality = tf.cast(X.shape[0], dtype=tf.float32)
    normed_distance = custom_norm(sin_values) / tf.sqrt(dimensionality)

    return normed_distance


# Grassmann manifold operations
def log_map(X, Y):
    In = tf.transpose(X) @ Y
    Inv = tf.linalg.inv(In)
    tmp = Y @ Inv - X @ In @ Inv
    eig_values, left_eig, right_eig = tf.linalg.svd(tmp, full_matrices=False)
    theta = tf.math.atan(eig_values)
    return tf.transpose((left_eig @ tf.linalg.diag(np.asarray(theta)) @ tf.transpose(right_eig)))


def geodesic(X, Z, t):
    eig_values, left_eig, right_eig = tf.linalg.svd(Z, full_matrices=False)
    tmp_1 = X @ right_eig @ tf.linalg.diag(tf.math.cos(t * eig_values)) @ tf.transpose(right_eig)
    tmp_2 = left_eig @ tf.linalg.diag(tf.math.sin(t * eig_values)) @ tf.transpose(right_eig)
    return tf.transpose(tmp_1 + tmp_2)


def rotate_on_manifold(X, Y, t):
    logmap = log_map(tf.transpose(X), tf.transpose(Y))
    return geodesic(tf.transpose(X), tf.transpose(logmap), t)


def RRCCA(
    view1,
    view2,
    num_shared_dim,
    residual,
    corr_reg,
    num_outer_loops=100,
    max_num_inner_loops=100,
    epsilon_inner_loops=1e-3,
):
    # Operations for computing U, U_x and U_y
    def update_U(view1, view2, num_shared_dim):
        assert view1.shape[1] == view2.shape[1]
        concatenation = np.concatenate([view1, view2], axis=0)
        D, P, Q = tf.linalg.svd(concatenation, full_matrices=False, compute_uv=True)
        U = tf.transpose(Q)[0:num_shared_dim, :]
        return U

    def compute_residual_vector(U_orthonormal, data1_orthonormal, data2_orthonormal, target_residual):
        def compute_Ux_Uy(t):
            U_x = rotate_on_manifold(U_orthonormal, data1_orthonormal, t=t)
            U_y = rotate_on_manifold(U_orthonormal, data2_orthonormal, t=t)
            return U_x, U_y

        def compute_residual_error(t):
            U_x, U_y = compute_Ux_Uy(t)
            residual = chordal_distance(U_x, U_y, corr_reg=corr_reg)
            error = tf.math.square(target_residual - residual)
            return error

        res = minimize(compute_residual_error, [1], method="Nelder-Mead", tol=1e-5)
        t = res.x
        U_x, U_y = compute_Ux_Uy(t=t)

        return U_x, U_y

    # Function to compute the loss for a shared U
    def compute_loss(Ux, Uy, X, Y):
        # squared_norm_2_X = tf.square(tf.linalg.norm(Ux - X, ord=2, axis=0))
        # squared_norm_2_Y = tf.square(tf.linalg.norm(Uy - Y, ord=2, axis=0))
        # return tf.reduce_mean(tf.add(squared_norm_2_X, squared_norm_2_Y))
        l1 = chordal_distance(Ux, X, corr_reg=corr_reg)
        l2 = chordal_distance(Uy, Y, corr_reg=corr_reg)
        return tf.reduce_mean([l1, l2])

    # Loop for optimization
    def optimize(num_steps, Ux, Uy, X, Y, Ax, Ay, epsilon_inner_loops):
        last_loss = tf.float32.max
        for i in range(num_steps):
            with tf.GradientTape() as tape:
                # Compute orthonormal data
                data1 = Ax @ tf.transpose(X)
                data2 = Ay @ tf.transpose(Y)

                eig_values, left_eig, right_eig = tf.linalg.svd(data1)
                data1_orthonormal = left_eig @ tf.transpose(right_eig)
                eig_values, left_eig, right_eig = tf.linalg.svd(data2)
                data2_orthonormal = left_eig @ tf.transpose(right_eig)

                # Compute loss
                loss = compute_loss(Ux, Uy, data1_orthonormal, data2_orthonormal)

            # Compute gradients
            gradients = tape.gradient(loss, [Ax, Ay])
            # Apply gradients
            optimizer.apply_gradients(zip(gradients, [Ax, Ay]))

            if last_loss - loss < epsilon_inner_loops:
                break
            last_loss = loss

        return loss

    # Define transformation matrices
    Ax, Ay = tf.Variable(np.eye(num_shared_dim, 4), dtype=tf.float32), tf.Variable(
        np.eye(num_shared_dim, 4), dtype=tf.float32
    )

    # Define optimizer
    optimizer = tf.keras.optimizers.Adam()

    for i in tqdm(range(num_outer_loops), leave=False):
        data1 = Ax @ tf.transpose(view1)
        data2 = Ay @ tf.transpose(view2)

        eig_values, left_eig, right_eig = tf.linalg.svd(data1)
        data1_orthonormal = left_eig @ tf.transpose(right_eig)
        eig_values, left_eig, right_eig = tf.linalg.svd(data2)
        data2_orthonormal = left_eig @ tf.transpose(right_eig)

        U_orthonormal = update_U(data1_orthonormal, data2_orthonormal, num_shared_dim=num_shared_dim)

        U_x, U_y = compute_residual_vector(U_orthonormal, data1_orthonormal, data2_orthonormal, residual)

        loss = optimize(max_num_inner_loops, U_x, U_y, view1, view2, Ax, Ay, epsilon_inner_loops)

    data1 = Ax @ tf.transpose(view1)
    data2 = Ay @ tf.transpose(view2)

    eig_values, left_eig, right_eig = tf.linalg.svd(data1)
    data1_orthonormal = left_eig @ tf.transpose(right_eig)
    eig_values, left_eig, right_eig = tf.linalg.svd(data2)
    data2_orthonormal = left_eig @ tf.transpose(right_eig)

    ccor = canonical_correlations(
        tf.transpose(data1_orthonormal), tf.transpose(data2_orthonormal), num_shared_dim, corr_reg
    )

    return Ax, Ay, data1_orthonormal, data2_orthonormal, ccor, U_x, U_y, U_orthonormal, loss
