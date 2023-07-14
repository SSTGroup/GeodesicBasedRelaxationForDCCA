import os
import pickle as pkl
from abc import ABC

import numpy as np
import tensorflow as tf
from scipy.optimize import minimize
from tqdm.auto import tqdm

from GeodesicRelaxationDCCA.algorithms.correlation_residual import (
    canonical_correlations,
    chordal_distance,
    geodesic,
    log_map,
    principal_angles,
    rotate_on_manifold,
)
from GeodesicRelaxationDCCA.algorithms.losses_metrics import EmptyWatchdog, MetricDict, get_rec_loss
from GeodesicRelaxationDCCA.algorithms.tf_summary import TensorboardWriter
from GeodesicRelaxationDCCA.algorithms.utils import logdir_update_from_params
from GeodesicRelaxationDCCA.architecture.encoder import MVEncoder, MVResidualSlackEncoder



class Experiment(ABC):
    """
    Experiment meta class
    """

    def __init__(
        self,
        architecture,
        dataprovider,
        shared_dim,
        optimizer,
        log_dir,
        summary_writer,
        eval_epochs,
        watchdog,
        val_default_value=0.0,
        convergence_threshold=0.001,
    ):
        self.architecture = architecture
        self.dataprovider = dataprovider
        self.optimizer = optimizer
        self.summary_writer = self.create_summary_writer(summary_writer, log_dir)
        self.log_dir = self.summary_writer.dir
        self.shared_dim = shared_dim
        self.watchdog = watchdog
        self.moving_metrics = self.get_moving_metrics()
        self.eval_epochs = eval_epochs

        self.epoch = 1
        self.continue_training = True
        self.best_val = val_default_value
        self.best_val_view0 = val_default_value
        self.best_val_view1 = val_default_value
        self.best_val_view2 = val_default_value
        self.best_val_avg = val_default_value
        # Convergence criteria
        self.loss_threshold = convergence_threshold
        self.prev_loss = 1e5
        self.prev_epoch = 0

    def get_moving_metrics(self):
        raise NotImplementedError

    def create_summary_writer(self, summary_writer, log_dir):
        # Define writer for tensorboard summary
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        return summary_writer(log_dir)

    def train_multiple_epochs(self, num_epochs):
        # Iterate over epochs
        for epoch in tqdm(range(num_epochs), desc="Epochs"):
            # Train one epoch
            self.train_single_epoch()

            if not self.continue_training:
                break

        self.save_weights("latest")

    def save_weights(self, subdir=None):
        if subdir is not None:
            save_path = os.path.join(self.log_dir, subdir)
            if not os.path.exists(save_path):
                os.mkdir(save_path)
        else:
            save_path = self.log_dir

        self.architecture.save_weights(filepath=save_path)

    def load_weights_from_log(self, subdir=None):
        if subdir is not None:
            save_path = os.path.join(self.log_dir, subdir)
        else:
            save_path = self.log_dir

        self.architecture.load_weights(filepath=save_path)

    def load_best(self):
        self.architecture.load_weights(filepath=self.log_dir)

    def train_single_epoch(self):
        for data in self.dataprovider.training_data:
            with tf.GradientTape() as tape:
                # Feed forward
                network_output = self.architecture(inputs=data, training=True)
                # Compute loss
                loss = self.compute_loss(network_output, data)

            # Compute gradients
            gradients = tape.gradient(loss, self.architecture.trainable_variables)
            # Apply gradients
            self.optimizer.apply_gradients(zip(gradients, self.architecture.trainable_variables))

        # Write metric summary
        self.log_metrics()

        # Increase epoch counter
        self.epoch += 1

        if self.epoch % 100 == 0:
            self.prev_loss = self.prev_loss + 1e5
            next_loss = loss + 1e5
            if tf.abs(self.prev_loss - next_loss) < self.loss_threshold:
                # self.shared_dim += 1
                # self.architecture.update_num_shared_dim(self.shared_dim)
                self.continue_training = False
            self.prev_loss = loss
            self.prev_epoch = self.epoch

    def predict(self, data_to_predict):
        outputs = MetricDict()
        for data in data_to_predict:
            network_output = self.architecture(inputs=data, training=True)
            outputs.update(network_output)

        return outputs.output()

    def save(self):
        self.architecture.save(self.log_dir)

    def compute_loss(self, network_output, data):
        raise NotImplementedError

    def log_metrics(self):
        raise NotImplementedError


class DeepCCAExperiment(Experiment):
    """
    Experiment class for DeepCCA
    """

    def __init__(
        self,
        log_dir,
        encoder_config_v1,
        encoder_config_v2,
        dataprovider,
        shared_dim,
        lambda_rad,
        topk,
        max_perc=1,
        lambda_l1=0,
        lambda_l2=0,
        cca_reg=0,
        eval_epochs=10,
        optimizer=None,
        val_default_value=0.0,
        convergence_threshold=0.001,
        watchdog=None,
    ):
        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam()

        if watchdog is None:
            watchdog = EmptyWatchdog()

        architecture = MVEncoder(
            encoder_config_v1=encoder_config_v1,
            encoder_config_v2=encoder_config_v2,
            cca_reg=cca_reg,
            num_shared_dim=shared_dim,
        )

        log_dir = logdir_update_from_params(
            log_dir=log_dir,
            shared_dim=shared_dim,
            num_neurons=encoder_config_v1[0][0],
            lambda_l1=lambda_l1,
            lambda_l2=lambda_l2,
            lambda_rad=lambda_rad,
            topk=topk,
        )

        super(DeepCCAExperiment, self).__init__(
            architecture=architecture,
            dataprovider=dataprovider,
            shared_dim=shared_dim,
            optimizer=optimizer,
            log_dir=log_dir,
            summary_writer=TensorboardWriter,
            eval_epochs=eval_epochs,
            watchdog=watchdog,
            val_default_value=val_default_value,
            convergence_threshold=convergence_threshold,
        )

        # Dimensions and lambdas
        self.shared_dim = shared_dim

        self.lambda_rad = lambda_rad
        self.topk = topk
        self.max_perc = max_perc
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2

    def compute_loss(self, network_output, data):
        # Compute CCA loss
        ccor = network_output["ccor"]
        cca_loss = -1 * tf.reduce_sum(ccor) / len(ccor)

        l1_loss = self.architecture.get_l1()
        l1_loss *= self.lambda_l1
        l2_loss = self.architecture.get_l2()
        l2_loss *= self.lambda_l2

        loss = cca_loss + l1_loss + l2_loss

        if self.epoch % 10 == 0:
            self.summary_writer.write_scalar_summary(
                epoch=self.epoch,
                list_of_tuples=[
                    (loss, "Loss/Total"),
                    (cca_loss, "Loss/CCA"),
                    (l1_loss, "Loss/L1"),
                    (l2_loss, "Loss/L2"),
                    (self.shared_dim, "MovingMean/Dimensions"),
                ],
            )
        return loss


class DeepCCAResidualSlackExperiment(Experiment):
    """
    Experiment class for ResiualDeepCCA, proposed implementation
    """

    def __init__(
        self,
        log_dir,
        encoder_config_v1,
        encoder_config_v2,
        dataprovider,
        shared_dim,
        residual,
        corr_reg,
        lambda_l1=0,
        lambda_l2=0,
        eval_epochs=10,
        optimizer=None,
        val_default_value=0.0,
        convergence_threshold=0.001,
    ):
        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam()

        architecture = MVResidualSlackEncoder(
            encoder_config_v1=encoder_config_v1,
            encoder_config_v2=encoder_config_v2,
            num_shared_dim=shared_dim,
            corr_reg=corr_reg,
        )

        log_dir = logdir_update_from_params(
            log_dir=log_dir,
            shared_dim=shared_dim,
            num_neurons=encoder_config_v1[0][0],
            lambda_l1=lambda_l1,
            lambda_l2=lambda_l2,
            residual=residual,
        )

        super(DeepCCAResidualSlackExperiment, self).__init__(
            architecture=architecture,
            dataprovider=dataprovider,
            shared_dim=shared_dim,
            optimizer=optimizer,
            log_dir=log_dir,
            summary_writer=TensorboardWriter,
            eval_epochs=eval_epochs,
            watchdog=EmptyWatchdog(),
            val_default_value=val_default_value,
            convergence_threshold=convergence_threshold,
        )

        # Ux and Uy
        v1_data_dim = encoder_config_v1[-1][0]
        v2_data_dim = encoder_config_v2[-1][0]
        batch_size = dataprovider.batch_size

        self.U1 = tf.Variable(tf.zeros((batch_size, v1_data_dim)), trainable=False)
        self.U2 = tf.Variable(tf.zeros((batch_size, v2_data_dim)), trainable=False)

        # Dimensions and lambdas
        self.shared_dim = shared_dim
        self.residual = residual
        self.corr_reg = corr_reg
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2

    def save_weights(self, subdir=None):
        if subdir is not None:
            save_path = os.path.join(self.log_dir, subdir)
            if not os.path.exists(save_path):
                os.mkdir(save_path)
        else:
            save_path = self.log_dir

        self.architecture.save_weights(filepath=save_path)
        cca_file = os.path.join(save_path, "CCA_variables.pkl")
        with open(cca_file, "wb") as f:
            pkl.dump(dict(U1=self.U1, U2=self.U2), f)

    def load_weights_from_log(self, subdir=None):
        if subdir is not None:
            save_path = os.path.join(self.log_dir, subdir)
        else:
            save_path = self.log_dir

        cca_file = os.path.join(save_path, "CCA_variables.pkl")
        with open(cca_file, "rb") as f:
            CCA_dict = pkl.load(f)
            U1 = CCA_dict["U1"]
            U2 = CCA_dict["U2"]

        self.architecture.load_weights(filepath=save_path)
        self.U1.assing(U1)
        self.U2.assing(U2)

    # Operations for computing U, U_x and U_y
    def update_U(self, view1, view2, num_shared_dim):
        assert view1.shape[1] == view2.shape[1]
        concatenation = np.concatenate([view1, view2], axis=0)
        D, P, Q = tf.linalg.svd(concatenation, full_matrices=False, compute_uv=True)
        U = tf.transpose(Q)[0:num_shared_dim, :]
        return U

    def compute_residual_vector(self, U_orthonormal, data1_orthonormal, data2_orthonormal, target_residual):
        def compute_Ux_Uy(t):
            U_x = rotate_on_manifold(U_orthonormal, data1_orthonormal, t=t)
            U_y = rotate_on_manifold(U_orthonormal, data2_orthonormal, t=t)
            return U_x, U_y

        def compute_residual_error(t):
            U_x, U_y = compute_Ux_Uy(t)
            residual = chordal_distance(U_x, U_y, corr_reg=self.corr_reg)
            error = tf.math.square(target_residual - residual)
            return error

        res = minimize(compute_residual_error, [1], method="Nelder-Mead", tol=1e-5)
        t = res.x
        U_x, U_y = compute_Ux_Uy(t=t)

        return U_x, U_y

    # Function to compute the loss for a shared U
    def compute_loss(self, Ux, Uy, X, Y):
        d1 = chordal_distance(Ux, X, corr_reg=self.corr_reg)
        d2 = chordal_distance(Uy, Y, corr_reg=self.corr_reg)
        distance_loss = tf.reduce_mean([d1, d2])

        assert not tf.math.is_nan(distance_loss)

        l1_loss = self.architecture.get_l1()
        l1_loss *= self.lambda_l1
        l2_loss = self.architecture.get_l2()
        l2_loss *= self.lambda_l2

        loss = distance_loss + l1_loss + l2_loss

        if self.epoch % 10 == 0:
            self.summary_writer.write_scalar_summary(
                epoch=self.epoch,
                list_of_tuples=[
                    (loss, "Loss/Total"),
                    (distance_loss, "Loss/Distance"),
                    (l1_loss, "Loss/L1"),
                    (l2_loss, "Loss/L2"),
                    (self.shared_dim, "MovingMean/Dimensions"),
                ],
            )
        return loss

    # Loop for inner optimization
    def optimize(self, num_steps, Ux, Uy, data, epsilon_inner_epochs):
        last_loss = tf.float32.max
        for i in range(num_steps):
            with tf.GradientTape() as tape:
                # Compute orthonormal data
                network_output = self.architecture(data)
                data1_orthonormal = network_output["rrcca_view_0"]
                data2_orthonormal = network_output["rrcca_view_1"]

                # Compute loss
                loss = self.compute_loss(Ux, Uy, data1_orthonormal, data2_orthonormal)

            # Compute gradients
            gradients = tape.gradient(loss, self.architecture.trainable_variables)
            assert not tf.reduce_any([tf.reduce_any(tf.math.is_nan(grad)) for grad in gradients])
            # Apply gradients
            self.optimizer.apply_gradients(zip(gradients, self.architecture.trainable_variables))

            if last_loss - loss < epsilon_inner_epochs:
                break
            last_loss = loss

        # Write metric summary
        self.log_metrics()

        # Increase epoch counter
        self.epoch += 1

    def train_multiple_epochs(self, num_epochs, num_inner_epochs, epsilon_inner_epochs):
        # Load training data once
        training_data = self.dataprovider.training_data

        # We just work with single-batch data
        for data in training_data:
            pass

        for epoch in tqdm(range(num_epochs), leave=False, desc="Epochs"):
            network_output = self.architecture(data)
            data1_orthonormal = network_output["rrcca_view_0"]
            data2_orthonormal = network_output["rrcca_view_1"]

            U_orthonormal = self.update_U(data1_orthonormal, data2_orthonormal, num_shared_dim=self.shared_dim)

            U_x, U_y = self.compute_residual_vector(U_orthonormal, data1_orthonormal, data2_orthonormal, self.residual)

            self.optimize(num_inner_epochs, U_x, U_y, data, epsilon_inner_epochs)

        self.save_weights("latest")
