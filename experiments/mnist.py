import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from GeodesicRelaxationDCCA.algorithms.clustering import kmeans_clustering_acc
from GeodesicRelaxationDCCA.algorithms.losses_metrics import MetricDict, MovingMetric
from GeodesicRelaxationDCCA.algorithms.correlation_residual import canonical_correlations
from GeodesicRelaxationDCCA.experiments.evaluation import Evaluation
from GeodesicRelaxationDCCA.experiments.template import (
    DeepCCAExperiment,
    DeepCCAResidualSlackExperiment,
    Experiment,
)
from sklearn.manifold import TSNE
from tqdm.auto import tqdm


class MNISTExperiment(Experiment):
    def get_moving_metrics(self):
        cor_movmetr = {
            "cor_" + str(num): MovingMetric(window_length=5, history_length=10, fun=tf.math.reduce_mean)
            for num in range(self.shared_dim)
        }

        acc_movmetr = {
            "acc_v0": MovingMetric(window_length=5, history_length=10, fun=tf.math.reduce_mean),
            "acc_v1": MovingMetric(window_length=5, history_length=10, fun=tf.math.reduce_mean),
        }

        return {**cor_movmetr, **acc_movmetr}

    def log_metrics(self):
        self.watchdog.decrease_counter()

        if self.epoch % 10 == 0:
            # Compute correlation values on training data
            training_outp = self.predict(self.dataprovider.training_data)
            ccor = training_outp["ccor"]

            l1 = self.architecture.get_l1()
            l2 = self.architecture.get_l2()

            self.summary_writer.write_scalar_summary(
                epoch=self.epoch,
                list_of_tuples=[(ccor[i], "Correlations/" + str(i)) for i in range(self.shared_dim)]
                + [
                    (l1, "Regularization/L1"),
                    (l2, "Regularization/L2"),
                ],
            )

        if self.epoch % self.eval_epochs == 0:
            acc_v0 = self.compute_clustering_accuracy(view="view0", split="eval")
            self.moving_metrics["acc_v0"].update_window(acc_v0)
            smoothed_acc_v0 = self.moving_metrics["acc_v0"].get_metric()

            acc_v1 = self.compute_clustering_accuracy(view="view1", split="eval")
            self.moving_metrics["acc_v1"].update_window(acc_v1)
            smoothed_acc_v1 = self.moving_metrics["acc_v1"].get_metric()

            acc_avg = (acc_v0 + acc_v1) / 2
            smoothed_acc_avg = (smoothed_acc_v0 + smoothed_acc_v1) / 2

            if smoothed_acc_v0 > self.best_val_view0:
                self.best_val_view0 = smoothed_acc_v0
                self.save_weights(subdir="view0")

            if smoothed_acc_v1 > self.best_val_view1:
                self.best_val_view1 = smoothed_acc_v1
                self.save_weights(subdir="view1")

            if smoothed_acc_avg > self.best_val_avg:
                self.best_val_avg = smoothed_acc_avg
                self.save_weights(subdir="avg")

            self.summary_writer.write_scalar_summary(
                epoch=self.epoch,
                list_of_tuples=[
                    (acc_v0, "Accuracy/View0"),
                    (acc_v1, "Accuracy/View1"),
                    (acc_avg, "Accuracy/Average"),
                    (smoothed_acc_v0, "AccuracySmoothed/View0"),
                    (smoothed_acc_v1, "AccuracySmoothed/View1"),
                    (smoothed_acc_avg, "AccuracySmoothed/Average"),
                ],
            )

    def compute_clustering_accuracy(self, view="view0", split="eval"):
        assert view in ["view0", "view1"]
        assert split in ["eval", "test"]
        if split == "eval":
            data_for_acc = self.dataprovider.eval_data
        else:
            data_for_acc = self.dataprovider.test_data

        outputs_met, labels_met = MetricDict(), MetricDict()
        for data in data_for_acc:
            outputs_met.update(self.architecture(inputs=data, training=False))
            labels_met.update(dict(labels=data["labels"].numpy()))

        netw_output = outputs_met.output()
        labels = labels_met.output()["labels"]

        if view == "view0":
            latent_repr = netw_output["cca_view_0"]
        elif view == "view1":
            latent_repr = netw_output["cca_view_1"]

        return kmeans_clustering_acc(data_points=latent_repr, labels=labels, num_classes=self.dataprovider.num_classes)

    def visualize_subspace(self, data_split="eval"):
        assert data_split in ["eval", "test"]
        if data_split == "eval":
            data_for_vis = self.dataprovider.eval_data
        else:
            data_for_vis = self.dataprovider.test_data

        outputs_met = MetricDict()
        for data in data_for_vis:
            outputs_met.update(self.architecture(inputs=data, training=False))

        netw_output = outputs_met.output()

        embedding_0_net = TSNE(
            n_components=2, learning_rate="auto", init="random", verbose=False, n_jobs=4
        ).fit_transform(netw_output["cca_view_0"])

        embedding_1_net = TSNE(
            n_components=2, learning_rate="auto", init="random", verbose=False, n_jobs=4
        ).fit_transform(netw_output["cca_view_1"])

        fig, ax = plt.subplots(2, 1, figsize=(10, 20))
        ax[0].scatter(
            embedding_0_net[:, 0], embedding_0_net[:, 1], c=self.dataprovider.view1_eval[1], cmap=plt.cm.tab10
        )
        ax[1].scatter(
            embedding_1_net[:, 0], embedding_1_net[:, 1], c=self.dataprovider.view2_eval[1], cmap=plt.cm.tab10
        )


class MNISTDeepCCAExperiment(DeepCCAExperiment, MNISTExperiment):
    pass


class MNISTDeepCCASlackExperiment(DeepCCAResidualSlackExperiment, MNISTExperiment):
    def log_metrics(self):
        self.watchdog.decrease_counter()

        if self.epoch % 10 == 0:
            # Compute correlation values on training data
            training_outp = self.predict(self.dataprovider.training_data)
            ccor = training_outp["ccor"]
            
            l1 = self.architecture.get_l1()
            l2 = self.architecture.get_l2()

            self.summary_writer.write_scalar_summary(
                epoch=self.epoch,
                list_of_tuples=[(ccor[i], "Correlations/" + str(i)) for i in range(self.shared_dim)]
                + [
                    (l1, "Regularization/L1"),
                    (l2, "Regularization/L2"),
                ],
            )

        if self.epoch % self.eval_epochs == 0:
            acc_v0 = self.compute_clustering_accuracy(view="view0", split="eval")
            self.moving_metrics["acc_v0"].update_window(acc_v0)
            smoothed_acc_v0 = self.moving_metrics["acc_v0"].get_metric()

            acc_v1 = self.compute_clustering_accuracy(view="view1", split="eval")
            self.moving_metrics["acc_v1"].update_window(acc_v1)
            smoothed_acc_v1 = self.moving_metrics["acc_v1"].get_metric()

            acc_avg = (acc_v0 + acc_v1) / 2
            smoothed_acc_avg = (smoothed_acc_v0 + smoothed_acc_v1) / 2

            if smoothed_acc_v0 > self.best_val_view0:
                self.best_val_view0 = smoothed_acc_v0
                self.save_weights(subdir="view0")

            if smoothed_acc_v1 > self.best_val_view1:
                self.best_val_view1 = smoothed_acc_v1
                self.save_weights(subdir="view1")

            if smoothed_acc_avg > self.best_val_avg:
                self.best_val_avg = smoothed_acc_avg
                self.save_weights(subdir="avg")

            self.summary_writer.write_scalar_summary(
                epoch=self.epoch,
                list_of_tuples=[
                    (acc_v0, "Accuracy/View0"),
                    (acc_v1, "Accuracy/View1"),
                    (acc_avg, "Accuracy/Average"),
                    (smoothed_acc_v0, "AccuracySmoothed/View0"),
                    (smoothed_acc_v1, "AccuracySmoothed/View1"),
                    (smoothed_acc_avg, "AccuracySmoothed/Average"),
                ],
            )

    def compute_clustering_accuracy(self, view="view0", split="eval"):
        assert view in ["view0", "view1"]
        assert split in ["eval", "test"]
        if split == "eval":
            data_for_acc = self.dataprovider.eval_data
        else:
            data_for_acc = self.dataprovider.test_data

        outputs_met, labels_met = MetricDict(), MetricDict()
        for data in data_for_acc:
            outputs_met.update(self.architecture(inputs=data, training=False))
            labels_met.update(dict(labels=data["labels"].numpy()))

        netw_output = outputs_met.output()
        labels = labels_met.output()["labels"]

        if view == "view0":
            latent_repr = tf.transpose(netw_output["rrcca_view_0"])
        elif view == "view1":
            latent_repr = tf.transpose(netw_output["rrcca_view_1"])

        return kmeans_clustering_acc(data_points=latent_repr, labels=labels, num_classes=self.dataprovider.num_classes)

    def pretrain(self, num_epochs):
        # Use encoder from architecture
        input_layer_v0 = tf.keras.layers.Input(shape=(784,))
        encoder_output_v0 = self.architecture.encoder_v0(input_layer_v0)

        input_layer_v1 = tf.keras.layers.Input(shape=(784,))
        encoder_output_v1 = self.architecture.encoder_v1(input_layer_v1)
        # Add decoder for pre-training
        decoder_model_v0 = tf.keras.models.Sequential()
        decoder_model_v0.add(tf.keras.layers.Dense(784, activation="sigmoid"))
        output_decoder_v0 = decoder_model_v0(encoder_output_v0)
        autoencoder_model_v0 = tf.keras.Model(input_layer_v0, output_decoder_v0)

        decoder_model_v1 = tf.keras.models.Sequential()
        decoder_model_v1.add(tf.keras.layers.Dense(784, activation="sigmoid"))
        output_decoder_v1 = decoder_model_v1(encoder_output_v1)
        autoencoder_model_v1 = tf.keras.Model(input_layer_v1, output_decoder_v1)

        # Load data
        data_for_pretraining = self.dataprovider.training_data
        for data in data_for_pretraining:
            pass

        pretrain_optimizer = tf.keras.optimizers.Adam()

        with tqdm(range(num_epochs)) as pbar:
            for i in pbar:
                with tf.GradientTape() as tape:
                    reconst_v0 = autoencoder_model_v0(data["nn_input_0"])
                    reconst_v1 = autoencoder_model_v1(data["nn_input_1"])
                    loss = tf.linalg.norm(reconst_v0 - tf.cast(data["nn_input_0"], tf.float32)) + tf.linalg.norm(
                        reconst_v1 - tf.cast(data["nn_input_1"], tf.float32)
                    )

                # Compute gradients
                gradients = tape.gradient(
                    loss, autoencoder_model_v0.trainable_variables + autoencoder_model_v1.trainable_variables
                )
                # Apply gradients
                pretrain_optimizer.apply_gradients(
                    zip(gradients, autoencoder_model_v0.trainable_variables + autoencoder_model_v1.trainable_variables)
                )

                pbar.set_postfix_str(str(loss.numpy()))


class MNISTEvaluation(Evaluation):
    def eval(self, model, split="test"):
        metrics = dict()

        assert split in ["eval", "test", "train"]
        if split == "eval":
            data_split = self.dataprovider.eval_data
        elif split == "test":
            data_split = self.dataprovider.test_data
        elif split == "train":
            data_split = self.dataprovider.training_data

        outputs_met, labels_met = MetricDict(), MetricDict()
        for data in data_split:
            outputs_met.update(model(inputs=data, training=False))
            labels_met.update(dict(labels=data["labels"].numpy()))

        netw_output = outputs_met.output()
        labels = labels_met.output()["labels"]

        if "cca_view_0" in netw_output.keys():
            data_points_v0 = netw_output["cca_view_0"]
            data_points_v1 = netw_output["cca_view_1"]
        elif "rrcca_view_0" in netw_output.keys():
            data_points_v0 = tf.transpose(netw_output["rrcca_view_0"])
            data_points_v1 = tf.transpose(netw_output["rrcca_view_1"])

        metrics["acc_v0"] = kmeans_clustering_acc(
            data_points=data_points_v0, labels=labels, num_classes=self.dataprovider.num_classes
        )

        metrics["acc_v1"] = kmeans_clustering_acc(
            data_points=data_points_v1, labels=labels, num_classes=self.dataprovider.num_classes
        )

        metrics["acc_avg"] = (metrics["acc_v0"] + metrics["acc_v1"]) / 2

        ccor = canonical_correlations(data_points_v0, data_points_v1, data_points_v0.shape[1], 1e-4)

        metrics['avg_ccor'] = tf.reduce_mean(ccor)
        metrics['max_ccor'] = tf.reduce_max(ccor)

        return metrics
