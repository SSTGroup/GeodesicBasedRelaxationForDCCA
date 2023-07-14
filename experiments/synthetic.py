import os
import pickle as pkl

import tensorflow as tf
from GeodesicRelaxationDCCA.algorithms.correlation import CCA
from GeodesicRelaxationDCCA.algorithms.losses_metrics import (
    EpochWatchdog,
    MetricDict,
    MovingMetric,
    get_mv_similarity_metric,
    get_similarity_metric_v1,
)
from GeodesicRelaxationDCCA.experiments.template import (
    DeepCCAExperiment,
    DeepCCAResidualSlackExperiment,
    Experiment,
)


class SyntheticExperiment(Experiment):
    def get_moving_metrics(self):
        if isinstance(self.watchdog, EpochWatchdog):
            cor_movmetr = {
                "cor_" + str(num): MovingMetric(window_length=50, history_length=100, fun=tf.math.reduce_mean)
                for num in range(self.architecture.min_out_dim)
            }
        else:
            cor_movmetr = {
                "cor_" + str(num): MovingMetric(window_length=50, history_length=100, fun=tf.math.reduce_mean)
                for num in range(self.shared_dim)
            }

        simi_movmetr = {
            "sim_v0": MovingMetric(window_length=50, history_length=100, fun=tf.math.reduce_mean),
            "sim_v1": MovingMetric(window_length=50, history_length=100, fun=tf.math.reduce_mean),
            "sim_avg": MovingMetric(window_length=50, history_length=100, fun=tf.math.reduce_mean),
        }

        return {**cor_movmetr, **simi_movmetr}

    def compute_similarity_scores(self):
        training_data = self.dataprovider.training_data

        outputs = MetricDict()
        for data in training_data:
            network_output = self.architecture(data, training=False)
            outputs.update(network_output)

        network_output = outputs.output()

        sim_v0 = get_mv_similarity_metric(
            S=tf.transpose(self.dataprovider.z_0)[: self.dataprovider.true_dim],
            U=tf.transpose(network_output["cca_view_0"])[: self.dataprovider.true_dim],
            dims=self.dataprovider.true_dim,
        )
        sim_v1 = get_mv_similarity_metric(
            S=tf.transpose(self.dataprovider.z_1)[: self.dataprovider.true_dim],
            U=tf.transpose(network_output["cca_view_1"])[: self.dataprovider.true_dim],
            dims=self.dataprovider.true_dim,
        )

        return sim_v0, sim_v1

    def log_metrics(self):
        self.watchdog.decrease_counter()

        if self.epoch % self.eval_epochs == 0:
            # Compute correlation values on training data
            training_outp = self.predict(self.dataprovider.training_data)
            ccor = training_outp["ccor"]

            sim_v0, sim_v1 = self.compute_similarity_scores()
            sim_avg = (sim_v0 + sim_v1) / 2

            l1 = self.architecture.get_l1()
            l2 = self.architecture.get_l2()

            self.moving_metrics["sim_v0"].update_window(sim_v0)
            smoothed_sim_v0 = self.moving_metrics["sim_v0"].get_metric()
            self.moving_metrics["sim_v1"].update_window(sim_v1)
            smoothed_sim_v1 = self.moving_metrics["sim_v1"].get_metric()
            self.moving_metrics["sim_avg"].update_window(sim_avg)
            smoothed_sim_avg = self.moving_metrics["sim_avg"].get_metric()

            if smoothed_sim_v0 < self.best_val_view0:
                self.best_val_view0 = smoothed_sim_v0
                self.save_weights(subdir="view0")

            if smoothed_sim_v1 < self.best_val_view1:
                self.best_val_view1 = smoothed_sim_v1
                self.save_weights(subdir="view1")

            if smoothed_sim_avg < self.best_val_avg:
                self.best_val_avg = smoothed_sim_avg
                self.save_weights(subdir="avg")

            for num in range(self.shared_dim):
                self.moving_metrics["cor_" + str(num)].update_window(ccor[num])

            self.summary_writer.write_scalar_summary(
                epoch=self.epoch,
                list_of_tuples=[(ccor[num], "Correlations/" + str(num)) for num in range(self.shared_dim)]
                + [
                    (self.moving_metrics["cor_" + str(num)].get_metric(), "MovingMean/Correlation_" + str(num))
                    for num in range(self.shared_dim)
                ]
                + [
                    (self.moving_metrics["sim_v0"].get_metric(), "Metrics/Smooth distance measure 1st view"),
                    (self.moving_metrics["sim_v1"].get_metric(), "Metrics/Smooth distance measure 2nd view"),
                    (self.moving_metrics["sim_avg"].get_metric(), "Metrics/Smooth distance measure average"),
                    (self.watchdog.compute(), "MovingMean/Watchdog"),
                    (sim_v0, "Metrics/Distance measure 1st view"),
                    (sim_v1, "Metrics/Distance measure 2nd view"),
                    (sim_avg, "Metrics/Distance measure avg"),
                    (l1, "Regularization/L1"),
                    (l2, "Regularization/L2"),
                ],
            )

            if self.watchdog.check():
                if self.watchdog.reset():
                    pass
                else:
                    self.continue_training = False


class SynthDeepCCAExperiment(DeepCCAExperiment, SyntheticExperiment):
    pass

class SynthDeepCCASlackExperiment(DeepCCAResidualSlackExperiment, SyntheticExperiment):
    def compute_similarity_scores(self):
        training_data = self.dataprovider.training_data

        outputs = MetricDict()
        for data in training_data:
            network_output = self.architecture(data, training=False)
            outputs.update(network_output)

        network_output = outputs.output()

        sim_v0 = get_mv_similarity_metric(
            S=tf.transpose(self.dataprovider.z_0)[: self.dataprovider.true_dim],
            U=network_output["rrcca_view_0"],
            dims=self.dataprovider.true_dim,
        )
        sim_v1 = get_mv_similarity_metric(
            S=tf.transpose(self.dataprovider.z_1)[: self.dataprovider.true_dim],
            U=network_output["rrcca_view_1"],
            dims=self.dataprovider.true_dim,
        )

        return sim_v0, sim_v1

    def log_metrics(self):
        self.watchdog.decrease_counter()

        if self.epoch % self.eval_epochs == 0:
            # Compute correlation values on training data
            training_outp = self.predict(self.dataprovider.training_data)
            ccor = training_outp["ccor"]

            sim_v0, sim_v1 = self.compute_similarity_scores()
            sim_avg = (sim_v0 + sim_v1) / 2

            l1 = self.architecture.get_l1()
            l2 = self.architecture.get_l2()

            self.moving_metrics["sim_v0"].update_window(sim_v0)
            smoothed_sim_v0 = self.moving_metrics["sim_v0"].get_metric()
            self.moving_metrics["sim_v1"].update_window(sim_v1)
            smoothed_sim_v1 = self.moving_metrics["sim_v1"].get_metric()
            self.moving_metrics["sim_avg"].update_window(sim_avg)
            smoothed_sim_avg = self.moving_metrics["sim_avg"].get_metric()

            if smoothed_sim_v0 < self.best_val_view0:
                self.best_val_view0 = smoothed_sim_v0
                self.save_weights(subdir="view0")

            if smoothed_sim_v1 < self.best_val_view1:
                self.best_val_view1 = smoothed_sim_v1
                self.save_weights(subdir="view1")

            if smoothed_sim_avg < self.best_val_avg:
                self.best_val_avg = smoothed_sim_avg
                self.save_weights(subdir="avg")

            for num in range(self.shared_dim):
                self.moving_metrics["cor_" + str(num)].update_window(ccor[num])

            self.summary_writer.write_scalar_summary(
                epoch=self.epoch,
                list_of_tuples=[(ccor[num], "Correlations/" + str(num)) for num in range(self.shared_dim)]
                + [
                    (self.moving_metrics["cor_" + str(num)].get_metric(), "MovingMean/Correlation_" + str(num))
                    for num in range(self.shared_dim)
                ]
                + [
                    (self.moving_metrics["sim_v0"].get_metric(), "Metrics/Smooth distance measure 1st view"),
                    (self.moving_metrics["sim_v1"].get_metric(), "Metrics/Smooth distance measure 2nd view"),
                    (self.moving_metrics["sim_avg"].get_metric(), "Metrics/Smooth distance measure average"),
                    (self.watchdog.compute(), "MovingMean/Watchdog"),
                    (sim_v0, "Metrics/Distance measure 1st view"),
                    (sim_v1, "Metrics/Distance measure 2nd view"),
                    (sim_avg, "Metrics/Distance measure avg"),
                    (l1, "Regularization/L1"),
                    (l2, "Regularization/L2"),
                ],
            )

            if self.watchdog.check():
                if self.watchdog.reset():
                    pass
                else:
                    self.continue_training = False


