import os
from collections import OrderedDict
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.svm import LinearSVC as SVM
from sklearn.metrics import accuracy_score
from GeodesicRelaxationDCCA.algorithms.losses_metrics import MetricDict, get_similarity_metric_v1
from GeodesicRelaxationDCCA.algorithms.correlation import CCA

class Evaluation():
    def __init__(self, dataprovider, experiment_dir, regex=None):
        self.dataprovider = dataprovider
        self.exp_dir = experiment_dir
        self.competitors = self.get_competitors(regex=regex)

        self.results = None

    def get_competitors(self, regex):
        # Get overview over all types of networks
        competitors = {file : [] for file in os.listdir(self.exp_dir) if os.path.isdir(os.path.join(self.exp_dir, file))}
        
        # Sort by name
        competitors = OrderedDict(sorted(competitors.items()))

        # Filter different trainings by using a regular expression
        if regex is not None:
            keys_to_pop = list()
            for competitor in competitors:
                if regex not in competitor:
                    keys_to_pop.append(competitor)
                    
            for key in keys_to_pop:
                _ = competitors.pop(key)

        # Get all network paths
        for competitor in competitors:
            path = os.path.join(self.exp_dir, competitor)
            files = os.listdir(path)
            for file in files:
                file_path = os.path.join(path,file)
                if os.path.isdir(file_path):
                    competitors[competitor].append(file_path)

        return competitors

    def validate_all(self, weights_to_load):
        results = dict()
        for comp in self.competitors:
            results[comp] = list()
            for model_path in self.competitors[comp]:
                try:
                    model = tf.keras.models.load_model(model_path)
                    model.saved_model_path = model_path
                    results[comp].append(self.eval(model, weights_to_load=weights_to_load))
                except OSError as e:
                    print(e)
            if len(results[comp]) == 0:
                del results[comp]

        self.results = results

        return results

    def generate_result_table(self):
        if self.results is None:
            assert False == True
        
        table = dict()
        for competitor in self.results.keys():
            runs = self.results[competitor]
            example = runs[0]
            metrics = list(runs[0].keys())
            table[competitor] = dict()
            
            for metric in metrics:
                if len(example[metric].shape) == 0:
                    table[competitor][metric] = np.mean([run[metric] for run in runs])
                elif len(example[metric].shape) == 1:
                    stack = np.mean(np.stack([run[metric] for run in runs], axis=1), axis=1)
                    for i, layer in enumerate(stack):
                        table[competitor][metric+'_'+str(i)] = layer
                        
        return pd.DataFrame(data=table)

    def generate_plots(self):
        if self.results is None:
            assert False == True

        table = dict()
        for competitor in self.results.keys():
            competitor_display_name = ' '.join(competitor.split('_')[:3])
            runs = self.results[competitor]
            example = runs[0]
            metrics = list(runs[0].keys())
            table[competitor_display_name] = dict()
            
            for metric in metrics:
                if len(example[metric].shape) == 0:
                    table[competitor_display_name][metric] = [run[metric] for run in runs]
                elif len(example[metric].shape) == 1:
                    stack = np.stack([run[metric] for run in runs], axis=1)
                    for i, layer in enumerate(stack):
                        table[competitor_display_name][metric+'_'+str(i)] = layer

        
        metrics_names = list(table[str(list(table.keys())[0])].keys())
        competitors = list(table.keys())

        fig, axs = plt.subplots(len(metrics_names),1, figsize=(len(competitors)*4, len(metrics_names)*7))
        fig.tight_layout()
        if len(metrics_names) == 1:
            metric = metrics_names[0]
            axs.set_title(metric)
            axs.set_ylim(bottom=0.0, top=1.0)
            axs.yaxis.set_ticks(np.arange(0, 1, 0.1))
            axs.grid(visible=True, axis='y')
            axs.boxplot([table[competitor][metric] for competitor in competitors], labels=competitors)
            _ = [axs.scatter( x=(j+1)*np.ones(len(table[competitor][metric])), y=table[competitor][metric], c='blue') \
                        for j, competitor in enumerate(competitors)]
        else:
            plt.subplots_adjust(wspace=0.2, hspace=0.2)
            for i, metric in enumerate(metrics_names):
                axs[i].set_title(metric)
                axs[i].set_ylim(bottom=0.0, top=1.0)
                axs[i].yaxis.set_ticks(np.arange(0, 1, 0.1))
                axs[i].grid(visible=True, axis='y')
                axs[i].boxplot([table[competitor][metric] for competitor in competitors], labels=competitors)
                _ = [axs[i].scatter( x=(j+1)*np.ones(len(table[competitor][metric])), y=table[competitor][metric], c='blue') \
                            for j, competitor in enumerate(competitors)]
                        

class SyntheticEvaluation(Evaluation):
    def eval(self, model, weights_to_load):
        if weights_to_load in ['view0', 'view1', 'avg', 'latest']:
            load_status = model.load_weights(os.path.join(model.saved_model_path, weights_to_load))
            assert load_status
        else:
            raise ValueError('weights_to_load must be either view0, view1, avg or latest')
        # Load training data once
        training_data = self.dataprovider.training_data
        outputs = MetricDict()

        for data in training_data:
            # Feed forward
            network_output = model(data, training=False)

            outputs.update(network_output)

        return self.compute_metrics(outputs.output())

    def compute_metrics(self, network_output):
        metrics = dict()
        # Compute all metrics
        sim_v0 = get_similarity_metric_v1(
            S=tf.transpose(self.dataprovider.z_0)[:self.dataprovider.true_dim], # should be true number of correlated components
            # to evaluate in cases where technique takes incorrect number of correlated components as input
            U=tf.transpose(network_output['cca_view_0']),
            dims=self.dataprovider.true_dim
        )
        sim_v1 = get_similarity_metric_v1(
            S=tf.transpose(self.dataprovider.z_1)[:self.dataprovider.true_dim],
            U=tf.transpose(network_output['cca_view_1']),
            dims=self.dataprovider.true_dim
        )

        sim_avg = (sim_v0 + sim_v1)/2

        metrics['ccor'], metrics['sim_v0'], metrics['sim_v1'], metrics['sim_avg'] = network_output['ccor'], sim_v0, sim_v1, sim_avg

        return metrics

class MNISTEvaluation(Evaluation):
    def eval(self, model, shared_dim):
        # Load training data once
        test_data = self.dataprovider.test_data
        outputs = MetricDict()

        for data in test_data:
            # Feed forward
            network_output = model(data)

            outputs.update(network_output)

        return self.compute_metrics(outputs.output(), data['labels'].numpy())

    def compute_metrics(self, network_output, labels):
        metrics = dict()
        # Compute all metrics
        clust_labels = SpectralClustering(
            n_clusters=self.dataprovider.num_classes,
            assign_labels='kmeans',
            affinity='nearest_neighbors',
            random_state=33,
            n_init=10).fit_predict(network_output['latent_view_0'])

        prediction = np.zeros_like(clust_labels)
        for i in range(self.dataprovider.num_classes):
            ids = np.where(clust_labels == i)[0]
            prediction[ids] = np.argmax(np.bincount(labels[ids]))

        m = tf.keras.metrics.Accuracy()
        m.update_state(prediction, labels)
        metrics['accuracy'] = m.result().numpy()

        return metrics

class EEGEvaluation(Evaluation):
    def validate_all(self):
        results = dict()
        for comp in self.competitors:
            results[comp] = list()
            for model_path in self.competitors[comp]:
                try:
                    model = tf.keras.models.load_model(model_path)
                    results[comp].append(self.eval(self.dataprovider.get_split(int(model_path[-4])), model))
                except OSError as e:
                    print(e)
            if len(results[comp]) == 0:
                del results[comp]

        self.results = results

    def eval(self, dataprovider, model):
        return self.compute_metrics(model, dataprovider)

    def compute_metrics(self, model, dataprovider):
        data_for_acc = dataprovider.test_data

        for data in dataprovider.training_data:
            netw_output_train = model(data)
            labels_train = data['labels'].numpy()

        svm_model = SVM(random_state=333)
        svm_model.fit(netw_output_train['latent_view_1'].numpy(), labels_train)

        for data in data_for_acc:
            netw_output_val = model(data)
            labels_val = data['labels'].numpy()

        predictions = svm_model.predict(netw_output_val['latent_view_1'].numpy())
        svm_acc = accuracy_score(labels_val, predictions)

        return dict(accuracy=svm_acc)

    def generate_result_table(self):
        if self.results is None:
            assert False == True
        
        unique_exp = list(set([res[:-3] for res in list(self.results.keys())]))

        table = {exp: dict() for exp in unique_exp}
        for competitor in self.results.keys():
            runs = self.results[competitor]
            example = runs[0]
            metrics = list(runs[0].keys())
            table[competitor[:-3]][competitor] = dict()
            
            for metric in metrics:
                if len(example[metric].shape) == 0:
                    table[competitor[:-3]][competitor][metric] = np.mean([run[metric] for run in runs])
                elif len(example[metric].shape) == 1:
                    stack = np.mean(np.stack([run[metric] for run in runs], axis=1), axis=1)
                    for i, layer in enumerate(stack):
                        table[competitor[:-3]][competitor][metric+'_'+str(i)] = layer

        result_table = dict()
        for key in table.keys():
            result_table[key] = dict()
            split_experiments = table[key]
            first_split = list(split_experiments.keys())[0]
            example = split_experiments[first_split]
            for metric in example.keys():
                result_table[key][metric] = np.mean([split_experiments[split][metric] for split in split_experiments.keys()])
            
        return pd.DataFrame(data=result_table)
