from abc import ABC, abstractmethod
import os
from collections import OrderedDict
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

class Evaluation(ABC):
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

    def validate_all(self, weights_to_load, split='test'):
        results = dict()
        for comp in self.competitors:
            results[comp] = list()
            for model_path in self.competitors[comp]:
                try:
                    model = tf.keras.models.load_model(model_path)
                    if weights_to_load in ['view0', 'view1', 'avg', 'latest']:
                        if weights_to_load != 'latest':
                            load_status = model.load_weights(os.path.join(model_path, weights_to_load))
                            assert load_status
                    else:
                        raise ValueError('weights_to_load must be either view0, view1, avg or latest')
                    results[comp].append(self.eval(model, split))
                except OSError as e:
                    print(e)
            if len(results[comp]) == 0:
                del results[comp]

        self.results = results

        return results

    def generate_result_table(self, summary_operation=np.mean):
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
                    table[competitor][metric] = summary_operation([run[metric] for run in runs])
                elif len(example[metric].shape) == 1:
                    stack = summary_operation(np.stack([run[metric] for run in runs], axis=1), axis=1)
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
   
    @abstractmethod
    def eval(self, model, weights_to_load):
        raise NotImplementedError