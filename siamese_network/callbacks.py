import copy
import numpy as np
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
import os
import warnings
import json


class SaveTrainingData(Callback):
    def __init__(self, model_path, params, save_best_only=False, monitor='val_loss'):
        super().__init__()
        self.history = dict()
        self.model_path = model_path
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.best_value = None
        self.parameters = copy.deepcopy(params)
        self.epochs = 0

    def on_train_begin(self, logs=None):
        if self.save_best_only is True:
            if 'loss' in self.monitor:
                self.best_value = np.Inf
            else:
                self.best_value = -np.Inf

    def on_epoch_end(self, epoch, logs={}):
        self.epochs += 1

        # Save training history
        for key in logs.keys():
            if key not in self.history.keys():
                self.history[key] = list()
            self.history[key].append(logs[key])

        if self.save_best_only is True:
            if self.monitor in logs.keys():
                if 'loss' in self.monitor:
                    if logs[self.monitor] < self.best_value:
                        self.best_value = logs[self.monitor]
                        self.save_to_disk()
                else:
                    if logs[self.monitor] > self.best_value:
                        self.best_value = logs[self.monitor]
                        self.save_to_disk()
            else:
                warnings.warn('Variable to monitor not found')
                self.save_to_disk()
        else:
            self.save_to_disk()

    def save_to_disk(self):
        # Save training plot
        for key in sorted(self.history.keys()):
            plt.plot(range(1, self.epochs+1), self.history[key], label=key)
        plt.grid()
        plt.legend()
        plt.xlabel('epochs')
        plt.title('Training evolution')
        plt.savefig(os.path.join(self.model_path, 'training.jpg'), dpi=150)
        plt.close()

        # Save json parameters, with updated epoch
        self.parameters['epochs'] = self.epochs
        with open(os.path.join(self.model_path, 'params.json'), 'w', encoding='utf-8') as f:
            json.dump(self.parameters, f, ensure_ascii=False, indent=2)
