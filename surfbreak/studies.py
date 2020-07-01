import os
import pkg_resources
import shutil

import matplotlib.pyplot as plt

import pytorch_lightning as pl
from pytorch_lightning import Callback
import torch.nn.functional as F
from torch.optim import Adam
import torch.utils.data

import optuna
from optuna.integration import PyTorchLightningPruningCallback

from surfbreak.loss_functions import wave_pml 
from surfbreak.datasets import WaveformVideoDataset, WaveformChunkDataset
from surfbreak.waveform_models import LitSirenNet


class MetricsCallback(pl.Callback):
    """PyTorch Lightning metric callback."""
    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)

def run_waveform_hyperparam_search(n_trials=100, timeout=60*60, max_epochs=10, logdir='logs', model_folder='results', prune=False):
    DIR = os.path.join(os.getcwd(), logdir)
    MODEL_DIR = os.path.join(DIR, model_folder)

    def objective(trial):
        # Filenames for each trial must be made unique in order to access each checkpoint.
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            os.path.join(MODEL_DIR, "trial_{}".format(trial.number), "{epoch}"), monitor="loss"
        )
        
        tb_logger = pl.loggers.TensorBoardLogger('logs/', name="optuna_tests")
        
        # The default logger in PyTorch Lightning writes to event files to be consumed by
        # TensorBoard. We don't use any logger here as it requires us to implement several abstract
        # methods. Instead we setup a simple callback, that saves metrics from each validation step.
        metrics_callback = MetricsCallback()
        
        trainer = pl.Trainer(
            logger=tb_logger,
            limit_val_batches=1,
            checkpoint_callback=checkpoint_callback,
            max_epochs=max_epochs,
            gpus=1 if torch.cuda.is_available() else None,
            callbacks=[metrics_callback],
            early_stop_callback=None#PyTorchLightningPruningCallback(trial, monitor="val_loss"),
            )
                                        
        model = LitSirenNet(hidden_features= trial.suggest_categorical('hidden_features', [128, 256]),
                            hidden_layers=3,
                            first_omega_0=trial.suggest_loguniform('first_omega_0', 0.1, 30.),
                            hidden_omega_0=trial.suggest_loguniform('hidden_omega_0', 1., 30.), 
                            squared_slowness=3.0)
        trainer.fit(model)

        return metrics_callback.metrics[-1]["val_loss"].item()

    if prune:
        pruner = optuna.pruners.MedianPruner()
    else:
        pruner = optuna.pruners.NopPruner()

    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=n_trials, timeout=timeout) # run for 30 minutes or 100 trials

    print("Number of finished trials: {}".format(len(study.trials)))

    trial = study.best_trial
    print("Best trial was #{}:".format(trial.number))
    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    shutil.rmtree(MODEL_DIR)

    return study