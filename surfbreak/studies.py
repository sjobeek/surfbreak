import os
import shutil
import optuna
import pytorch_lightning as pl
from optuna.integration import PyTorchLightningPruningCallback

class MetricsCallback(pl.Callback):
    """PyTorch Lightning metric callback."""
    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)

def run_waveform_hyperparam_search(objective, n_trials=100, timeout=1*60*60, model_dir='logs/opt_models', prune=False, n_startup_trials=3, n_warmup_steps=5):
    if prune:
        pruner = optuna.pruners.MedianPruner(n_startup_trials=n_startup_trials, n_warmup_steps=n_warmup_steps)
    else:
        pruner = optuna.pruners.NopPruner()
    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=n_trials, timeout=timeout) 

    print("Number of finished trials: {}".format(len(study.trials)))
    trial = study.best_trial
    print("Best trial was #{}:".format(trial.number))
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    shutil.rmtree(model_dir)

    return study
