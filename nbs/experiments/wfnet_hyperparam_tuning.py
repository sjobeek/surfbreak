import os
import torch
import wandb
import pytorch_lightning as pl
from surfbreak.waveform_models import WaveformNet
from surfbreak.datasets import WavefrontDatasetTXYC, MaskedWavefrontBatchesNC, CachedDataset
from optuna.integration import PyTorchLightningPruningCallback
from surfbreak.studies import run_waveform_hyperparam_search, MetricsCallback

os.chdir('/home/erik/work/surfbreak/nbs')

# NOTE! Change this environment variable during each optimization experiemnt to group them properly on wandb
EXPERIMENT_NAME = "test4"
os.environ["WANDB_RUN_GROUP"] = EXPERIMENT_NAME

LOGDIR = 'wandb'
MODELDIR = os.path.join(LOGDIR, 'opt_models')

def objective(trial):
    checkpoint_callback = pl.callbacks.ModelCheckpoint( # Filenames for each trial must be made unique
        os.path.join(MODELDIR, "trial_{}".format(trial.number), "{epoch}"), monitor="val_loss")

    wandb_logger = pl.loggers.wandb.WandbLogger(name=f"wfnet_opt_{trial.number}", save_dir=LOGDIR, project='surfbreak', log_model=True)

    #tb_logger = pl.loggers.TensorBoardLogger(LOGDIR+'/', name="opt_v2")
    metrics_callback = MetricsCallback()     # Simple callback that saves metrics from each validation step.
    
    pl.seed_everything(42)

    training_video = '../data/shirahama_1590387334_SURF-93cm.ts'
    cnn_checkpoint ='../models/simplecnn_shirahama.ckpt'
    start_s=60 
    duration_s=30
    max_epochs=10
    wf_net_kwargs = dict(        
        hidden_features=trial.suggest_categorical('hidden_features', [256]),
        hidden_layers=trial.suggest_categorical('hidden_layers', [3]),
        first_omega_0=trial.suggest_uniform('first_omega_0', 2.5, 4.5), #2.5, 
        hidden_omega_0=trial.suggest_uniform('hidden_omega_0', 10, 15), #11,
        squared_slowness=trial.suggest_uniform('squared_slowness', 1.0, 2.0), #1.0,
        learning_rate=2e-4,
        wavefunc_loss_scale=trial.suggest_loguniform('wavefunc_loss_scale', 1e-11, 1e-9), #1e-9,
        wavespeed_norm_loss_scale=trial.suggest_loguniform('wavespeed_norm_loss_scale', 1e-8, 1e-5), #1e-12,
        wavespeed_delta_loss_scale=trial.suggest_loguniform('wavespeed_delta_loss_scale', 1e-8, 1e-5),
        wavespeed_first_omega_0=trial.suggest_uniform('ws_fo0', 2.0,3.0), #0.5
        wavespeed_hidden_omega_0=trial.suggest_uniform('ws_ho0',2.0,3.0), #2.0
        wfloss_growth_scale=trial.suggest_loguniform('wfloss_growth_scale', 1.1, 2.0),
        pretrain_epochs=trial.suggest_int('pretrain_epochs',4,4)
    )

    # Train consists of 4-second chunks with a 1-second gap between each
    txy_train = CachedDataset(WavefrontDatasetTXYC, training_video, timerange=(start_s,start_s+duration_s), 
                                                    time_chunk_duration_s=3, time_chunk_stride_s=4, 
                                                    wavecnn_ckpt=cnn_checkpoint)
    wf_train_dataset = MaskedWavefrontBatchesNC(txy_train, samples_per_batch=600, included_time_fraction=1.0)
    
    # Validation covers last few seconds if the waveform, plus next few seconds (ability to extrapolate is desireable)
    txy_valid = CachedDataset(WavefrontDatasetTXYC, training_video, timerange=(start_s+2,start_s+duration_s), 
                                                    time_chunk_duration_s=2, time_chunk_stride_s=4, 
                                                    wavecnn_ckpt=cnn_checkpoint)
    wf_valid_dataset = MaskedWavefrontBatchesNC(txy_valid, samples_per_batch=600, included_time_fraction=0.25)
    
    # Visualize the last 25s of the waveform, plus the 5 seconds of validation-only data 
    viz_inftxy_dataset = WavefrontDatasetTXYC(training_video, timerange=(start_s+duration_s-25,start_s+duration_s+5), 
                                             time_chunk_duration_s=30, time_chunk_stride_s=30,
                                             wavecnn_ckpt=cnn_checkpoint)
   
    wavefunc_model = WaveformNet(train_dataset=wf_train_dataset, valid_dataset=wf_valid_dataset,
                                 viz_dataset=viz_inftxy_dataset, batch_size=100,
                                 **wf_net_kwargs)    
    # Magic   (log gradients, metrics, and the graph)
    wandb_logger.watch(wavefunc_model, log='gradients', log_freq=100)

    trainer = pl.Trainer(logger=wandb_logger, #limit_val_batches=50,
                         max_epochs=max_epochs, 
                         gpus=1 if torch.cuda.is_available() else None,
                         callbacks=[metrics_callback],
                         early_stop_callback=PyTorchLightningPruningCallback(trial, monitor="val_loss"),
                        )

    trainer.fit(wavefunc_model)
    return metrics_callback.metrics[-1]["val_loss"].item()


study = run_waveform_hyperparam_search(objective, n_trials=100, timeout=12*60*60, model_dir=MODELDIR, 
                                       prune=True, n_startup_trials=3, n_warmup_steps=5)
