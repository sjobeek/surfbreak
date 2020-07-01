import os
import pkg_resources
import shutil

import matplotlib.pyplot as plt

import pytorch_lightning as pl
from pytorch_lightning import Callback
import torch
import torch.nn.functional as F
from torch.optim import Adam
import torch.utils.data
from surfbreak import train_utils

import optuna
from optuna.integration import PyTorchLightningPruningCallback

from surfbreak.loss_functions import wave_pml 
from surfbreak.datasets import WaveformVideoDataset, WaveformChunkDataset
import explore_siren as siren


        
class LitSirenNet(pl.LightningModule):                       # Omega values extrapolate well (from some hyperparameter tuning studies)  
    def __init__(self, hidden_features=256, hidden_layers=3, first_omega_0=3.8, hidden_omega_0=2.8, squared_slowness=3.0,
                 steps_per_vid_chunk=150, learning_rate=1e-4):
        """steps_per_vid_chunk defines the single-tensor resampled dataset length"""
        super().__init__()
        self.save_hyperparameters()
        self.model = siren.Siren(in_features=3, 
                                 out_features=1, 
                                 hidden_features=hidden_features,
                                 hidden_layers=hidden_layers, outermost_linear=True,
                                 first_omega_0=first_omega_0,
                                 hidden_omega_0=hidden_omega_0,
                                 squared_slowness=squared_slowness) 

        self.steps_per_vid_chunk=steps_per_vid_chunk
        self.learning_rate=learning_rate
        
        self.example_input_array = torch.ones(1,1337,3)

    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, batch_nb):
        model_input, ground_truth = batch
        wf_values_out, coords_out = self.model(model_input['masked_coords'])
        loss = F.mse_loss(wf_values_out, ground_truth['masked_wf_values'])
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        # Ignore batch size, and always measure on the middle video (which spans the 'gap')
        model_input, ground_truth = self.wf_valid_video_dataset[1]
        cpu_model = self.model.cpu()
        wf_values_out, coords_out = cpu_model(model_input['all_coords'])
        loss = F.mse_loss(wf_values_out, ground_truth['all_wavefront_values'])
        return {'val_loss':loss}

    def validation_epoch_end(self, outputs):

        model_input, ground_truth = self.wf_valid_video_dataset[1]
        cpu_model = self.model.cpu()
        wf_values_out, coords_out = cpu_model(model_input['all_coords'])

        wf_gt_txy = ground_truth['all_wavefront_values'].reshape(ground_truth['full_tensor_shape'])
        wf_out_txy = wf_values_out.reshape(ground_truth['full_tensor_shape'])
        fig = train_utils.waveform_tensors_plot(wf_out_txy, wf_gt_txy, coords=coords_out)
        self.logger.experiment.add_figure('val_xyslice', fig, self.current_epoch)

        # Also plot one of the training dataset images
        model_input, ground_truth =  self.wf_train_video_dataset[0]
        wf_values_out, coords_out = cpu_model(model_input['all_coords'])
        wf_gt_txy = ground_truth['all_wavefront_values'].reshape(ground_truth['full_tensor_shape']).cpu()
        wf_out_txy = wf_values_out.reshape(ground_truth['full_tensor_shape']).cpu()
        fig = train_utils.waveform_tensors_plot(wf_out_txy, wf_gt_txy, coords=coords_out)
        self.logger.experiment.add_figure('train_xyslice', fig, self.current_epoch)

        self.model.cuda()

        avg_loss = sum(x["val_loss"] for x in outputs) / len(outputs)

        # Pass the accuracy to the `DictLogger` via the `'log'` key.
        tensorboard_logs = {'avg_val_loss': avg_loss}
        return {"val_loss": avg_loss, "log":tensorboard_logs}

    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=self.learning_rate)
    
    def setup(self, stage):
        # Train on a dataset consisting of 30-second chunks offset by 30 seconds
        self.wf_train_video_dataset = WaveformVideoDataset(ydim=120, xrange=(10,130), timerange=(0,61), time_chunk_duration_s=30, 
                                                     time_chunk_stride_s=30, time_axis_scale=0.5)
        self.wf_train_chunk_dataset = WaveformChunkDataset(self.wf_train_video_dataset, xy_bucket_sidelen=20, samples_per_xy_bucket=100, 
                                                     time_sample_interval=5, steps_per_video_chunk=self.steps_per_vid_chunk)
        # Validate on a dataset centered on the gap between the two training video chunks. Evaluate the MSE in this center area.
        # Having the same center timepoint will ensure the centered time representations are aligned between training and validation
        self.wf_valid_video_dataset = WaveformVideoDataset(ydim=120, xrange=(10,130), timerange=(0,61), time_chunk_duration_s=20, 
                                                     time_chunk_stride_s=20, time_axis_scale=0.5)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.wf_train_chunk_dataset, batch_size=1, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.wf_valid_video_dataset, batch_size=1, shuffle=False)
