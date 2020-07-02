import os
import pkg_resources
import shutil
import itertools

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

from surfbreak.loss_functions import wave_pml_2

        
class LitSirenNet(pl.LightningModule):    # With no gradient or wave loss, oemega values of (1, 5) work well (best at epochs 3~5)  
    def __init__(self, hidden_features=256, hidden_layers=3, first_omega_0=1.0, hidden_omega_0=5.0, squared_slowness=3.0,
                 steps_per_vid_chunk=150, learning_rate=1e-4, grad_loss_scale=1e-4, wavefunc_loss_scale=1e-7, 
                 wavespeed_loss_scale=1e-2):
        """steps_per_vid_chunk defines the single-tensor resampled dataset length"""
        super().__init__()
        self.save_hyperparameters('first_omega_0', 'hidden_omega_0', 'squared_slowness', 'wavefunc_loss_scale', 'grad_loss_scale','wavespeed_loss_scale',
                                  'hidden_features', 'hidden_layers', 'steps_per_vid_chunk', 'learning_rate' )
        self.model = siren.Siren(in_features=3, 
                                 out_features=1, 
                                 hidden_features=hidden_features,
                                 hidden_layers=hidden_layers, outermost_linear=True,
                                 first_omega_0=first_omega_0,
                                 hidden_omega_0=hidden_omega_0)
        if wavespeed_loss_scale not in [None, 0, 0.]:
            self.slowness_model = siren.Siren(in_features=2,     
                                            out_features=1, 
                                            hidden_features=64,
                                            hidden_layers=2, outermost_linear=True,
                                            first_omega_0=3.5, #1.5
                                            hidden_omega_0=15.) #10.

        self.steps_per_vid_chunk=steps_per_vid_chunk
        self.learning_rate=learning_rate
        self.grad_loss_scale=grad_loss_scale
        self.squared_slowness = squared_slowness
        self.wavefunc_loss_scale=wavefunc_loss_scale
        self.wavespeed_loss_scale=wavespeed_loss_scale
        
        self.example_input_array = torch.ones(1,1337,3)

    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, batch_nb):
        model_input, ground_truth = batch
        wf_values_out, coords_out = self.model(model_input['masked_coords'])
        
        mse_loss = F.mse_loss(wf_values_out, ground_truth['masked_wf_values'])
        avg_loss = wf_values_out.mean()**2 * 0.01 # Gentle pressure to have mean-zero across entire image.
        tensorboard_logs = {'train/mse_loss':mse_loss,
                            'train/avg_loss':avg_loss}

        if self.grad_loss_scale in [None, 0, 0.]:
            grad_loss = 0; laplace_loss = 0
        else:
            grad_loss    =  siren.gradient(wf_values_out, coords_out).abs().mean()*self.grad_loss_scale
            laplace_loss =  siren.laplace(wf_values_out, coords_out).abs().mean()*self.grad_loss_scale*0.1
            tensorboard_logs['train/grad_loss'] = grad_loss
            tensorboard_logs['train/laplace_loss'] = laplace_loss

        if self.squared_slowness in [None, 0, 0.] or self.wavefunc_loss_scale in [None, 0, 0.]:
            wavefunc_loss = 0
        else:
            if self.wavespeed_loss_scale in [None, 0, 0.]:
                squared_slowness_tensor = torch.ones_like(coords_out) * self.squared_slowness
                wavespeed_loss = 0
            else:
                slow_vals_out, slow_coords_out = self.slowness_model(model_input['masked_coords'][...,1:]) # Omit the first channel (time)
                squared_slowness_tensor = slow_vals_out.repeat(1,1,3).clamp(min=1e-5) # do not allow negative or zero squared slowness values to ruin the physics
                # Gently push towards known a good value for squared_slowness,                     and heavily penalize all nonsensical negative values
                wavespeed_loss =  (slow_vals_out - self.squared_slowness).abs().mean()*self.wavespeed_loss_scale - slow_vals_out.clamp(max=0).sum()
                tensorboard_logs['train/wavespeed_loss'] = wavespeed_loss
                assert squared_slowness_tensor.shape == coords_out.shape
            wave_loss_dict = wave_pml_2(wf_values_out, coords_out, squared_slowness_tensor)
            wavefunc_loss = wave_loss_dict['diff_constraint_hom']*self.wavefunc_loss_scale # * min(1, (step/ total_steps)**2)
            tensorboard_logs['train/wavefunc_loss'] = wavefunc_loss

        train_loss = mse_loss + avg_loss + grad_loss + laplace_loss + grad_loss + laplace_loss + wavefunc_loss + wavespeed_loss

        tensorboard_logs['train/loss'] = train_loss

        return {'loss': train_loss, 'log': tensorboard_logs}

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

        # Calcuate and plot the inferred squared_slowness field in x and y
        cpu_slow_model =self.slowness_model.cpu()
        slow_vals_out, slow_coords_out = cpu_slow_model(model_input['all_coords'][...,1:]) # Omit the first channel (time)
        slow_vals_txy = slow_vals_out.reshape(ground_truth['full_tensor_shape'])
        fig2 = plt.figure()
        plt.imshow(slow_vals_txy[0].detach().numpy().T)
        plt.colorbar()
        plt.title('Squared slowness estimate')
        self.logger.experiment.add_figure('squared_slowness', fig2, self.current_epoch)
        self.slowness_model.cuda()

        avg_loss = sum(x["val_loss"] for x in outputs) / len(outputs)

        # Pass the accuracy to the `DictLogger` via the `'log'` key.
        tensorboard_logs = {'val/avg_loss': avg_loss}
        return {"val_loss": avg_loss, "log":tensorboard_logs}

    def configure_optimizers(self):
        if self.wavespeed_loss_scale not in [None, 0, 0.]:
            return Adam(itertools.chain(self.model.parameters(), self.slowness_model.parameters()), lr=self.learning_rate)
        else:
            return Adam(self.model.parameters(), lr=self.learning_rate)
    
    def setup(self, stage):
        # Train on a dataset consisting of 30-second chunks offset by 30 seconds
        self.wf_train_video_dataset = WaveformVideoDataset(ydim=120, xrange=(10,130), timerange=(0,61), time_chunk_duration_s=30, 
                                                     time_chunk_stride_s=30, time_axis_scale=0.5)
        self.wf_train_chunk_dataset = WaveformChunkDataset(self.wf_train_video_dataset, xy_bucket_sidelen=20, samples_per_xy_bucket=10, 
                                                     time_sample_interval=5, steps_per_video_chunk=self.steps_per_vid_chunk)
        # Validate on a dataset centered on the gap between the two training video chunks. Evaluate the MSE in this center area.
        # Having the same center timepoint will ensure the centered time representations are aligned between training and validation
        self.wf_valid_video_dataset = WaveformVideoDataset(ydim=120, xrange=(10,130), timerange=(0,61), time_chunk_duration_s=20, 
                                                     time_chunk_stride_s=20, time_axis_scale=0.5)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.wf_train_chunk_dataset, batch_size=1, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.wf_valid_video_dataset, batch_size=1, shuffle=False, num_workers=2)
