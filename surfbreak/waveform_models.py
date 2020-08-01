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
from surfbreak.datasets import (WaveformVideoDataset, WaveformChunkDataset, InferredWaveformDataset, 
                                WavefrontSupervisionDataset, MaskedWavefrontDataset, MaskedCNNWavefrontDataset)
from surfbreak import base_models, diff_operators

from surfbreak.loss_functions import wave_pml_2

        
class LitSirenNet(pl.LightningModule):    # With no gradient or wave loss, oemega values of (1, 5) work well (best at epochs 3~5)  
    def __init__(self, video_filepath, hidden_features=256, hidden_layers=3, first_omega_0=1.0, hidden_omega_0=5.0, squared_slowness=3.0,
                 steps_per_vid_chunk=150, learning_rate=1e-4, grad_loss_scale=1e-4, wavefunc_loss_scale=1e-7, 
                 wavespeed_loss_scale=1e-2, xrange=(10,130), timerange=(0,31), chunk_duration=30, chunk_stride=15,
                 vid_dataset=None, chunk_dataset=None):
        """steps_per_vid_chunk defines the single-tensor resampled dataset length"""
        super().__init__()
        self.save_hyperparameters('first_omega_0', 'hidden_omega_0', 'squared_slowness', 'wavefunc_loss_scale', 'xrange', 'timerange', 'grad_loss_scale','wavespeed_loss_scale',
                                  'hidden_features', 'hidden_layers', 'steps_per_vid_chunk', 'learning_rate' )
        self.model = base_models.Siren(in_features=3, 
                                 out_features=1, 
                                 hidden_features=hidden_features,
                                 hidden_layers=hidden_layers, outermost_linear=True,
                                 first_omega_0=first_omega_0,
                                 hidden_omega_0=hidden_omega_0)
        if wavespeed_loss_scale not in [None, 0, 0.]:
            self.slowness_model = base_models.Siren(in_features=2,     
                                            out_features=1, 
                                            hidden_features=64,
                                            hidden_layers=2, outermost_linear=True,
                                            first_omega_0=3.5, #1.5
                                            hidden_omega_0=15., #10.
                                            softmax_output=True) # Prevent negative or zero squared slowness values from ruining the physics

        self.video_filepath = video_filepath
        self.steps_per_vid_chunk=steps_per_vid_chunk
        self.learning_rate=learning_rate
        self.grad_loss_scale=grad_loss_scale
        self.squared_slowness = squared_slowness
        self.wavefunc_loss_scale=wavefunc_loss_scale
        self.wavespeed_loss_scale=wavespeed_loss_scale
        self.timerange=timerange
        self.chunk_duration=chunk_duration
        self.chunk_stride=chunk_stride
        self.xrange = xrange
        self.wf_train_video_dataset = vid_dataset
        self.wf_train_chunk_dataset = chunk_dataset
        
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
            grad_loss    =  diff_operators.gradient(wf_values_out, coords_out).abs().mean()*self.grad_loss_scale
            laplace_loss =  diff_operators.laplace(wf_values_out, coords_out).abs().mean()*self.grad_loss_scale*0.1
            tensorboard_logs['train/grad_loss'] = grad_loss
            tensorboard_logs['train/laplace_loss'] = laplace_loss

        if self.squared_slowness in [None, 0, 0.] or self.wavefunc_loss_scale in [None, 0, 0.]:
            wavefunc_loss = 0
            wavespeed_loss = 0
        else:
            if self.wavespeed_loss_scale in [None, 0, 0.]:
                squared_slowness_tensor = torch.ones_like(coords_out) * self.squared_slowness
                wavespeed_loss = 0
            else:
                slow_vals_out, slow_coords_out = self.slowness_model(model_input['masked_coords'][...,1:]) # Omit the first channel (time)
                squared_slowness_tensor = slow_vals_out.repeat(1,1,3)
                # Gently push towards known a good value for squared_slowness
                wavespeed_loss =  (slow_vals_out - self.squared_slowness).abs().mean()*self.wavespeed_loss_scale
                tensorboard_logs['train/wavespeed_loss'] = wavespeed_loss
                assert squared_slowness_tensor.shape == coords_out.shape

            wave_loss_dict = wave_pml_2(wf_values_out, coords_out, squared_slowness_tensor)
            wavefunc_loss = wave_loss_dict['diff_constraint_hom']*self.wavefunc_loss_scale # * min(1, (step/ total_steps)**2)
            tensorboard_logs['train/wavefunc_loss'] = wavefunc_loss

        train_loss = mse_loss + avg_loss + grad_loss + laplace_loss + grad_loss + laplace_loss + wavefunc_loss + wavespeed_loss

        tensorboard_logs['train/loss'] = train_loss
        tensorboard_logs['train/sample_fraction'] = self.wf_train_chunk_dataset.sample_fraction 

        return {'loss': train_loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        model_input, ground_truth = batch
        bdim, tdim, xdim, ydim, channels = model_input['coords_txyc'].shape
        # Evaluate only on the center 1/2 of coordinate values (where valid wave data is likely)
        stp = 3 # Step skipped between t,x,ycoordinates to evaluate (just to reduce memory usage)
        eval_coords = model_input['coords_txyc'][:,::stp, xdim//4:-xdim//4:stp, ydim//4:-ydim//4:stp, :].reshape(1,-1,channels)
        wf_values_out, coords_out = self.model(eval_coords)
        loss = F.mse_loss(wf_values_out, ground_truth['wavefronts_txy'][:,::stp, xdim//4:-xdim//4:stp, ydim//4:-ydim//4:stp].reshape(1,-1,1))

        coords_txyc = model_input['coords_txyc'][0]        # Removing the batch dimension
        wavefronts_txy = ground_truth['wavefronts_txy'][0] # Removing the batch dimension
        first_video_image_xy = ground_truth['video_txy'][0,0] # First batch, first image
        fig0 = train_utils.plot_waveform_tensors(self.model, coords_txyc, wavefronts_txy, first_video_image_xy)
        self.logger.experiment.add_figure(f'valchunk{batch_nb}/waveforms', fig0, self.current_epoch)

        # Squared slowness estimate is independent of batch, so only calculate this once
        if batch_nb==0 and self.wavespeed_loss_scale not in [None, 0, 0.]:
            # Calcuate and plot the inferred squared_slowness field in x and y
            slow_vals_out, _ = self.slowness_model(coords_txyc[0,:,:,1:].reshape(-1,2)) # Skipping the first T channel
            slow_vals_array = slow_vals_out.reshape(coords_txyc[0,:,:,0].shape).cpu().detach().numpy()
            img_aspect = coords_txyc.shape[2]/coords_txyc.shape[3]
            fig2 = plt.figure(figsize=(10,5))
            img = plt.imshow(slow_vals_array.T)
            plt.title('Squared slowness estimate')
            if img_aspect > 1:
                plt.colorbar(img, fraction=0.046, pad=0.15, orientation='horizontal')
            else:
                plt.colorbar(img, fraction=0.046, pad=0.04, orientation='vertical')
            self.logger.experiment.add_figure('squared_slowness', fig2, self.current_epoch)
            self.slowness_model.cuda()



        return {'val_loss':loss}


    def validation_epoch_end(self, outputs):

        

        avg_loss = sum(x["val_loss"] for x in outputs) / len(outputs)

        # Pass the accuracy to the `DictLogger` via the `'log'` key.
        tensorboard_logs = {'val/avg_loss': avg_loss}

        return {"val_loss": avg_loss, "log":tensorboard_logs}

    def on_epoch_end(self):
        # Add half of a chunk's worth of samples to the current training batch each epoch
        self.wf_train_chunk_dataset.sample_fraction = min(1.0, self.wf_train_chunk_dataset.sample_fraction + 0.5/len(self.wf_train_video_dataset))

    def configure_optimizers(self):
        if self.wavespeed_loss_scale not in [None, 0, 0.]:
            return Adam(itertools.chain(self.model.parameters(), self.slowness_model.parameters()), lr=self.learning_rate)
        else:
            return Adam(self.model.parameters(), lr=self.learning_rate)
    
    def setup(self, stage):
        # Train on a dataset consisting of 30-second chunks offset by 30 seconds
        if self.wf_train_video_dataset is None:
            self.wf_train_video_dataset = WaveformVideoDataset(self.video_filepath, ydim=120, xrange=self.xrange, timerange=self.timerange, time_chunk_duration_s=self.chunk_duration, 
                                                               time_chunk_stride_s=self.chunk_stride, time_axis_scale=0.5)
        n_vid_chunks = len(self.wf_train_video_dataset)
        if self.wf_train_chunk_dataset is None:
            self.wf_train_chunk_dataset = WaveformChunkDataset(self.wf_train_video_dataset, xy_bucket_sidelen=20, samples_per_xy_bucket=15, 
                                                               time_sample_interval=15, steps_per_video_chunk=self.steps_per_vid_chunk,
                                                               sample_fraction=1.0/n_vid_chunks)
        # Validate on three chunks spread along the full time duration.
        # Having the same center timepoint will ensure the centered time representations are aligned between training and validation
        self.wf_valid_video_dataset = WaveformVideoDataset(self.video_filepath,ydim=120, xrange=self.xrange, timerange=self.timerange, time_chunk_duration_s=self.chunk_duration, 
                                                           time_chunk_stride_s=(self.timerange[1]-self.timerange[0])//3, time_axis_scale=0.5)


    def train_dataloader(self):
                                                        # Shuffling is handled by the chunk_dataset when sample_fraction < 1.0
        return torch.utils.data.DataLoader(self.wf_train_chunk_dataset, batch_size=1, shuffle=False, num_workers=4)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.wf_valid_video_dataset, batch_size=1, shuffle=False, num_workers=4)


class LitWaveCNN(pl.LightningModule): 
    def __init__(self, video_filepath, learning_rate=1e-4, wf_model_checkpoint=None, ydim=128,
                 train_dataset=None, val_dataset=None, batch_size=8,
                 timerange=(0,30), val_timerange=(30,40), chunk_duration=1, chunk_stride=1, n_input_channels=2):
        """"""
        super().__init__()
        self.save_hyperparameters('video_filepath', 'learning_rate', 'timerange', 'chunk_duration')
        self.model = base_models.WaveUnet(n_class=1, n_input_channels=n_input_channels)
       
        self.video_filepath = video_filepath
        self.learning_rate=learning_rate
        self.timerange=timerange
        self.val_timerange=val_timerange
        self.chunk_duration=chunk_duration
        self.chunk_stride=chunk_stride
        self.ydim = ydim
        self.n_input_channels=n_input_channels
        self.wf_model_checkpoint = wf_model_checkpoint
        self.train_dataset=train_dataset
        self.val_dataset=val_dataset
        self.batch_size=batch_size
        
        self.example_input_array = torch.ones(1,n_input_channels,128,256)

    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, batch_nb):
        model_input, ground_truth = batch
        wf_values_out = self.model(model_input)
        mse_loss = F.mse_loss(wf_values_out, ground_truth)
        # Gentle pressure to have mean-zero across entire image
        zeromean_loss = wf_values_out.mean()**2 * 0.01
        loss = mse_loss + zeromean_loss

        tensorboard_logs = {'train/loss':loss,
                            'train/mse_loss': mse_loss,
                            'train/avg_loss': zeromean_loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        model_input, ground_truth = batch
        model_output = self.model(model_input)
        loss = F.mse_loss(model_output, ground_truth)
        return {'val_loss':loss}

    def validation_epoch_end(self, outputs):
        avg_loss = sum(x["val_loss"] for x in outputs) / len(outputs)
        tensorboard_logs = {'val/avg_loss': avg_loss}
        return {"val_loss": avg_loss, "log":tensorboard_logs}

    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=self.learning_rate)
    
    def setup(self, stage):
        if self.train_dataset is None:
            self.train_wfs_dataset = WavefrontSupervisionDataset(self.video_filepath, ydim=self.ydim, timerange=self.timerange,
                                                                time_chunk_duration_s=self.chunk_duration, time_chunk_stride_s=self.chunk_stride)
            self.train_dataset = MaskedCNNWavefrontDataset(self.train_wfs_dataset)
        if self.val_dataset is None:
            self.val_wfs_dataset = WavefrontSupervisionDataset(self.video_filepath, ydim=self.ydim, timerange=self.val_timerange,
                                                                time_chunk_duration_s=self.chunk_duration, time_chunk_stride_s=self.chunk_stride)
            self.val_dataset = MaskedCNNWavefrontDataset(self.val_wfs_dataset)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)


class WaveformNet(pl.LightningModule):    # With no gradient or wave loss, oemega values of (1, 5) work well (best at epochs 3~5)  
    def __init__(self, hidden_features=256, hidden_layers=3, first_omega_0=2.5, hidden_omega_0=11, squared_slowness=0.2,
                 learning_rate=1e-4, wavefunc_loss_scale=5.5e-9, wavespeed_loss_scale=4e-4, 
                 wavespeed_first_omega_0=3.5, wavespeed_hidden_omega_0=15, 
                 video_filepath=None, train_dataset=None, valid_dataset=None, viz_dataset=None, batch_size=64):
        """Learns a low-dimensional function which maps t,x,y video coordinates to waveform magnitude. 
           The model applies a physics-based waveform cost function to regularize the signal, using a 2nd-order PDE (see the SIREN paper).
           The model also jointly inferrs the static wave propogation velocity field used the waveform loss (in units of squared slowness).             
           Setting wavespeed_loss_scale to None will result in assuming the uniform, static wave velocity given in squared_slowness
           """
        super().__init__()
        if video_filepath is None:
            assert train_dataset is not None and valid_dataset is not None, "Either a video_filepath or datasets must be provided."
        self.save_hyperparameters('first_omega_0', 'hidden_omega_0', 'squared_slowness', 'wavefunc_loss_scale', 'wavespeed_loss_scale',
                                  'hidden_features', 'hidden_layers', 'learning_rate' )
        self.model = base_models.Siren(in_features=3, 
                                 out_features=1, 
                                 hidden_features=hidden_features,
                                 hidden_layers=hidden_layers, outermost_linear=True,
                                 first_omega_0=first_omega_0,
                                 hidden_omega_0=hidden_omega_0)
        if wavespeed_loss_scale not in [None, 0, 0.]:
            self.slowness_model = base_models.Siren(in_features=2,     
                                            out_features=1, 
                                            hidden_features=64,
                                            hidden_layers=2, outermost_linear=True,
                                            first_omega_0=wavespeed_first_omega_0, #1.5
                                            hidden_omega_0=wavespeed_hidden_omega_0, #10.
                                            softmax_output=True) # Prevent negative or zero squared slowness values from ruining the physics

        self.learning_rate=learning_rate
        self.squared_slowness = squared_slowness
        self.wavefunc_loss_scale=wavefunc_loss_scale
        self.wavespeed_loss_scale=wavespeed_loss_scale
        self.wf_train_dataset = train_dataset
        self.wf_valid_dataset = valid_dataset
        self.viz_dataset = viz_dataset
        self.video_filepath = video_filepath
        self.batch_size = batch_size
        self.inferred_slowness=None

        self.example_input_array = torch.ones(1,1337,3)

    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, batch_nb):
        model_input, ground_truth = batch
        wf_values_out, coords_out = self.model(model_input['coords_sc'])
        
        ## Calculate the basic mean squared error loss using the training data
        mse_loss = F.mse_loss(wf_values_out, ground_truth['wavefront_values_sc'])
        # TODO: Devise method to normalize unsampled regions to have zero average amplitude
        avg_loss = (wf_values_out**2).mean() * 1e-6 # Gentle pressure to have zero amplitude across entire video.
        tensorboard_logs = {'train/mse_loss':mse_loss,
                            'train/avg_loss':avg_loss}

        ## Calculate the loss used to jointly learn the wave velocity field (if learning it - )
        if self.wavespeed_loss_scale in [None, 0, 0.]:
            squared_slowness_tensor = torch.ones_like(coords_out) * self.squared_slowness
            wavespeed_loss = 0
        else:
            slow_vals_out, slow_coords_out = self.slowness_model(model_input['coords_sc'][...,1:]) # Omit the first channel (time)
            squared_slowness_tensor = slow_vals_out.repeat(1,1,3)
            # Gently push towards known a good value for squared_slowness
            wavespeed_loss =  (slow_vals_out - self.squared_slowness).abs().mean()*self.wavespeed_loss_scale
            #TODO: Add in an (optional) loss term that encourages smooth, low-frequency wavespeed fields 
            #grad_loss    =  diff_operators.gradient(slow_vals_out, slow_coords_out).abs().mean()*self.grad_loss_scale
            #laplace_loss =  diff_operators.laplace(slow_vals_out, slow_coords_out).abs().mean()*self.grad_loss_scale*0.1
            #tensorboard_logs['train/ws_grad_loss'] = grad_loss
            #tensorboard_logs['train/ws_laplace_loss'] = laplace_loss
            tensorboard_logs['train/wavespeed_loss'] = wavespeed_loss
            assert squared_slowness_tensor.shape == coords_out.shape

        # Calculate the physics-based wave function loss
        wave_loss_dict = wave_pml_2(wf_values_out, coords_out, squared_slowness_tensor)
        wavefunc_loss = wave_loss_dict['diff_constraint_hom']*self.wavefunc_loss_scale # * min(1, (step/ total_steps)**2)
        tensorboard_logs['train/wavefunc_loss'] = wavefunc_loss

        train_loss = mse_loss + avg_loss + wavefunc_loss + wavespeed_loss

        tensorboard_logs['train/loss'] = train_loss
        tensorboard_logs['train/included_time_fraction'] = self.wf_train_dataset.included_time_fraction 

        return {'loss': train_loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        model_input, ground_truth = batch
        wf_values_out, _ = self.model(model_input['coords_sc'])
        if batch_nb < 5:
            self.logger.experiment.add_histogram('val/tcoords', model_input['coords_sc'][...,0], self.current_epoch)
            self.logger.experiment.add_histogram('val/xcoords', model_input['coords_sc'][...,1], self.current_epoch)
            self.logger.experiment.add_histogram('val/ycoords', model_input['coords_sc'][...,2], self.current_epoch)
            self.logger.experiment.add_histogram('val/wf_values_out', wf_values_out, self.current_epoch)

        loss = F.mse_loss(wf_values_out, ground_truth['wavefront_values_sc'])
        return {'val_loss':loss}

    def validation_epoch_end(self, outputs):
        if self.viz_dataset is not None:
            model_input, ground_truth = self.viz_dataset[0]

             # Calculate the latest x,y array of inferred slowness values across the image 
            coords_txyc = model_input['coords_txyc'].cuda()
            slow_vals_out, _ = self.slowness_model(coords_txyc[0,:,:,1:].reshape(-1,2)) # Skipping the first T channel
            slow_vals_array = slow_vals_out.reshape(coords_txyc[0,:,:,0].shape).cpu().detach().numpy()
            self.slowness_array = slow_vals_array

            wavefronts_txy = ground_truth['wavefronts_txy'].cuda() 
            first_video_image_xy = ground_truth['video_txy'][0].cuda() # First batch, first image
            fig0 = train_utils.plot_waveform_tensors(self.model, coords_txyc, wavefronts_txy, first_video_image_xy)
            self.logger.experiment.add_figure(f'valchunk0/waveforms', fig0, self.current_epoch)

            # Squared slowness estimate is independent of batch, so only calculate this once
            if self.wavespeed_loss_scale not in [None, 0, 0.]:
                # Calcuate and plot the inferred squared_slowness field in x and y
                img_aspect = coords_txyc.shape[2]/coords_txyc.shape[3]
                fig2 = plt.figure(figsize=(10,5))
                img = plt.imshow(slow_vals_array.T)
                plt.title('Squared slowness estimate')
                if img_aspect > 1:
                    plt.colorbar(img, fraction=0.046, pad=0.15, orientation='horizontal')
                else:
                    plt.colorbar(img, fraction=0.046, pad=0.04, orientation='vertical')
                self.logger.experiment.add_figure('squared_slowness', fig2, self.current_epoch)
                self.slowness_model.cuda()
        avg_loss = sum(x["val_loss"] for x in outputs) / len(outputs)
        # Pass the accuracy to the `DictLogger` via the `'log'` key.
        tensorboard_logs = {'val/avg_loss': avg_loss}
        return {"val_loss": avg_loss, "log":tensorboard_logs}

    def on_epoch_end(self):
        # Add 10% of time samples to the current training batch each epoch
        self.wf_train_dataset.included_time_fraction = min(1.0, self.wf_train_dataset.included_time_fraction + 0.1)

    def configure_optimizers(self):
        if self.wavespeed_loss_scale not in [None, 0, 0.]:
            return Adam(itertools.chain(self.model.parameters(), self.slowness_model.parameters()), lr=self.learning_rate)
        else:
            return Adam(self.model.parameters(), lr=self.learning_rate)
    
    def setup(self, stage):        
        if self.wf_valid_dataset is None:
            self.wf_valid_dataset = WavefrontSupervisionDataset(self.video_filepath)
        
        if self.wf_train_dataset is None:
            self.wf_train_dataset = MaskedWavefrontDataset(self.wf_valid_dataset)

    def train_dataloader(self):
                                           # Shuffling is handled by the chunk_dataset when sample_fraction < 1.0
        return torch.utils.data.DataLoader(self.wf_train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.wf_valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)