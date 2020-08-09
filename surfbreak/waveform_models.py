import os
import pkg_resources
import shutil
import itertools
import copy

import matplotlib.pyplot as plt

import pytorch_lightning as pl
from pytorch_lightning import Callback
import torch
import torch.nn.functional as F
from torch.optim import Adam
import torch.utils.data
from surfbreak import train_utils
import wandb

import optuna
from optuna.integration import PyTorchLightningPruningCallback

from surfbreak.loss_functions import wave_pml 
from surfbreak.datasets import WavefrontDatasetTXYC, WavefrontCNNDatasetCXY
from surfbreak import base_models, diff_operators

from surfbreak.loss_functions import wave_pml_2


class WaveformNet(pl.LightningModule):    # With no gradient or wave loss, oemega values of (1, 5) work well (best at epochs 3~5)  
    def __init__(self, hidden_features=256, hidden_layers=3, first_omega_0=2.5, hidden_omega_0=11, squared_slowness=0.2,
                 learning_rate=1e-4, wavefunc_loss_scale=5.5e-9, wfloss_growth_scale=1.0, wavespeed_norm_loss_scale=4e-4, wavespeed_delta_loss_scale=1e-6, 
                 wavespeed_first_omega_0=3.5, wavespeed_hidden_omega_0=15, pretrain_epochs=3, 
                 video_filepath=None, train_dataset=None, valid_dataset=None, viz_dataset=None, batch_size=64):
        """Learns a low-dimensional function which maps t,x,y video coordinates to waveform magnitude. 
           The model applies a physics-based waveform cost function to regularize the signal, using a 2nd-order PDE (see the SIREN paper).
           The model also jointly inferrs the static wave propogation velocity field used the waveform loss (in units of squared slowness).             
           Setting wavespeed_loss_scale to None will result in assuming the uniform, static wave velocity given in squared_slowness
           """
        super().__init__()
        if video_filepath is None:
            assert train_dataset is not None and valid_dataset is not None, "Either a video_filepath or datasets must be provided."
        self.save_hyperparameters('first_omega_0', 'hidden_omega_0', 'squared_slowness', 'wavefunc_loss_scale', 'wfloss_growth_scale',
                                  'wavespeed_norm_loss_scale', 'wavespeed_delta_loss_scale',
                                  'hidden_features', 'hidden_layers', 'learning_rate', 'batch_size', 'pretrain_epochs',
                                  'wavespeed_first_omega_0', 'wavespeed_hidden_omega_0')
        self.model = base_models.Siren(in_features=3, 
                                 out_features=1, 
                                 hidden_features=hidden_features,
                                 hidden_layers=hidden_layers, outermost_linear=True,
                                 first_omega_0=first_omega_0,
                                 hidden_omega_0=hidden_omega_0)

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
        self.wfloss_growth_scale= wfloss_growth_scale
        self.wavespeed_norm_loss_scale=wavespeed_norm_loss_scale
        self.wf_train_dataset = train_dataset
        self.wf_valid_dataset = valid_dataset
        self.viz_dataset = viz_dataset
        self.video_filepath = video_filepath
        self.batch_size = batch_size
        self.inferred_slowness=None
        self.val_fig=None
        self.best_val_loss=None
        self.pretrain_epochs=pretrain_epochs
        self.wavespeed_delta_loss_scale = wavespeed_delta_loss_scale
        self.prev_slowness_model = None # Copy of last epoch's model so that rapid changes to output can be penalized

        self.example_input_array = torch.ones(1,1337,3)

    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, batch_nb):
        model_input, ground_truth = batch
        wf_values_out, coords_out = self.model(model_input['coords_sc'])

         ####### Primary waveform (space-time) loss calculations ########
        ## Calculate the basic mean squared error loss using the training data
        mse_loss = F.mse_loss(wf_values_out, ground_truth['wavefront_values_sc'])
        # Enforce zero mean amplitude at each time (=each batch), not just over the entire x,y,t video cube!
        avg_wfval_loss = (wf_values_out**2).mean(dim=1).sum() * 1e-6 # Gentle pressure to have zero amplitude at each timestep.
        

        ####### wavespeed (spatial) loss functions  ########
        ## Calculate the loss used to jointly learn the wave velocity field 
        slow_vals_out, _ = self.slowness_model(model_input['coords_sc'][...,1:]) # Omit the first channel (time)
        slow_vals_out_tensor = slow_vals_out.repeat(1,1,3)
        # Gently push towards known a good value for squared_slowness
        wavespeed_norm_loss =  (slow_vals_out - self.squared_slowness).abs().mean()*self.wavespeed_norm_loss_scale

        if self.prev_slowness_model is not None:
            prev_slow_vals_out, _ = self.prev_slowness_model(model_input['coords_sc'][...,1:]) # Omit the first channel (time)
            wavespeed_delta_loss = F.mse_loss(slow_vals_out, prev_slow_vals_out.detach())*self.wavespeed_delta_loss_scale # Penalize large changes from last epoch's model
        else:
            wavespeed_delta_loss = 0

        ###### wavefunction normalization (slow!) for learning wave field and regularizing waveform ########
        if self.current_epoch >= self.pretrain_epochs:
            squared_slowness_tensor = slow_vals_out_tensor
            # Calculate the physics-based wave function loss
            wave_loss_dict = wave_pml_2(wf_values_out, coords_out, squared_slowness_tensor)
            wavefunc_loss = wave_loss_dict['diff_constraint_hom']*self.wavefunc_loss_scale # * min(1, (step/ total_steps)**2)
        else:
            wavefunc_loss = 0

        #TODO: Add in an (optional) loss term that encourages a smooth, low-frequency wavespeed spatial field 
        #grad_loss    =  diff_operators.gradient(slow_vals_out, slow_coords_out).abs().mean()*self.grad_loss_scale
        #laplace_loss =  diff_operators.laplace(slow_vals_out, slow_coords_out).abs().mean()*self.grad_loss_scale*0.1
        #log_dict['train/ws_grad_loss'] = grad_loss
        #log_dict['train/ws_laplace_loss'] = laplace_loss
       
        train_loss = mse_loss + avg_wfval_loss + wavefunc_loss + wavespeed_norm_loss + wavespeed_delta_loss

        log_dict = {
            'train/loss': train_loss,
            'train/mse_loss': mse_loss,
            'train/avg_wfval_loss': avg_wfval_loss,
            'train/wavefunc_loss': wavefunc_loss,
            'train/wavespeed_norm_loss': wavespeed_norm_loss,
            'train/wavespeed_delta_loss': wavespeed_delta_loss,
            #'train/included_time_fraction': self.wf_train_dataset.included_time_fraction
        } 
        return {'loss': train_loss, 'log': log_dict}

    def validation_step(self, batch, batch_nb):
        model_input, ground_truth = batch
        wf_values_out, _ = self.model(model_input['coords_sc'])
        if batch_nb < 5:
            wandb.log({"diagnostics/tcoords": wandb.Histogram(model_input['coords_sc'][...,0].cpu())}, commit=False)
            wandb.log({"diagnostics/xcoords": wandb.Histogram(model_input['coords_sc'][...,1].cpu())}, commit=False)
            wandb.log({"diagnostics/ycoords": wandb.Histogram(model_input['coords_sc'][...,2].cpu())}, commit=False)
            wandb.log({"diagnostics/wf_values_out": wandb.Histogram(wf_values_out.cpu())}, commit=False) 
        loss = F.mse_loss(wf_values_out, ground_truth['wavefront_values_sc'])
        return {'val_loss':loss}

    def validation_epoch_end(self, outputs):
        log_dict = {}
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
            log_dict["waveforms"] = fig0
            self.val_fig = fig0

            # Squared slowness estimate is independent of batch, so only calculate this once
            # Calcuate and plot the inferred squared_slowness field in x and y
            img_aspect = coords_txyc.shape[2]/coords_txyc.shape[3]
            fig2 = plt.figure(figsize=(10,5))
            img = plt.imshow(slow_vals_array.T)
            plt.title('Squared slowness estimate')
            if img_aspect > 1:
                plt.colorbar(img, fraction=0.046, pad=0.15, orientation='horizontal')
            else:
                plt.colorbar(img, fraction=0.046, pad=0.04, orientation='vertical')
            log_dict["squared_slowness"] = fig2
            self.slowness_model.cuda()
        avg_loss = sum(x["val_loss"] for x in outputs) / len(outputs)
        plt.close('all') 
        # Pass the accuracy to the `DictLogger` via the `'log'` key.
        log_dict['val/avg_loss'] = avg_loss
        if self.best_val_loss is None or avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            wandb.run.summary["best_val_loss"] = self.best_val_loss
        return {"val_loss": avg_loss, "log":log_dict}

    def on_epoch_end(self):
        # Add 10% of time samples to the current training batch each epoch
        self.prev_slowness_model = copy.deepcopy(self.slowness_model)
        #self.wf_train_dataset.included_time_fraction = min(1.0, self.wf_train_dataset.included_time_fraction + 0.1)
        if self.current_epoch >= self.pretrain_epochs:
            self.wavefunc_loss_scale *= self.wfloss_growth_scale
        wandb.log({'train/wavefunc_loss_scale': self.wavefunc_loss_scale}, commit=False)

    def configure_optimizers(self):
        return Adam(itertools.chain(self.model.parameters(), self.slowness_model.parameters()), lr=self.learning_rate)
    
    def setup(self, stage):        
        assert self.wf_valid_dataset is not None and self.wf_train_dataset is not None, "Default datasets not yet imlemented"
        
    def train_dataloader(self):
                                           # Shuffling is handled by the chunk_dataset when sample_fraction < 1.0
        return torch.utils.data.DataLoader(self.wf_train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=6)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.wf_valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=6)



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

        log_dict = {'train/loss':loss,
                            'train/mse_loss': mse_loss,
                            'train/avg_loss': zeromean_loss}
        return {'loss': loss, 'log': log_dict}

    def validation_step(self, batch, batch_nb):
        model_input, ground_truth = batch
        model_output = self.model(model_input)
        loss = F.mse_loss(model_output, ground_truth)
        return {'val_loss':loss}

    def validation_epoch_end(self, outputs):
        avg_loss = sum(x["val_loss"] for x in outputs) / len(outputs)
        log_dict = {'val/avg_loss': avg_loss}
        return {"val_loss": avg_loss, "log":log_dict}

    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=self.learning_rate)
    
    def setup(self, stage):
        if self.train_dataset is None:
            self.train_wfs_dataset = WavefrontDatasetTXYC(self.video_filepath, ydim=self.ydim, timerange=self.timerange,
                                                                time_chunk_duration_s=self.chunk_duration, time_chunk_stride_s=self.chunk_stride)
            self.train_dataset = WavefrontCNNDatasetCXY(self.train_wfs_dataset)
        if self.val_dataset is None:
            self.val_wfs_dataset = WavefrontDatasetTXYC(self.video_filepath, ydim=self.ydim, timerange=self.val_timerange,
                                                                time_chunk_duration_s=self.chunk_duration, time_chunk_stride_s=self.chunk_stride)
            self.val_dataset = WavefrontCNNDatasetCXY(self.val_wfs_dataset)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)