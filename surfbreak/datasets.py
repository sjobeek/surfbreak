# Reference tests: 08_training_datasets.ipynb (unless otherwise specified).

# Cell
from surfbreak import graphutils, supervision
from surfbreak import pipelines
import graphchain
import dask
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

def normalize_tensor(tensor, clip_max=1):
    if clip_max is None:
        return ((tensor - tensor.mean()) / tensor.std())
    else:
        clipped_tensor = ((tensor - tensor.mean()) / tensor.std()).clip(max=clip_max)
        return clipped_tensor - clipped_tensor.mean()

def raw_wavefront_array_to_normalized_txy(wavefront_array, ydim_out, duration_s=30, time_axis_scale=0.5, SAMPLING_HZ=10,
                                      clip_max=None):

    waveform_array_yxt = wavefront_array[:,:,:int(duration_s*SAMPLING_HZ)]
    assert waveform_array_yxt.shape[2] == int(duration_s*SAMPLING_HZ)

    # Reshape to the standard channel axis order for learning (T, X, Y)
    waveform_array_txy = np.transpose(waveform_array_yxt, (2,1,0))

    # Incredibly important to clip the large peak values in this raw waveform_array,
    # since it's mostly a binary indicator of where a wave-foam front was detected
    waveform_tensor = torch.from_numpy(normalize_tensor(waveform_array_txy.astype('float32'), clip_max=clip_max))
    _, xdim_in, ydim_in = waveform_tensor.shape
    xdim_out = int(ydim_out * (xdim_in / ydim_in))
    tdim_out = int(duration_s * SAMPLING_HZ * time_axis_scale)

    # [None, None,...] nonsense below is adding and removing [batch, channel] dimensions required by F.interpolate
    resized_tensor = F.interpolate(waveform_tensor[None,None,...], size=(tdim_out, xdim_out, ydim_out),
                                   mode='trilinear', align_corners=False)

    return resized_tensor[0,0,...].numpy()


def get_mgrid(sidelen_tuple, tcoord_range=None):
    '''Generates a flattened grid of (t,x,y) coordinates in a range of -1 to 1.
    sidelen_tuple: tuple of coordinate side lengths (t,x,y)
    '''
    xscale = sidelen_tuple[1]/sidelen_tuple[2]
    if tcoord_range is None:
        tensors = tuple([torch.linspace(-1, 1, steps=sidelen_tuple[0]),
                         torch.linspace(-xscale, xscale, steps=sidelen_tuple[1]),
                         torch.linspace(-1, 1, steps=sidelen_tuple[2])])
    else:
        tensors = tuple([torch.linspace(*tcoord_range, steps=sidelen_tuple[0]),
                         torch.linspace(-xscale, xscale, steps=sidelen_tuple[1]),
                         torch.linspace(-1, 1, steps=sidelen_tuple[2])])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, len(sidelen_tuple))
    return mgrid


import pickle
import os

def video_txy_to_wavecnn_array_cxy(video_txy, t):
    # Create an image with 2 channels - intensity, and the one-step intensity delta
    assert t < video_txy.shape[0]-1, "Cannot infer last frame (or beyond)"
    input_img_array_cxy = torch.cat((video_txy[t][None,...], 
                                     video_txy[t+1][None,...] - video_txy[t][None,...]), dim=0)
    return input_img_array_cxy


def cache_video_dataset_as_inferred_xy_images(video_dataset, model, tmpdir='./tmp/dataset_cache'):
    """Returns a list of filename tuples which are written as pickle files to the folder tmpdir"""
    if not os.path.exists(tmpdir):
            os.makedirs(tmpdir)
    cached_image_pair_filenames = []
    for chunk_idx, (m_input, gt) in enumerate(video_dataset):
        n_timesteps_this_chunk = m_input['coords_txyc'].shape[0]
        assert n_timesteps_this_chunk == gt['video_txy'].shape[0]
        for t_idx in range(n_timesteps_this_chunk - 1): # Due to time delta, one less length
            # Infer the target x,y image at time t using the pre-trained model (as training target)
            waveform_array_xy, _ = model(m_input['coords_txyc'][t_idx])
            waveform_array_fname = f"c{chunk_idx}_t{t_idx}_waveform_xyc"
            with open(os.path.join(tmpdir, waveform_array_fname), 'wb') as f:
                pickle.dump(waveform_array_xy.detach().numpy()[...,0] , f) # [...,0] to drop trailing empty dimension
            # Also save the corresponding video image (as training input)
            img_array_fname =   f"c{chunk_idx}_t{t_idx}_video_xy"
            # Create an image with 2 channels - intensity, and the one-step intensity delta 
            input_img_array_cxy = video_txy_to_wavecnn_array_cxy(gt['video_txy'], t_idx)
            with open(os.path.join(tmpdir, img_array_fname), 'wb') as f:
                pickle.dump(input_img_array_cxy.numpy(), f)
            # Order is (train input, train target)
            cached_image_pair_filenames.append((img_array_fname, waveform_array_fname))
    return cached_image_pair_filenames


from surfbreak.train_utils import slugify
    
def trim_img_to_nearest_multiple(tensor, divisor=4):
    max_xdim = tensor.shape[-2] - tensor.shape[-2]%divisor  # Must be divisible by 4 - just crop for now. 
    max_ydim = tensor.shape[-1] - tensor.shape[-1]%divisor  # Must be divisible by 4 - just crop for now. 
    return tensor[..., :max_xdim,:max_ydim]

def get_trimmed_tensor(wf_labeling_training_video, graph_key='trimmed_wavefront_array', calibrate=True,
                       start_s=5, duration_s=1.2, time_axis_scale=0.5, ydim_out=128, xdim_max=None):
    from surfbreak.pipelines import video_to_trimmed_tensor
    wf_graph = pipelines.video_to_waveform_array_txy(wf_labeling_training_video, ydim_out=ydim_out, calibrate=calibrate,
                                                     start_s=start_s, duration_s=duration_s, time_axis_scale=time_axis_scale)
    txy_array = graphchain.get(wf_graph, graph_key) 
    if xdim_max is not None and txy_array.shape[1] > xdim_max:
        return trim_img_to_nearest_multiple(txy_array[:,:xdim_max], divisor=4) # TODO: Center this! 
    else:
        return trim_img_to_nearest_multiple(txy_array, divisor=4) # Ensure the dimensions are compativle with CNN dilation 

from surfbreak.supervision import wavefront_diff_tensor

class WavefrontDatasetTXYC(Dataset):
    def __init__(self, video_filepath, ydim=128, timerange=(0,60), time_chunk_duration_s=2, time_chunk_stride_s=2, time_axis_scale=0.5,
                 max_width=None, calibrate_h=True, wavecnn_ckpt=None):
        super().__init__()
        from surfbreak.waveform_models import LitWaveCNN
        self.video_filepath = video_filepath
        self.ydim = ydim
        self.max_width = max_width
        self.time_axis_scale = time_axis_scale
        self.time_chunk_duration_s = time_chunk_duration_s
        self.time_chunk_stride_s = time_chunk_stride_s
        self.t_coords_per_second = 10 * time_axis_scale
        self.average_wavefront_xy = torch.zeros(1)
        self.std_wavefront_xy = torch.zeros(1)
        self.calibrate_h=calibrate_h
        self.wavecnn_ckpt = wavecnn_ckpt
        if wavecnn_ckpt is None:
            self.wavecnn_model = None
        else:        
            self.wavecnn_model = LitWaveCNN.load_from_checkpoint(wavecnn_ckpt).cuda()

        start_s, end_s = timerange
        self.video_chunk_timeranges = np.array(list((start, start+time_chunk_duration_s) for start in
                                                    range(start_s, end_s + 1 - time_chunk_duration_s, time_chunk_stride_s)))
        item_start_s, item_end_s = self.video_chunk_timeranges[0]
        self.first_raw_vid_tensor = get_trimmed_tensor(self.video_filepath, ydim_out=self.ydim, calibrate=self.calibrate_h, 
                                                       start_s=item_start_s, duration_s=(item_end_s - item_start_s),
                                                       time_axis_scale=self.time_axis_scale, xdim_max=self.max_width)
                
        # Accumulate statistics for the wavefront images (mean and standard deviation)
        firstout, firstgt = self[0]
        acc_avg_wf_img = torch.zeros_like(firstgt['wavefronts_txy'].mean(axis=0))
        acc_std_wf_img = torch.zeros_like(firstgt['wavefronts_txy'])
        for model_in, model_gt in self:
            acc_avg_wf_img += model_gt['wavefronts_txy'].mean(axis=0)
            acc_std_wf_img = np.concatenate((acc_std_wf_img, model_gt['wavefronts_txy']), axis=0)
        self.average_wavefront_xy = acc_avg_wf_img / len(self)
        self.std_wavefront_xy = acc_std_wf_img.std(axis=0)
        
    def __len__(self):
        return len(self.video_chunk_timeranges)

    def __getitem__(self, idx):
        item_start_s, item_end_s = self.video_chunk_timeranges[idx]
        
        vid_array_txy = get_trimmed_tensor(self.video_filepath, graph_key='trimmed_video_array', calibrate=self.calibrate_h, 
                                            start_s=item_start_s, 
                                            duration_s=(item_end_s - item_start_s), # Add one extra timestep here
                                            time_axis_scale=self.time_axis_scale, ydim_out=self.ydim, xdim_max=self.max_width)

        wf_array_txy = get_trimmed_tensor(self.video_filepath, graph_key='trimmed_wavefront_array', calibrate=self.calibrate_h, 
                                            start_s=item_start_s, 
                                            duration_s=(item_end_s - item_start_s), # Add one extra timestep here
                                            time_axis_scale=self.time_axis_scale, ydim_out=self.ydim, xdim_max=self.max_width)
        wavefront_array_txy = normalize_tensor(wf_array_txy, clip_max=1)

        # For now, abstract time coordinates will be in centered minutes (so a 2 minute video spans -1 to 1, and a 30 minute video spans -15 to 15)
        full_duration_s = self.video_chunk_timeranges.max() - self.video_chunk_timeranges.min()
        full_tcoord_range = -((full_duration_s/60) / 2), ((full_duration_s/60) / 2)

        this_chunk_tcoord_range = [self.video_chunk_timeranges[idx][0]/60 - full_tcoord_range[1],
                                   self.video_chunk_timeranges[idx][1]/60 - full_tcoord_range[1]]

        all_coords = get_mgrid(vid_array_txy.shape, tcoord_range=this_chunk_tcoord_range)

     # Infer wavefronts if checkpoint available
        if self.wavecnn_model is None:
            wavefronts_txy = torch.from_numpy(wavefront_array_txy)
        else:
            wavecnn_input_tcxy = torch.stack((torch.from_numpy(vid_array_txy), 
                                              torch.from_numpy(wavefront_array_txy)), dim=0).permute(1,0,2,3)            
            assert wavecnn_input_tcxy.shape[0] == vid_array_txy.shape[0] # Ensure length of time dimension is identical
            # Process the input video tensors one timestep at a time on the GPU, to avoid using too much memory
            frames_out = []
            for t in range(wavecnn_input_tcxy.shape[0]):
                this_wavecnn_input = wavecnn_input_tcxy[t].cuda()[None,...]
                frames_out.append(self.wavecnn_model(this_wavecnn_input)[:,0].detach().cpu()) # Remove the empty second channel dimension
            wavecnn_out = torch.cat(frames_out, dim=0)
            assert vid_array_txy.shape == wavecnn_out.shape

            wavefronts_txy = wavecnn_out

        model_input = {
            'coords_txyc':all_coords.reshape(*vid_array_txy.shape,3).clone()
        }

        assert vid_array_txy.shape == wavefront_array_txy.shape
        
        ground_truth = {
            "video_txy": torch.tensor(vid_array_txy, dtype=torch.float),
            "wavefronts_txy": wavefronts_txy,
            "timerange": (item_start_s, item_end_s),
            "wavefront_loss_mask": torch.as_tensor(self.std_wavefront_xy > self.std_wavefront_xy.mean(), dtype=torch.bool)
        }

        return model_input, ground_truth

class WavefrontCNNDatasetCXY(Dataset):
    def __init__(self, wf_supervision_dataset):
        """Creates training data for the CNN-based wavefront inference"""
        self.wf_supervision_dataset = wf_supervision_dataset
        _, ground_truth = self.wf_supervision_dataset[0]
                                    # One less timestep per segment due to use of time-delta
        self.timesteps_per_segment = ground_truth['wavefronts_txy'].shape[0] - 1
        self.total_values_per_masked_image = ground_truth['wavefront_loss_mask'].sum()

    def __len__(self):
        return len(self.wf_supervision_dataset) * (self.timesteps_per_segment)
    
    def __getitem__(self, idx):
        segment_idx = int(idx // self.timesteps_per_segment)
        t_idx = idx % self.timesteps_per_segment
        _, ground_truth = self.wf_supervision_dataset[segment_idx]
        cnn_input_cxy = torch.stack((ground_truth['video_txy'][t_idx],
                                     ground_truth['wavefronts_txy'][t_idx]))
        # Learn to target the waveform from the NEXT timestep as a form of self-supervised regularization
        masked_waveform = (ground_truth['wavefront_loss_mask'] * ground_truth['wavefronts_txy'][t_idx + 1])[None, ]
                # Ensure image dimensions are a multiple of 4, for compatibility with CNN dilations
        return trim_img_to_nearest_multiple(cnn_input_cxy), trim_img_to_nearest_multiple(masked_waveform)
        

class MaskedWavefrontBatchesNC(Dataset):
    def __init__(self, wf_supervision_dataset, samples_per_batch=1000, included_time_fraction=1.0, resample_fraction=True):
        """Subsamples a (t,x,y,c) WaveformSupervisionDataset into individual (n,c) masked batches that can fit in memory"""
        self.wf_supervision_dataset = wf_supervision_dataset
        self.samples_per_batch = samples_per_batch
        self.included_time_fraction = included_time_fraction
        _, ground_truth = self.wf_supervision_dataset[0]
        self.timesteps_per_segment = ground_truth['wavefronts_txy'].shape[0] # Time coordinate
        self.total_values_per_masked_image = ground_truth['wavefront_loss_mask'].sum()
        self.batches_per_image = max(1, int(self.total_values_per_masked_image // samples_per_batch))
        self.resample_fraction=resample_fraction
    
    def __len__(self):
        return int(len(self.wf_supervision_dataset) * self.timesteps_per_segment * self.batches_per_image * self.included_time_fraction)

    def __getitem__(self, idx):
        if self.resample_fraction:
            # Get a single index
            idx = torch.randint(low=0, high=int(len(self)*self.included_time_fraction), size=(1,))
      
        segment_idx = int(idx // (self.timesteps_per_segment * self.batches_per_image))
        this_video_first_idx = int(segment_idx * self.timesteps_per_segment * self.batches_per_image)
        t_idx = int((idx - this_video_first_idx) // self.batches_per_image)
        batch_idx = idx - ( (segment_idx * self.timesteps_per_segment * self.batches_per_image) 
                            + (t_idx * self.batches_per_image))

        # Get the waveform values and coordinates for this timestep of this video segment
        model_input, ground_truth = self.wf_supervision_dataset[segment_idx]
        wf_values_xy = ground_truth['wavefronts_txy'][t_idx]
        coords_xyc = model_input['coords_txyc'][t_idx]
        
        # Mask and reshape to keep only useful values  
        xy_mask = ground_truth['wavefront_loss_mask']
        assert xy_mask.shape == wf_values_xy.shape
        xyc_mask = xy_mask[...,None].repeat(1,1,3) # Add in 3 coordinate channels for t,x,y
        assert xyc_mask.shape == coords_xyc.shape
        
        masked_wf_values_nc = wf_values_xy.reshape(-1,1)[xy_mask.reshape(-1,1)][...,None] # Add the one-dimensional channel dimension
        masked_coords_nc = coords_xyc.reshape(-1,1)[xyc_mask.reshape(-1,1)].reshape(-1,3)
        assert masked_wf_values_nc.shape[0] == masked_coords_nc.shape[0]
        assert self.total_values_per_masked_image == masked_wf_values_nc.shape[0]
        # Get evenly spaced samples from across the image
        subsampled_wf_values_sc = masked_wf_values_nc[batch_idx::self.total_values_per_masked_image//self.samples_per_batch,:][:self.samples_per_batch]
        subsampled_coords_sc =       masked_coords_nc[batch_idx::self.total_values_per_masked_image//self.samples_per_batch,:][:self.samples_per_batch]

        # TODO: Add in a small, uniformly-sampled group of coordinates, with corresponding zero-valued waveforms (as regularization)

        model_input = {
            #'coords_xyc': coords_xyc,
            'coords_sc': subsampled_coords_sc
        }

        assert subsampled_coords_sc.shape[0] == subsampled_wf_values_sc.shape[0]

        ground_truth = {
            #"wavefront_values_xy": wf_values_xy,
            "wavefront_values_sc": subsampled_wf_values_sc,
            "timerange": ground_truth['timerange'],
            "segment_idx":segment_idx,
            "t_idx": t_idx,
            "batch_idx": batch_idx
        }

        return model_input, ground_truth

from datetime import datetime
def cache_dataset_with_graphchain(dataset, cache_folder='./__graphchain_cache__'):
    now = datetime.now() # current date and time
    date_time = now.strftime("%m-%d-%Y_%H-%M-%S")
    dataset_foldername = 'dataset_'+date_time
    if not os.path.exists(os.path.join(cache_folder, dataset_foldername)):
            os.makedirs(os.path.join(cache_folder, dataset_foldername))
    tuple_filename_list = []
    for idx, result_tuple in enumerate(dataset):
        tuple_filename = os.path.join(cache_folder, dataset_foldername, 'tuple_'+str(idx)+'.pkl') 
        with open(tuple_filename, 'wb') as f:
                torch.save(result_tuple, f)
        tuple_filename_list.append(tuple_filename)
    return tuple_filename_list


class CachedDataset(Dataset):
    def __init__(self, dataset_class, *args, **kwargs):
        """Caches a dataset as pickles using a unique time-based filename,
           and will re-use previous cache if exists and the dataset class source hasn't changed"""
        from dask.compatibility import apply
        filename_graph = {'dataset':       (apply, dataset_class, args, kwargs), # equivalent to dataset_class(*args, **kwargs)
                          'output_tuples': (cache_dataset_with_graphchain, 'dataset')}  
        self.cached_tuple_filenames = graphchain.get(filename_graph, 'output_tuples')

    def __len__(self):
        return len(self.cached_tuple_filenames)

    def __getitem__(self, idx):
        tuple_filename = self.cached_tuple_filenames[idx]
        with open(tuple_filename, 'rb') as f:
                output = torch.load(f)
        return output