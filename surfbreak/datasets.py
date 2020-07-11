# Reference tests: 08_training_datasets.ipynb (unless otherwise specified).

__all__ = ['normalize_tensor', 'get_wavefront_tensor_txy', 'get_mgrid', 'WaveformVideoDataset',
           'subsample_strided_buckets', 'WaveformChunkDataset']

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
        return ((tensor - tensor.mean()) / tensor.std()).clip(max=clip_max)

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


def get_wavefront_tensor_txy(video_filepath, ydim_out, slice_xrange=(30,90), output_dim=3, start_s=0, duration_s=30, time_axis_scale=0.5,
                             target_graph_key="result"):
    """Supplying target_traph_key='clipped_image_tensor' will give an equivalently scaled version of the raw video instead """

    waveform_slice_graph = pipelines.video_to_waveform_tensor(video_filepath, ydim_out=ydim_out,
                                                                duration_s=duration_s, start_s=start_s,
                                                                slice_xrange=slice_xrange, output_dim=output_dim,
                                                                time_axis_scale=time_axis_scale)
    
    return torch.from_numpy(graphchain.get(waveform_slice_graph, target_graph_key))


# Cell
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


class WaveformVideoDataset(Dataset):
    def __init__(self, video_filepath, ydim, xrange=(30,90), timerange=(0,61), time_chunk_duration_s=30, time_chunk_stride_s=15, time_axis_scale=0.5):
        super().__init__()
        start_s, end_s = timerange
        self.video_filepath = video_filepath
        self.ydim = ydim
        self.xrange = xrange
        self.time_axis_scale = time_axis_scale
        self.time_chunk_duration_s = time_chunk_duration_s
        self.time_chunk_stride_s = time_chunk_stride_s
        self.t_coords_per_second = 10 * time_axis_scale

        self.video_chunk_timeranges = np.array(list((start, start+time_chunk_duration_s) for start in
                                                    range(start_s, end_s + 1 - time_chunk_duration_s, time_chunk_stride_s)))
        self.cached_chunk_data = None
        self.cached_chunk_idx = None
        
    def __len__(self):
        return len(self.video_chunk_timeranges)

    def __getitem__(self, idx):
        if idx != self.cached_chunk_idx:
            item_start_s, item_end_s = self.video_chunk_timeranges[idx]
            full_wf_tensor = get_wavefront_tensor_txy(self.video_filepath, self.ydim, slice_xrange=self.xrange, output_dim=3,
                                                start_s=item_start_s, duration_s=(item_end_s - item_start_s),
                                                time_axis_scale=self.time_axis_scale, target_graph_key="result")
            full_vid_tensor = get_wavefront_tensor_txy(self.video_filepath, self.ydim, slice_xrange=self.xrange, output_dim=3,
                                                start_s=item_start_s, duration_s=(item_end_s - item_start_s),
                                                time_axis_scale=self.time_axis_scale, target_graph_key="scaled_video_tensor")
            # For now, abstract time coordinates will be in centered minutes (so a 2 minute video spans -1 to 1, and a 30 minute video spans -15 to 15)
            full_duration_s = self.video_chunk_timeranges.max() - self.video_chunk_timeranges.min()
            full_tcoord_range = -((full_duration_s/60) / 2), ((full_duration_s/60) / 2)

            this_chunk_tcoord_range = [self.video_chunk_timeranges[idx][0]/60 - full_tcoord_range[1],
                                    self.video_chunk_timeranges[idx][1]/60 - full_tcoord_range[1]]

            all_coords = get_mgrid(full_wf_tensor.shape, tcoord_range=this_chunk_tcoord_range)

            model_input = {
                'coords_txyc':all_coords.reshape(*full_vid_tensor.shape,3)
            }

            assert full_vid_tensor.shape == full_wf_tensor.shape
            ground_truth = {
                "video_txy": full_vid_tensor,
                "wavefronts_txy": full_wf_tensor,
                "timerange": (item_start_s, item_end_s),

            }
            self.cached_chunk_idx = idx
            self.cached_chunk_data = (model_input, ground_truth)
        else:
            model_input, ground_truth = self.cached_chunk_data
            
        return model_input, ground_truth


# Cell
def subsample_strided_buckets(txyc_tensor, bucket_sidelength, samples_per_bucket=100,
                              return_xy_buckets=False, sample_offset=0, filter_std_below=0.2):
    """Samples channel values form yxt tensors along bucketed spatial (x,y) dimensions.
       Leaves the time dimension the same. Return a tensor of dimensions (time, channel, buckets, samples)
       Input:  tensor (time, x, y, channels)
       Output: tensor (bucket, samples, time, channels)
               OR (x,y,s,t,c) if return_xy_buckets=True

       Only samples from the first element in the batch
       if return_xy_buckets=True, buckets are indexed by x and ycoordinates"""

    tcxy_tensor = txyc_tensor.permute(0,3,1,2)
    tdim, cdim, xdim, ydim = tcxy_tensor.shape
    stride = bucket_sidelength

    # Fold: Expands a rolling window of kernel_size, with overlap between windows if stride < kernel_size
    #       input tensor of shape (N,C,T) , where N is the batch dimension, C is the channel dimension,
    #       and * represent arbitrary spatial dimensions
    #          See https://pytorch.org/docs/master/generated/torch.nn.Unfold.html#torch.nn.Unfold
    uf = F.unfold(tcxy_tensor,kernel_size=bucket_sidelength, stride=stride)
    n_buckets = uf.shape[-1]
    tcsb = uf.reshape(tdim, cdim, -1, n_buckets)
    n_total_samples = tcsb.shape[2]

    subsample_stride = n_total_samples//samples_per_bucket
    offset_idx = sample_offset%subsample_stride
    tcsb_sampled = tcsb[...,:offset_idx + samples_per_bucket*subsample_stride:subsample_stride,:]

    if return_xy_buckets:
        tcsxy = tcsb_sampled.reshape(tdim, cdim, tcsb_sampled.shape[2], xdim//stride, ydim//stride)
        xystc = tcsxy.permute(3,4,2,0,1)
        return xystc
    else:
        bstc = tcsb_sampled.permute(3,2,0,1)
        return bstc

class WaveformChunkDataset(Dataset):
    def __init__(self, wf_video_dataset, video_index=None, xy_bucket_sidelen=10, samples_per_xy_bucket=10, time_sample_interval=4,
                 steps_per_video_chunk=1000, bucket_mask_minstd=0.2, sample_fraction=1.0):
        self.wf_video_dataset = wf_video_dataset
        self.video_index = video_index
        self.xy_bucket_sidelen = xy_bucket_sidelen
        self.samples_per_xy_bucket = samples_per_xy_bucket
        self.time_sample_interval = time_sample_interval
        self.steps_per_video_chunk = steps_per_video_chunk
        self.bucket_mask_minstd = bucket_mask_minstd
        self.sample_fraction = sample_fraction
    
    def __len__(self):
        return len(self.wf_video_dataset) * self.steps_per_video_chunk

    def __getitem__(self, idx):
        if self.sample_fraction < 0.9999:
            # Get a single index
            idx = torch.randint(low=0, high=int(len(self.wf_video_dataset) * self.steps_per_video_chunk * self.sample_fraction), size=(1,))

        if self.video_index is None:
            video_idx = int(idx // self.steps_per_video_chunk)
        else:
            video_idx = self.video_index
        t_idx = idx % self.steps_per_video_chunk
        
        model_input, ground_truth = self.wf_video_dataset[video_idx] 

        all_wf_values_txyc = ground_truth['wavefronts_txy'][..., None]
        all_coords_txyc = model_input['coords_txyc']

        xy_subsampled_wf_values_bstc = subsample_strided_buckets(all_wf_values_txyc,
                                                                 bucket_sidelength=self.xy_bucket_sidelen,
                                                                 samples_per_bucket=self.samples_per_xy_bucket,
                                                                 sample_offset=t_idx)
        xy_subsampled_coords_bstc =    subsample_strided_buckets(all_coords_txyc,
                                                                 bucket_sidelength=self.xy_bucket_sidelen,
                                                                 samples_per_bucket=self.samples_per_xy_bucket,
                                                                 sample_offset=t_idx)

        ti = self.time_sample_interval

        subsampled_wf_values_bstc = xy_subsampled_wf_values_bstc[:,:,t_idx%ti::ti,:]
        subsampled_coords_bstc =       xy_subsampled_coords_bstc[:,:,t_idx%ti::ti,:]

        # Filters out buckets with low standard deviation (probably just background ocean pixels - NOT waves)
        n_buckets = subsampled_wf_values_bstc.shape[0]
        bucket_mask = subsampled_wf_values_bstc.reshape(n_buckets,-1).std(dim=1) > self.bucket_mask_minstd

        model_input = {
            'coords': subsampled_coords_bstc.reshape(-1, 3),
            'masked_coords': subsampled_coords_bstc[bucket_mask].reshape(-1, 3)
        }

        assert subsampled_wf_values_bstc.shape[:3] == subsampled_coords_bstc.shape[:3]

        ground_truth = {
            "wavefront_values": subsampled_wf_values_bstc.reshape(-1,1),
            "masked_wf_values": subsampled_wf_values_bstc[bucket_mask].reshape(-1,1),
            "bst_shape": subsampled_wf_values_bstc.shape[:3],
            "masked_bst_shape": subsampled_wf_values_bstc[bucket_mask].shape[:3],
            "timerange": ground_truth['timerange'],
            'time_sampling_offset': t_idx%ti

        }

        return model_input, ground_truth


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

class InferredWaveformDataset(Dataset):
    def __init__(self, video_filepath, trained_waveform_model, ydim, xrange=(30,90), timerange=(0,61), time_chunk_duration_s=30, time_chunk_stride_s=15, time_axis_scale=0.5,
                 tmpdir='./tmp/dataset_cache'):
        super().__init__()
        start_s, end_s = timerange
        self.video_filepath = video_filepath
        self.ydim = ydim
        self.xrange = xrange
        self.time_axis_scale = time_axis_scale
        self.time_chunk_duration_s = time_chunk_duration_s
        self.time_chunk_stride_s = time_chunk_stride_s
        self.t_coords_per_second = 10 * time_axis_scale
        self.tmpdir=tmpdir
        self.waveform_model = trained_waveform_model
        
        self.wf_train_video_dataset = WaveformVideoDataset(video_filepath, ydim=ydim, xrange=xrange, timerange=timerange, time_chunk_duration_s=time_chunk_duration_s, 
                                                           time_chunk_stride_s=time_chunk_stride_s, time_axis_scale=time_axis_scale)
        
        self.fname_pairs = cache_video_dataset_as_inferred_xy_images(self.wf_train_video_dataset, self.waveform_model, tmpdir=self.tmpdir)        
        
    def __len__(self):
        return len(self.fname_pairs)

    def __getitem__(self, idx):
        vid_fname, wf_fname = self.fname_pairs[idx]
        with open(os.path.join(self.tmpdir, vid_fname), 'rb') as f:
            model_input_cxy = torch.from_numpy(pickle.load(f)) # Input has a channel dimension already
        with open(os.path.join(self.tmpdir, wf_fname), 'rb') as f:
            ground_truth_cxy = torch.from_numpy(pickle.load(f))[None,...] # Add empty channel dimension for now
        assert model_input_cxy.shape[1:] == ground_truth_cxy.shape[1:]
 
        return (trim_img_to_nearest_multiple(model_input_cxy,  divisor=4), # Must be divisible by 4 - just crop for now.
                trim_img_to_nearest_multiple(ground_truth_cxy, divisor=4))  # video_image_xy, inferred_waveform_xy
    
def trim_img_to_nearest_multiple(tensor, divisor=4):
    max_xdim = tensor.shape[-2] - tensor.shape[-2]%divisor  # Must be divisible by 4 - just crop for now. 
    max_ydim = tensor.shape[-1] - tensor.shape[-1]%divisor  # Must be divisible by 4 - just crop for now. 
    return tensor[..., :max_xdim,:max_ydim]

def detect_wavefronts(wf_cnn_checkpoint, video_tensor):
    wavecnn_model = LitWaveCNN.load_from_checkpoint(wf_cnn_checkpoint)
    # Append the second channel with time-delta intensity
    input_img_array_tcxy = np.concatenate((video_tensor[:-1][None,...], 
                                          video_tensor[1:][None,...] - video_tensor[:-1][None,...]), axis=0).transpose(1,0,2,3)
    return wavecnn_model(torch.from_numpy(input_img_array_tcxy))

def get_trimmed_tensor(wf_labeling_training_video, start_s=5, duration_s=1.2, time_axis_scale=0.5):
    from surfbreak.pipelines import video_to_trimmed_tensor
    wf_graph = video_to_trimmed_tensor(wf_labeling_training_video, start_s=start_s, duration_s=duration_s, time_axis_scale=time_axis_scale)
    return graphchain.get(wf_graph, 'result')

from surfbreak.supervision import wavefront_diff_tensor

class CNNChunkDataset(Dataset):
    def __init__(self, video_filepath, wavecnn_ckpt=None, ydim=150, timerange=(0,61), time_chunk_duration_s=1, time_chunk_stride_s=1, time_axis_scale=0.5):
        super().__init__()
        from surfbreak.waveform_models import LitWaveCNN
        self.video_filepath = video_filepath
        self.wavecnn_ckpt = wavecnn_ckpt
        self.ydim = ydim
        self.time_axis_scale = time_axis_scale
        self.time_chunk_duration_s = time_chunk_duration_s
        self.time_chunk_stride_s = time_chunk_stride_s
        self.t_coords_per_second = 10 * time_axis_scale
        self.average_wavefront_xy = np.zeros(1)
        self.std_wavefront_xy = np.zeros(1)

        start_s, end_s = timerange
        self.video_chunk_timeranges = np.array(list((start, start+time_chunk_duration_s) for start in
                                                    range(start_s, end_s + 1 - time_chunk_duration_s, time_chunk_stride_s)))
        item_start_s, item_end_s = self.video_chunk_timeranges[0]
        self.first_raw_vid_tensor = get_trimmed_tensor(self.video_filepath, start_s=item_start_s, duration_s=(item_end_s - item_start_s),
                                                       time_axis_scale=self.time_axis_scale)
        
        if self.wavecnn_ckpt is None:
            self.wavecnn_model = None
        else:
            self.wavecnn_model = LitWaveCNN.load_from_checkpoint(wavecnn_ckpt, wf_model_checkpoint=None).cuda()
        
        # Accumulate statistics for the wavefront images (mean and standard deviation)
        firstout, firstgt = self[0]
        acc_avg_wf_img = np.zeros_like(firstgt['wavefronts_txy'].mean(axis=0))
        acc_std_wf_img = np.zeros_like(firstgt['wavefronts_txy'])
        for model_in, model_gt in self:
            acc_avg_wf_img += model_gt['wavefronts_txy'].mean(axis=0)
            acc_std_wf_img = np.concatenate((acc_std_wf_img, model_gt['wavefronts_txy']), axis=0)
        self.average_wavefront_xy = acc_avg_wf_img / len(self)
        self.std_wavefront_xy = acc_std_wf_img.std(axis=0)
        
    def __len__(self):
        return len(self.video_chunk_timeranges)

    def __getitem__(self, idx):
        item_start_s, item_end_s = self.video_chunk_timeranges[idx]
        
        if self.wavecnn_model is None: # Use a simple time-delta of intensities if no CNN-based wave detector given
            vid_tensor_tplus3 = get_trimmed_tensor(self.video_filepath, start_s=item_start_s, 
                                               duration_s=(item_end_s - item_start_s + 3/self.t_coords_per_second), # Add one extra timestep here
                                               time_axis_scale=self.time_axis_scale)
            vid_tensor_txy = vid_tensor_tplus3[:-3] # Remove the extra timestep which was needed for the wavecnn input calculation

            wavecnn_label = normalize_tensor(wavefront_diff_tensor(vid_tensor_tplus3.transpose(1,2,0)).transpose(2,0,1), clip_max=1)
        
        else: # Do inference using a CNN
            vid_tensor_tplus1 = get_trimmed_tensor(self.video_filepath, start_s=item_start_s, 
                                               duration_s=(item_end_s - item_start_s + 1/self.t_coords_per_second), # Add one extra timestep here
                                               time_axis_scale=self.time_axis_scale)
            vid_tensor_txy = vid_tensor_tplus1[:-1] # Remove the extra timestep which was needed for the wavecnn input calculation
                                               
            wavecnn_input_tcxy = np.concatenate((vid_tensor_tplus1[:-1][None,...],  # Get a 2-channel representation (intensity, timedelta)
                                        vid_tensor_tplus1[1:][None,...] - vid_tensor_tplus1[:-1][None,...]), axis=0).transpose(1,0,2,3)            
            assert wavecnn_input_tcxy.shape[0] == vid_tensor_txy.shape[0] # Ensure length of time dimension is identical
            # Process the input video tensors one timestep at a time on the GPU, to avoid using too much memory
            frames_out = []
            for t in range(wavecnn_input_tcxy.shape[0]):
                this_wavecnn_input = torch.from_numpy(wavecnn_input_tcxy[t]).cuda()[None,...]
                frames_out.append(self.wavecnn_model(this_wavecnn_input)[:,0].detach().cpu().numpy()) # Remove the empty second channel dimension
            wavecnn_label = np.concatenate(frames_out, axis=0)

        # For now, abstract time coordinates will be in centered minutes (so a 2 minute video spans -1 to 1, and a 30 minute video spans -15 to 15)
        full_duration_s = self.video_chunk_timeranges.max() - self.video_chunk_timeranges.min()
        full_tcoord_range = -((full_duration_s/60) / 2), ((full_duration_s/60) / 2)

        this_chunk_tcoord_range = [self.video_chunk_timeranges[idx][0]/60 - full_tcoord_range[1],
                                   self.video_chunk_timeranges[idx][1]/60 - full_tcoord_range[1]]

        all_coords = get_mgrid(vid_tensor_txy.shape, tcoord_range=this_chunk_tcoord_range)

        model_input = {
            'coords_txyc':all_coords.reshape(*vid_tensor_txy.shape,3)
        }

        assert vid_tensor_txy.shape == wavecnn_label.shape
        ground_truth = {
            "video_txy": vid_tensor_txy,
            "wavefronts_txy": wavecnn_label - self.average_wavefront_xy,
            "timerange": (item_start_s, item_end_s),
            "wavefront_loss_mask": self.std_wavefront_xy > self.std_wavefront_xy.mean()
        }

        return model_input, ground_truth