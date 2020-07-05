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

def raw_wavefront_array_to_txy_tensor(wavefront_array, ydim_out, duration_s=30, time_axis_scale=0.5, SAMPLING_HZ=10,
                                      clip_max=None):

    waveform_array_yxt = wavefront_array[:,:,:duration_s*SAMPLING_HZ]
    assert waveform_array_yxt.shape[2] == duration_s*SAMPLING_HZ

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

    return resized_tensor[0,0,...]


def get_wavefront_tensor_txy(ydim_out, slice_xrange=(30,90), output_dim=3, start_s=0, duration_s=30, time_axis_scale=0.5,
                             target_graph_key="result"):
    """Supplying target_traph_key='clipped_image_tensor' will give an equivalently scaled version of the raw video instead """

    waveform_slice_graph = pipelines.video_to_waveform_tensor('../tmp/shirahama_1590387334_SURF-93cm.ts', ydim_out=ydim_out,
                                                                duration_s=duration_s, start_s=start_s,
                                                                slice_xrange=slice_xrange, output_dim=output_dim,
                                                                time_axis_scale=time_axis_scale)
    
    return graphchain.get(waveform_slice_graph, target_graph_key)


def get_wavefront_tensor_txy_old(ydim_out, slice_xrange=(30,90), output_dim=3, start_s=0, duration_s=30, time_axis_scale=0.5,
                             target_graph_key="result"):
    """Supplying target_traph_key='clipped_image_tensor' will give an equivalently scaled version of the raw video instead """

    # Get a little more than the required duration, then clip to the appropriate length
    # (pre-processing with delta-time result 2 less samples)
    waveform_slice_graph = pipelines.video_to_waveform_slice('../tmp/shirahama_1590387334_SURF-93cm.ts',
                                                                duration_s=duration_s+1, start_s=start_s,
                                                                slice_xrange=slice_xrange, output_dim=output_dim)

    waveform_array_yxt = graphchain.get(waveform_slice_graph, target_graph_key)
    
    # SAMPLING_HZ must match that defined within preprocessing pipeline steps (10hz)
    output_wavefront_tensor = raw_wavefront_array_to_tensor(waveform_array_yxt, ydim_out=ydim_out, duration_s=duration_s, 
                                                            time_axis_scale=time_axis_scale, SAMPLING_HZ=10)

    return output_wavefront_tensor


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
    def __init__(self, ydim, xrange=(30,90), timerange=(0,61), time_chunk_duration_s=30, time_chunk_stride_s=15, time_axis_scale=0.5):
        super().__init__()
        start_s, end_s = timerange
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
            full_wf_tensor = get_wavefront_tensor_txy(self.ydim, slice_xrange=self.xrange, output_dim=3,
                                                start_s=item_start_s, duration_s=(item_end_s - item_start_s),
                                                time_axis_scale=self.time_axis_scale, target_graph_key="result")
            full_vid_tensor = get_wavefront_tensor_txy(self.ydim, slice_xrange=self.xrange, output_dim=3,
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
