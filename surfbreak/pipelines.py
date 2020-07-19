#! /usr/bin/python3
from surfbreak import detection, load_videos, transform, supervision, datasets
import dask
import graphchain
import numpy
import pprint as pp

SURFSPOT_CALIBRATION_VIDEOS = {
    "shirahama": [
        '/home/erik/work/surfbreak/data/shirahama_1590387334_SURF-93cm.ts',
        '/home/erik/work/surfbreak/data/shirahama_1590312313_SURF-101cm.ts',
        '/home/erik/work/surfbreak/data/shirahama_1592790996_SURF-129cm.ts',
        '/home/erik/work/surfbreak/data/shirahama_1590479277_SURF-59cm.ts',
        '/home/erik/work/surfbreak/data/shirahama_1590378088_SURF-93cm.ts'
    ]
}

def vid_to_fit_mean_flow_graph(video_file, n_samples=10, duration_s=1, processes=4, draw_fit=False):
    """Detects the region of the image which contains breaking waves. 
        Returns a 3-tuple (mean_flow_xy_tensor, xrange, yrange):"""

    start_times = detection.get_sample_start_times(video_file, n_samples=n_samples)
    print("Video subsample start times (seconds):", start_times)
               
    flow_tensor_nodes = {f'flow_tensor_{idx}': (detection.avg_wave_flows, video_file, start_s, duration_s)
                         for idx, start_s in enumerate(start_times)}    

    result = {'result': (detection.fit_mean_flows, list(flow_tensor_nodes.keys()))}

    dask_graph = {**flow_tensor_nodes, **result}
    return dask_graph


def video_to_calibrated_image_tensor(video_filename, duration_s, start_s, surfspot=None, calibration_videos=None):
    """ Image tensors are 10hz by default (1/6th of the frames from a of 60Hz video)"""
    if surfspot is None and calibration_videos is None:
        filename_header = video_filename.split('/')[-1].split('_')[0]
        if filename_header in SURFSPOT_CALIBRATION_VIDEOS.keys():
            surfspot = filename_header
        else:
            raise ValueError('surfspot cannot be inferred from filename,'
                             'and both `surfspot` and `calibration_videos` are unspecified.\n'
                             'Please specify surfspot present in pipelines.SURFSPOT_CALIBRATION_VIDEOS.')
    if calibration_videos is None:
        calibration_videos = SURFSPOT_CALIBRATION_VIDEOS[surfspot]
    
    dask_graph = {
        'calibration_parameters': (transform.run_surfcam_calibration, calibration_videos),
        'result': (transform.video_file_to_calibrated_image_tensors, video_filename, 'calibration_parameters', duration_s, start_s)
    }
    return dask_graph
    

def video_to_waveform_slice(video_filename, duration_s, start_s, surfspot=None, calibration_videos=None, 
                            slice_xrange=(30,50), output_dim=2):
    """ Image tensors are 10hz by default (1/6th of the frames from a of 60Hz video)"""
    dask_graph = video_to_calibrated_image_tensor(video_filename, duration_s, start_s, 
                                                  surfspot=surfspot, calibration_videos=calibration_videos)
    dask_graph['image_tensor'] = dask_graph['result']
    dask_graph['clipped_image_tensor'] = (supervision.vertical_waveform_slice,'image_tensor', slice_xrange, output_dim)
    dask_graph['result'] = (supervision.generate_waveform_slice, 'image_tensor', slice_xrange, output_dim)
    return dask_graph


def video_to_waveform_tensor(video_filename, ydim_out, slice_xrange=(30,90),
                             start_s=0, duration_s=30, time_axis_scale=0.5, 
                             output_dim=3, calibration_videos=None, surfspot=None):
    
    # Get a little more than the required duration for the raw video, then clip to the appropriate length
    # (pre-processing with delta-time result 2 less samples)
    """ Image tensors are 10hz by default (1/6th of the frames from a of 60Hz video)"""
    dask_graph = video_to_calibrated_image_tensor(video_filename, duration_s+1, start_s, 
                                                  surfspot=surfspot, calibration_videos=calibration_videos)
    dask_graph['image_tensor'] = dask_graph['result']
    dask_graph['clipped_image_tensor'] = (supervision.vertical_waveform_slice,'image_tensor', slice_xrange, output_dim)
    dask_graph['scaled_video_tensor'] = (datasets.raw_wavefront_array_to_normalized_txy, 'clipped_image_tensor', ydim_out, duration_s, 
                                         time_axis_scale, 10) #SAMPLING_HZ
    dask_graph['waveform_slice'] = (supervision.generate_waveform_slice, 'image_tensor', slice_xrange, output_dim)
    dask_graph['result'] = (datasets.raw_wavefront_array_to_normalized_txy, 'waveform_slice', ydim_out, duration_s, 
                             time_axis_scale, 10, 1.0) #SAMPLING_HZ, clip_max (clipping wavefronts to max 1.0 very important)
    return dask_graph

def video_to_trimmed_tensor(video_filename, ydim_out=128,
                             start_s=0, duration_s=30, time_axis_scale=0.5, sampling_hz=10,
                             output_dim=3, calibration_videos=None, surfspot=None):
    
    # Get a little more than the required duration for the raw video, then clip to the appropriate length
    # (pre-processing with delta-time result 2 less samples)
    """ Image tensors are 10hz by default (1/6th of the frames from a of 60Hz video)"""
    dask_graph = video_to_calibrated_image_tensor(video_filename, duration_s, start_s,
                                                  surfspot=surfspot, calibration_videos=calibration_videos)
    dask_graph['image_tensor'] = dask_graph['result']
    dask_graph['normalized_tensor'] = (datasets.raw_wavefront_array_to_normalized_txy, 'image_tensor', ydim_out, duration_s, 
                                         time_axis_scale, sampling_hz) #SAMPLING_HZ
                                    # Trim trailing 2 dimensions so they evenly divide into 4 (for CNN model)
    dask_graph['result'] = (datasets.trim_img_to_nearest_multiple, 'normalized_tensor')
    return dask_graph

def wavedetection_cnn_training(training_video):
    from surfbreak.model_training import train_basic_waveform_model, train_wavefront_detection_cnn
    dask_graph = {
        'rough_waveform_model_ckpt': (train_basic_waveform_model, training_video, 20),
        'waveform_cnn_ckpt': (train_wavefront_detection_cnn, training_video, 'rough_waveform_model_ckpt', 20)
    }
    return dask_graph

def video_to_trimmed_array_yxt(video_filename, duration_s, start_s, surfspot=None, calibration_videos=None):
    """ Image tensors are 10hz by default (1/6th of the frames from a of 60Hz video)"""
    if surfspot is None and calibration_videos is None:
        filename_header = video_filename.split('/')[-1].split('_')[0]
        if filename_header in SURFSPOT_CALIBRATION_VIDEOS.keys():
            surfspot = filename_header
        else:
            raise ValueError('surfspot cannot be inferred from filename,'
                             'and both `surfspot` and `calibration_videos` are unspecified.\n'
                             'Please specify surfspot present in pipelines.SURFSPOT_CALIBRATION_VIDEOS.')
    if calibration_videos is None:
        calibration_videos = SURFSPOT_CALIBRATION_VIDEOS[surfspot]
    
    dask_graph = {
        'calibration_ranges': (transform.run_surfzone_detection, calibration_videos),
        'result': (transform.video_file_to_trimmed_image_xyt, video_filename, 'calibration_ranges', duration_s, start_s)
    }
    return dask_graph


def video_to_waveform_tensor_simple(video_filename, ydim_out, slice_xrange=None,
                             start_s=0, duration_s=30, time_axis_scale=0.5, 
                             output_dim=3, calibration_videos=None, surfspot=None):
    
    # Get a little more than the required duration for the raw video, then clip to the appropriate length
    # (pre-processing with delta-time result 2 less samples)
    """ Image tensors are 10hz by default (1/6th of the frames from a of 60Hz video)"""
    dask_graph = video_to_trimmed_array_yxt(video_filename, duration_s+1, start_s, 
                                                  surfspot=surfspot, calibration_videos=calibration_videos)
    dask_graph['trimmed_array_yxt'] = dask_graph['result']
    dask_graph.pop('result') # Remove this key
    dask_graph['clipped_image_array'] = (supervision.vertical_waveform_slice,'trimmed_array_yxt', slice_xrange, output_dim)
    dask_graph['trimmed_video_array'] = (datasets.raw_wavefront_array_to_normalized_txy, 'clipped_image_array', ydim_out, duration_s, 
                                         time_axis_scale, 10) #SAMPLING_HZ
    dask_graph['waveform_slice'] = (supervision.generate_waveform_slice, 'trimmed_array_yxt', slice_xrange, output_dim)
    dask_graph['trimmed_wavefront_array'] = (datasets.raw_wavefront_array_to_normalized_txy, 'waveform_slice', ydim_out, duration_s, 
                             time_axis_scale, 10, 1.0) #SAMPLING_HZ, clip_max (clipping wavefronts to max 1.0 very important)
    return dask_graph


if __name__ == "__main__":
    video = '../tmp/shirahama_1590387334_SURF-93cm.ts'

    vmf_graph = vid_to_fit_mean_flow_graph(video)
    pp.pprint(vmf_graph)
    with dask.config.set(num_workers=4):
        mean_flow, xrange, yrange = graphchain.get(vmf_graph, 'result', scheduler=dask.threaded.get)
    print(mean_flow.mean())


