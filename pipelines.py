#! /usr/bin/python3
from surfbreak import detection, load_videos
import dask
import graphchain
import numpy
import pprint as pp

def vid_to_fit_mean_flow_graph(video_file, n_samples=10, duration_s=1, processes=4, draw_fit=False):
    """Detects the region of the image which contains breaking waves, using cached graph 
        Returns a 3-tuple (mean_flow_xy, xrange, yrange):"""

    start_times = detection.get_sample_start_times(video_file, n_samples=n_samples)
    print("Video subsample start times (seconds):", start_times)
               
    flow_tensor_nodes = {f'flow_tensor_{idx}': (detection.avg_wave_flows, video_file, start_s, duration_s)
                         for idx, start_s in enumerate(start_times)}    

    result = {'result': (detection.fit_mean_flows, list(flow_tensor_nodes.keys()))}

    dask_graph = {**flow_tensor_nodes, **result}
    return dask_graph

def graph_entry_to_string(graph_entry):
    return graph_entry[0].__module__ + "." + graph_entry[0].__name__ + "(" +\
           ", ".join(arg.__repr__() for arg in graph_entry[1:]) + ")"

def describe_graph(dask_graph):
    graph_description = {key: graph_entry_to_string(dask_graph[key])
                         for key in dask_graph.keys()}
    return graph_description

if __name__ == "__main__":
    video = './tmp/shirahama_1590387334_SURF-93cm.ts'

    vmf_graph = vid_to_fit_mean_flow_graph(video)
    pp.pprint(vmf_graph)
    with dask.config.set(num_workers=4):
        mean_flow, xrange, yrange = graphchain.get(vmf_graph, 'result', scheduler=dask.threaded.get)
    print(mean_flow.mean())


