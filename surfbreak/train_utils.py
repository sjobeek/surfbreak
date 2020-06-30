import torch
import matplotlib.pyplot as plt

def pretty_size(size):
	"""Pretty prints a torch.Size object"""
	assert(isinstance(size, torch.Size))
	return " × ".join(map(str, size))

def dump_tensors(gpu_only=True):
	"""Prints a list of the Tensors being tracked by the garbage collector."""
	import gc
	total_size = 0
	for obj in gc.get_objects():
		try:
			if torch.is_tensor(obj):
				if not gpu_only or obj.is_cuda:
					print("%s:%s%s %s" % (type(obj).__name__, 
										  " GPU" if obj.is_cuda else "",
										  " pinned" if obj.is_pinned else "",
										  pretty_size(obj.size())))
					total_size += obj.numel()
			elif hasattr(obj, "data") and torch.is_tensor(obj.data):
				if not gpu_only or obj.is_cuda:
					print("%s → %s:%s%s%s%s %s" % (type(obj).__name__, 
												   type(obj.data).__name__, 
												   " GPU" if obj.is_cuda else "",
												   " pinned" if obj.data.is_pinned else "",
												   " grad" if obj.requires_grad else "", 
												   " volatile" if obj.volatile else "",
												   pretty_size(obj.data.size())))
					total_size += obj.data.numel()
		except Exception as e:
			pass        
	print("Total size:", total_size)

def waveform_tensors_plot(waveform_out_txy, waveform_gt_txy, coords):
	assert waveform_out_txy.shape == waveform_gt_txy.shape
	_, xdim, _ = waveform_out_txy.shape
	tmin = coords[...,0].min() # Coordinate channels are (t,x,y)
	tmax = coords[...,0].max()
	vmin = waveform_out_txy.min()
	vmax = waveform_out_txy.max()
	fig, axes = plt.subplots(ncols=3,nrows=2, figsize=(10,6), sharey=True)
	axes[0][0].set_title(f"min t coord: {tmin:0.4f}")
	axes[0][1].set_title(f"value range: ({vmin:0.4f}, {vmax:0.4f})")
	axes[0][2].set_title(f"max t coord: {tmax:0.4f}")	
	axes[0][0].imshow(waveform_out_txy[0,:,:].T, vmin=vmin, vmax=vmax)
	axes[1][0].imshow(waveform_gt_txy[0,:,:].T, vmin=vmin, vmax=vmax)
	axes[0][1].imshow(waveform_out_txy[:,xdim//2,:].T, vmin=vmin, vmax=vmax)
	axes[1][1].imshow(waveform_gt_txy[:,xdim//2,:].T, vmin=vmin, vmax=vmax)
	for ax in (axes[0], axes[1]): # Plot vertical lines to show where x,y slices are in t,y slice
		ax[0].axvline(xdim//2, color='grey', ls='--')
		ax[2].axvline(xdim//2, color='grey', ls='--')
	axes[0][2].imshow(waveform_out_txy[-1,:,:].T, vmin=vmin, vmax=vmax)
	axes[1][2].imshow(waveform_gt_txy[-1,:,:].T, vmin=vmin, vmax=vmax)
	fig.tight_layout()
	return fig