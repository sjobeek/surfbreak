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


def infer_xyslice(model, coords_txyc, slice_t_idx=0):
	image_coords = coords_txyc[slice_t_idx].reshape(1, -1, 3)
	wf_values_out, _ = model(image_coords)
	image_vals_xy = wf_values_out.reshape(coords_txyc[slice_t_idx].shape[:-1])
	return image_vals_xy

def infer_tyslice(model, coords_txyc, slice_x_idx=None):
	if slice_x_idx is None: # Get the center slice if no idx specified
		slice_x_idx = coords_txyc.shape[1]//2
	image_coords = coords_txyc[:,slice_x_idx].reshape(1, -1, 3)
	wf_values_out, _ = model(image_coords)
	image_vals_ty = wf_values_out.reshape(coords_txyc[:,slice_x_idx].shape[:-1])
	return image_vals_ty

def plot_waveform_tensors(model, coords_txyc, wavefronts_txy):
	# Run the slice coordinates through the model to get the output samples
	first_image =          infer_xyslice(model, coords_txyc, slice_t_idx=0)
	left_tyslice_image =   infer_tyslice(model, coords_txyc, slice_x_idx=coords_txyc.shape[1]//4)
	center_tyslice_image = infer_tyslice(model, coords_txyc, slice_x_idx=None) # Center by default
	right_tyslice_image =  infer_tyslice(model, coords_txyc, slice_x_idx=(coords_txyc.shape[1]*3)//4)
	tyslice_fullimg = torch.cat((left_tyslice_image, center_tyslice_image, right_tyslice_image), dim=0)
	wavefront_tyslice = torch.cat((wavefronts_txy[:,coords_txyc.shape[1]//4], 
								   wavefronts_txy[:,(coords_txyc.shape[1]*2)//4], 
								   wavefronts_txy[:,(coords_txyc.shape[1]*3)//4]), dim=0)
	# And plot them
	fig, axes = plt.subplots(nrows=3, figsize=(10,10))
	axes[0].imshow(first_image.cpu().T)
	axes[0].set_title("first x,y slice")
	axes[0].axvline(coords_txyc.shape[1]//4, color='grey', ls='--')
	axes[0].axvline((coords_txyc.shape[1]*2)//4, color='grey', ls='--')
	axes[0].axvline((coords_txyc.shape[1]*3)//4, color='grey', ls='--')
	axes[1].imshow(tyslice_fullimg.cpu().T)
	axes[1].set_title("t,y values over time (left, center, right)")
	axes[2].imshow(wavefront_tyslice.cpu().T)
	axes[2].set_title("t,y wavefront training signal (left, center, right")
	return fig

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