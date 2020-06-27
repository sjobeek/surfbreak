# Test file: 05_waveform_supervision.ipynb (unless otherwise specified).

# Cell
import numpy as np
from matplotlib import animation
from IPython.display import HTML
import matplotlib.pyplot as plt

def animate_tensor(image_tensor, colorbar=True):
    tensor_aspect = image_tensor.shape[0] / image_tensor.shape[1]
    if tensor_aspect < 1:
        figsize = (20, tensor_aspect*20)
    else:
        figsize = (5/tensor_aspect, 5 + 1)
    fig, ax = plt.subplots(figsize=figsize)
    img = ax.imshow(image_tensor[..., 0])

    norm = plt.Normalize()
    colors = plt.cm.jet(norm(image_tensor))
    if colorbar:
        plt.colorbar(img, norm=colors, fraction=0.046, pad=0.04)
    def animate(i):
        img.set_data(image_tensor[... ,i])

    plt.rcParams["animation.embed_limit"] = 100.0 #100 MB max video size
    ani = animation.FuncAnimation(fig, animate, interval=100, frames=image_tensor.shape[-1])
    plt.close()
    return HTML(ani.to_jshtml())

def show_image(img):
    plt.figure(figsize=(20,2))
    plt.imshow(img)
    plt.colorbar()

# Cell
import cv2

def wavefront_diff_tensor(image_tensor):
    norm_tensor = (image_tensor - image_tensor.mean()) / image_tensor.std()
    """Averages sets of 2 frames over the time dimension, then finds the differences between them"""
    frames_offset_3_right =  norm_tensor[...,3:]
    frames_offset_2_right =  norm_tensor[...,2:-1]
    frames_offset_1_right =  norm_tensor[...,1:-2]
    frames_offset_zero =     norm_tensor[...,0:-3]

    # Right is future frames, left is past frames
    right_avg = (frames_offset_3_right + frames_offset_2_right) / 2
    left_avg  = (frames_offset_1_right + frames_offset_zero) / 2

    # Positive values indicate a change from dark -> bright (e.g. the foam of the wave-front)
    time_diff = right_avg - left_avg
    # keep only positive (wave front), and clip this to max 1 standard deviation of image brightness from the original normalized greyscale
    return time_diff.clip(min=0, max=1)

def dilate_tensor(image_tensor, erode_size=2, dilation_size=5, clip_range=(0.1, 1)):
    """Dilates the image tensor with a 2x bias for the horizontal (to emphasize horizontally-aligned wavefronts)"""
    dilated_tensor = np.zeros_like(image_tensor)
    for idx in range(dilated_tensor.shape[-1]):
        # Erode uniformly
        if erode_size > 0:
            erode_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_size, erode_size))
            eroded_tensor = cv2.erode(image_tensor[...,idx], erode_element);
        else: eroded_tensor = image_tensor[...,idx]

        eroded_tensor = eroded_tensor.clip(min=clip_range[0], max=clip_range[1])

        # Favor horizontal direction in dilation
        dilate_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dilation_size, dilation_size))
        dilated_tensor[...,idx] = cv2.dilate(eroded_tensor, dilate_element)
    return dilated_tensor

def vertical_waveform_slice(wavefront_tensor, xrange=(30,50), output_dim=2):
    if output_dim == 2:
        return wavefront_tensor[:,xrange[0]:xrange[1]].mean(axis=1)
    elif output_dim == 3:
        return wavefront_tensor[:,xrange[0]:xrange[1]]

# Cell
def generate_waveform_slice(image_tensor, slice_xrange=(30,50), output_dim=2):
    dtensor = dilate_tensor(wavefront_diff_tensor(image_tensor)).astype('float32')
    return vertical_waveform_slice(dtensor, xrange=slice_xrange, output_dim=output_dim)
