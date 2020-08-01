# File containing tests: 03_perspective_transform.ipynb

# Cell
from surfbreak import detection, graphutils, load_videos
from surfbreak import pipelines
import dask
import graphchain
import cv2

# Cell
import numpy as np
import matplotlib.pyplot as plt

def normalized(a, axis=-1, order=2):
    """General function for normalizing the length of a vector to 1"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

def shift_img(img, xshift, yshift):
    M = np.array([[1.,0.,xshift],[0.,1.,yshift]], dtype='float32')
    out_img = cv2.warpAffine(img, M, dsize=img.T.shape, borderMode=cv2.BORDER_TRANSPARENT)
    return out_img

def trim_image(image, xrange, yrange):
    return image[yrange[0]:yrange[1], xrange[0]:xrange[1]]


# Cell
def fit_line(heatmap, plot=False):

    high_indicies = np.where(heatmap > heatmap.mean())

    high_idx_array = np.dstack(high_indicies)[0][:,::-1]

    # https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=fitline#fitline
    line_def = cv2.fitLine(high_idx_array, 1, 0, 1, 0.1)
    # Y is down, X is to right in image space
    vx, vy, x0, y0 = line_def

    line_half_length = heatmap.shape[1]/3
    x1 = x0 - vx*line_half_length
    y1 = y0 - vy*line_half_length
    x2 = x0 + vx*line_half_length
    y2 = y0 + vy*line_half_length
    end_points = ((x1,y1), (x2,y2))

    if plot:
        img = cv2.line(heatmap.copy(), (x1, y1), (x2, y2), .5, thickness=8)
        plt.imshow(img, )

    return line_def, end_points

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


# Cell
def flow_mag_at_point(flow_mag, point, windowsize=100, percentile=90):
    xrange = (int(point[0] - windowsize/2), int(point[0] + windowsize/2))
    yrange = (int(point[1] - windowsize/2), int(point[1] + windowsize/2))
    window_mag = trim_image(flow_mag, xrange, yrange)
    # Find the 90th percentile value of the magnitudes which are greater than the mean
    # (Goal is to find the highest "reasonable" value)
    flow_magnitude = np.percentile(window_mag[window_mag > window_mag.mean()],percentile)
    return flow_magnitude

def fit_surfzone_homography_points(magnitude_img, percentile=90):
    """Returns four points aligned to a line fitted through the surfzone,
       Currently scales to the avg optical flow magnitude of each end of the surfzone (needs improvement)"""
    line, end_points = fit_line(magnitude_img, plot=False)
    leftmag = flow_mag_at_point(magnitude_img, end_points[0], percentile=percentile)
    rightmag = flow_mag_at_point(magnitude_img, end_points[1], percentile=percentile)

    dx = line[0]; dy = line[1]

    # Now calculate the position of the four corner points pre-transform
    # Note. here dy is being used for X coordinates and vice/versa because perpendeicular = swapped dx/dy ratio
    left_top_pt     = (end_points[0][0] - 100*dy*leftmag,  end_points[0][1] + 100*dx*leftmag)
    left_bottom_pt  = (end_points[0][0] + 100*dy*leftmag,  end_points[0][1] - 100*dx*leftmag)
    right_top_pt    = (end_points[1][0] - 100*dy*rightmag, end_points[1][1] + 100*dx*rightmag)
    right_bottom_pt = (end_points[1][0] + 100*dy*rightmag, end_points[1][1] - 100*dx*rightmag)
    pre_warp_corners = np.stack([np.concatenate(tup) for tup in
                                   [left_top_pt, left_bottom_pt, right_top_pt, right_bottom_pt]], axis=0)

    # Calculate the positionb of these same corners post-transform
    # Since this is now aligned to X axis, dy is all 0 and dx is 1.  Also, rightmag is set equal to leftmag
    len_line = np.sqrt((end_points[1][0]-end_points[0][0])**2 + (end_points[1][1]-end_points[0][1])**2)
    post_left_top_pt     = (end_points[0][0],            end_points[0][1] + 100*leftmag)
    post_left_bottom_pt  = (end_points[0][0],            end_points[0][1] - 100*leftmag)
    post_right_top_pt    = (end_points[0][0] + len_line, end_points[0][1] + 100*leftmag)
    post_right_bottom_pt = (end_points[0][0] + len_line, end_points[0][1] - 100*leftmag)
    post_warp_corners = np.stack([np.concatenate(tup) for tup in
                                 [post_left_top_pt, post_left_bottom_pt, post_right_top_pt, post_right_bottom_pt]], axis=0)

    return pre_warp_corners, post_warp_corners

# Cell
def pix_to_crop_from_dir(magnitude_img, crop_from, mass_crop_pct=0.05, backoff_pct=0.2, min_backoff_pix=10, ):
    assert crop_from in ["top", 'bottom', 'left', 'right']
    max_to_crop = magnitude_img.sum() * mass_crop_pct
    cropped_sum = 0
    if crop_from in ["top", 'bottom']:
        max_px_idx = magnitude_img.shape[0]
    elif crop_from in ["left", 'right']:
        max_px_idx = magnitude_img.shape[1]

    for i in range(max_px_idx):
        if cropped_sum > max_to_crop:
                return max(0, i - int(backoff_pct * i) - min_backoff_pix)  # subtract margin to back off from edge of image
        else:
            if crop_from is "bottom":
                cropped_sum += magnitude_img[max_px_idx - i - 1].sum()
            elif crop_from is "top":
                cropped_sum += magnitude_img[i].sum()
            elif crop_from is "left":
                cropped_sum += magnitude_img[:,i].sum()
            elif crop_from is "right":
                cropped_sum += magnitude_img[:,max_px_idx - i - 1].sum()

def find_crop_range(magnitude_img):
    """Returns cropped coordinates ((xmin,xmax), (ymin,ymax)) """
    px_bottom = pix_to_crop_from_dir(magnitude_img, 'bottom', mass_crop_pct=0.05, backoff_pct=0.2)
    px_top =    pix_to_crop_from_dir(magnitude_img, 'top',    mass_crop_pct=0.05, backoff_pct=0.4)
    y_crop_mag = magnitude_img[px_top : -1 - px_bottom]
    px_left  = pix_to_crop_from_dir(y_crop_mag, 'left',  mass_crop_pct=0.00001, backoff_pct=0, min_backoff_pix=1)
    px_right = pix_to_crop_from_dir(y_crop_mag, 'right', mass_crop_pct=0.00001, backoff_pct=0, min_backoff_pix=1)
    # Range defined
    crop_range = ((px_left, magnitude_img.shape[1] - px_right),(px_top,  magnitude_img.shape[0] - px_bottom))
    return crop_range

# Cell
def image_flow_magnitude(mean_flow):
    return np.sqrt(mean_flow[:,:,0]*mean_flow[:,:,0] + mean_flow[:,:,1]*mean_flow[:,:,1])

def fit_homography_to_mean_flow(meanflow, xrange, yrange):
    mean_flow_mag = image_flow_magnitude(meanflow)
    trimmed_mag = detection.trim_image(mean_flow_mag, xrange, yrange)
    pre_warp_corners, post_warp_corners = fit_surfzone_homography_points(trimmed_mag)
    H, _ = cv2.findHomography(pre_warp_corners, post_warp_corners)
    return H

def warp_trimmed_image(trimmed_image, H):
    return cv2.warpPerspective(trimmed_image, H, (trimmed_image.shape[1], trimmed_image.shape[0]))


def find_surfspot_calibration_params(mean_flow, rx_range, ry_range):
    H = fit_homography_to_mean_flow(mean_flow, rx_range, ry_range)

    mean_flow_mag = image_flow_magnitude(mean_flow)
    trimmed_mag = detection.trim_image(mean_flow_mag, rx_range, ry_range)
    mag_warped = warp_trimmed_image(trimmed_mag, H)
    tx_range, ty_range = find_crop_range(mag_warped)

    surfspot_calibration_params = {
        'crop_xrange': rx_range,
        'crop_yrange': ry_range,
        'h_matrix': H,
        'warped_xrange':tx_range,
        'warped_yrange':ty_range
    }

    return surfspot_calibration_params


def normalize_image(image, crop_xrange, crop_yrange, h_matrix, warped_xrange, warped_yrange, greyscale=True):
    trimmed_img = detection.trim_image(image, crop_xrange, crop_yrange)
    warped_img = warp_trimmed_image(trimmed_img, h_matrix)
    clipped_warped_img = warped_img[warped_yrange[0]:warped_yrange[1], warped_xrange[0]:warped_xrange[1]]
    if greyscale:
        return cv2.cvtColor(clipped_warped_img, cv2.COLOR_RGB2GRAY)
    else:
        return clipped_warped_img


def plot_mean_flow_mag(meanflow_tensor, axis=None, title=None):
    mean_flow_mag = np.sqrt(meanflow_tensor[:,:,0]*meanflow_tensor[:,:,0] +
                            meanflow_tensor[:,:,1]*meanflow_tensor[:,:,1])
    if axis is not None:
        axis.imshow(mean_flow_mag)
        axis.set_title(title)
    else:
        plt.imshow(mean_flow_mag)
        plt.gca().set_title(title)

# Cell
def run_surfcam_calibration(calibration_videos, num_workers=3):
    """Runs the optical flow fitting pipeline on the calibration videos, then uses the results
       to calculate a set of calibration parameters used by `transform.normalize_image()`"""
    video_mean_flows = []
    for vid in calibration_videos:
        flow_fit_graph = pipelines.vid_to_fit_mean_flow_graph(vid, n_samples=10, duration_s=2)

        with dask.config.set(num_workers=num_workers):
            meanflow, xyrange, yrange = graphchain.get(flow_fit_graph, 'result', scheduler=dask.threaded.get)
        video_mean_flows.append(meanflow)

    # Ignore the fit ranges from the individual videos, and re-fit across videos
    acc_mean_flow, acc_xrange, acc_yrange = detection.fit_mean_flows(video_mean_flows, draw_fit=True)

    calibration_params = find_surfspot_calibration_params(acc_mean_flow, acc_xrange, acc_yrange)
    return calibration_params

def run_surfzone_detection(calibration_videos, num_workers=3):
    """Runs the optical flow fitting pipeline on the calibration videos, 
       then calculates and returns an accumulated xrange,yrange tuple for all videos"""
    video_mean_flows = []
    for vid in calibration_videos:
        flow_fit_graph = pipelines.vid_to_fit_mean_flow_graph(vid, n_samples=10, duration_s=2)

        with dask.config.set(num_workers=num_workers):
            meanflow, xyrange, yrange = graphchain.get(flow_fit_graph, 'result', scheduler=dask.threaded.get)
        video_mean_flows.append(meanflow)

    # Ignore the fit ranges from the individual videos, and re-fit across videos
    acc_mean_flow, acc_xrange, acc_yrange = detection.fit_mean_flows(video_mean_flows, draw_fit=True)

    return acc_xrange, acc_yrange

def video_file_to_calibrated_image_tensors(video_file, calibration_params, duration_s, start_s, one_image_per_n_frames=6):
    frames = load_videos.decode_frame_sequence(video_file, duration_s=duration_s, start_s=start_s, RGB=True,
                                                  one_image_per_n_frames=one_image_per_n_frames)
    img_tensor = np.stack([normalize_image(frame, **calibration_params, greyscale=True)
                           for frame in frames], axis=-1)
    return img_tensor

def video_file_to_trimmed_image_xyt(video_file, trim_ranges, duration_s, start_s, one_image_per_n_frames=6):
    frames = load_videos.decode_frame_sequence(video_file, duration_s=duration_s, start_s=start_s, RGB=True,
                                                  one_image_per_n_frames=one_image_per_n_frames)
    img_tensor = np.stack([detection.trim_image(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), trim_ranges[0], trim_ranges[1])
                           for frame in frames], axis=-1)
    return img_tensor                                     
