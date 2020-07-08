# AUTOGENERATED! DO NOT EDIT! File to edit: 00_load_videos.ipynb (unless otherwise specified).

__all__ = ['get_surfdata', 'SURF_SPOTS', 'save_forecast_and_stream', 'video_length_s', 'decode_frame_sequence']

# Cell
import requests
def get_surfdata(spotid):
    """Retrieves current and forecast surf conditions for the next 24 hours from Surfline.
       Wave data interval is 6 hours from 12 AM, wind and weather interval is 2 hours """
    payload = {"spotId":spotid, "days":"2", "intervalHours":"6"}
    wave_data = requests.get("https://services.surfline.com/kbyg/spots/forecasts/wave", params=payload).json()
    payload["intervalHours"] = "2"
    wind_data = requests.get("https://services.surfline.com/kbyg/spots/forecasts/wind", params=payload).json()
    tides_data = requests.get("https://services.surfline.com/kbyg/spots/forecasts/tides", params=payload).json()
    weather_data = requests.get("https://services.surfline.com/kbyg/spots/forecasts/weather", params=payload).json()

    surf_data = wave_data
    surf_data['associated']['spotId'] = spotid
    surf_data['data']['wind'] = wind_data['data']['wind']
    surf_data['associated']['tideLocation'] = tides_data['associated']['tideLocation']
    surf_data['data']['tides'] = tides_data['data']['tides']
    surf_data['data']['weather'] = weather_data['data']['weather']

    return surf_data

# Cell
SURF_SPOTS = {"shirahama": {"cam_url": "https://www.youtube.com/watch?v=xuP8xIbZvmo",
                            "spotid": "584204204e65fad6a77098c4"},
              "shinmaiko": {"cam_url": "https://www.youtube.com/watch?v=3dkBkAjNay4",  # A link to map location is on this page
                            "spotid": "584204204e65fad6a77097f3"} # This is actually for Isonoura, but closest available
                                                                  # Actual waves will be much smaller than reported here...
             }

# Cell
import time
import os
import subprocess
from datetime import date
import json

def save_forecast_and_stream(surf_spot="shirahama", folder="data", duration_s=120, capture_output=False):

    # Get the surf data and save it to a descriptively named file
    surf_data = get_surfdata(SURF_SPOTS[surf_spot]["spotid"])

    # Extract values used in the filename
    # Wave data interval is 6 hours from 12 AM, wind and weather interval is 3 hours
    max_surf_6am_cm = int(surf_data['data']['wave'][1]['surf']['max']*100)

    unix_time = int(time.time())
    descriptive_name = f"{surf_spot}_{unix_time}_SURF-{max_surf_6am_cm}cm"
    wave_data_filepath = os.path.join(folder, descriptive_name + '.json')

    with open(wave_data_filepath, 'w') as file:
        json.dump(surf_data, file, indent=4)

    # Then download the surf cam footage and save in a descriptively named file
    video_filepath = os.path.join(folder, descriptive_name + '.ts')
    video_url  = SURF_SPOTS[surf_spot]['cam_url']
    # Using the cmd_string with shell=True is insecure.
    # cmd_string = f'streamlink --force --hls-duration {duration_s} -o {video_filename} {video_url} 1080p,best'
    # String conversion to list done via `import shlex; shlex.split(cmd_string)`
    cmd_list = ['streamlink','--force','--hls-duration', str(int(duration_s)), '-o', video_filepath, video_url, '1080p,best']

    if capture_output:
        stdout=subprocess.PIPE # stdout=PIPE same as capture_output=True
    else:
        stdout = None

    proc = subprocess.run(args=cmd_list, timeout=duration_s+60, stdout=stdout)


    return {"video_filepath": video_filepath,
            "video_subprocess": proc,
            "surf_data": surf_data}



# Cell
import cv2

def video_length_s(stream_filepath):
    cap = cv2.VideoCapture(stream_filepath)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count < 1:
        print("No frames in the video file")
        return None
    file_duration_s = frame_count/video_fps
    return file_duration_s

def decode_frame_sequence(stream_filepath, duration_s=10, start_s=0, RGB=False, one_image_per_n_frames=4):
    cap = cv2.VideoCapture(stream_filepath)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count < 1:
        print("No frames in the video file")
        return None
    file_duration_s = frame_count/video_fps
    wait_ms = int(1000/video_fps)
    print(f'Decoding {stream_filepath}  Duration: {file_duration_s/60:0.1f}m ({(file_duration_s):0.2f}s)  FPS: {video_fps}'
          f'  Emitting 1/{one_image_per_n_frames} of frames ')
    n_frames = int(duration_s*int(video_fps))
    start_frame_index = int(start_s*int(video_fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_index);
    frames = []

    for i in range(n_frames):
        # cache several frames
        ret, frame = cap.read()
        if i % one_image_per_n_frames != 0:
            continue
        if not ret:
            print("Reached end of video file, length may be shorter than expected.")
            break
        else:
            if RGB:
                frames.append(frame[:,:,::-1]) # BGR 2 RGB
            else:
                frames.append(frame) # BGR
    return frames
