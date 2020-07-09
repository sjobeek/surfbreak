#! /usr/bin/python3

import time
import os
import argparse
from datetime import datetime
from surfbreak.load_videos import save_forecast_and_stream, SURF_SPOTS

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--data_folder", type=str, default='data',
                        help="Folder where the videos will be saved")
    parser.add_argument("-s", "--surfspot", type=str, default='shirahama',
                        help="Surfspot to record. Usually shirahama or shinmaiko.")
    parser.add_argument("-d", "--video_duration", type=int, default=10,
                        help="Minutes of surf video to record")
    parser.add_argument("-b", "--minutes_between_recordings", type=int, default=50,
                        help="Delay between recordings")
    parser.add_argument("-hr", "--hours_to_record", type=float, default=12,
                        help="Total number of hours to record")

    args = parser.parse_args()
    
    # Create the directory for today's data
    dtnow = datetime.now() # current date and time
    date_string = dtnow.strftime("%Y-%m-%d")
    todays_folder = os.path.join(args.data_folder, date_string)
    print("Saving videos to " + todays_folder)
    if not os.path.exists(todays_folder):
            os.makedirs(todays_folder)

    start_time_s = time.time()
    while time.time() - start_time_s < args.hours_to_record*60*60:
        minutes_remaining = (args.hours_to_record * 60) - (time.time() - start_time_s)/60
        # Don't sleep on the first iteration
        if time.time() - start_time_s > 1:
            print(f"Sleeping for {args.minutes_between_recordings} minutes ({minutes_remaining:0.1f} minutes remaining total)")
            time.sleep(args.minutes_between_recordings)

        # Actually record the surf forecast and video 
        print(f"Starting recording for {args.surfspot}, duration {args.video_duration} minute(s). ")
        save_forecast_and_stream(args.surfspot, duration_s=args.video_duration*60, folder=todays_folder) # Record 20 minutes at a time by default

    print(f"All recordings complete. Recorded videos can be found in: " + todays_folder)