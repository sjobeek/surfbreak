#! /usr/bin/python3

from surfbreak.load_videos import save_forecast_and_stream, SURF_SPOTS

print("Starting recording... ")
save_forecast_and_stream('shirahama', duration_s=20*60) # Record 20 minutes at a time by default

save_forecast_and_stream('shinmaiko', duration_s=5*60)  # Record just 5 minutes since there usually aren't waves here...