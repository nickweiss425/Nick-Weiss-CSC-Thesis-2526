import numpy as np
import pandas as pd

def sync_data_with_video(csv_path: str, trimmed_vid_len: float):
    # get dataframe of merged sensor data
    df = pd.read_csv(csv_path + "merged.csv")

    # start time should be when the event marker goes high
    start_time = df.loc[df['Event_Marker'] > 0, 'Time (s)'].iloc[0]

    # crop the dataframe to start at start_time and end based on total trimmed video length
    synced_df = df[(df["Time (s)"] >= start_time) & (df["Time (s)"] <= trimmed_vid_len + start_time)].copy()
    
    # reset the time column to start from 0 again
    synced_df["Time (s)"] = synced_df["Time (s)"] - start_time
    synced_df.reset_index(drop=True, inplace=True)
    synced_df.to_csv(csv_path + "synced.csv", index=False)


def drop_initial_setup_time(csv_path: str, start_time: float):
    df = pd.read_csv(csv_path + "synced.csv")

    # only include rows that are past the start time
    df = df[df['Time (s)'] > start_time].reset_index(drop=True)

    # re-zero the time column
    first_timestamp = df.iloc[0]['Time (s)']
    df['Time (s)'] = df['Time (s)'] - first_timestamp

    df.to_csv(csv_path + "trimmed_synced.csv", index=False)   

    