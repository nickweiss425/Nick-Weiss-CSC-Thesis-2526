import numpy as np
import pandas as pd 
import json

def label_data(trial_folder: str, total_frames: int):
    json_path = trial_folder + "labels" + ".json"
    df = pd.read_csv(trial_folder + "trimmed_synced.csv")

    # load JSON labels
    with open(json_path, "r") as f:
        data = json.load(f)

    # collect start/end frames per annotation
    annotations_list = []
    for task in data:
        for annotation in task.get("annotations", []):
            for result in annotation.get("result", []):
                label = result["value"]["timelinelabels"][0]
                start = result["value"]["ranges"][0]["start"]
                end = result["value"]["ranges"][0]["end"]

                annotations_list.append({
                    "label": label,
                    "start_frame": start,
                    "end_frame": end
                })

    # get fps of video
    fps = get_fps(df, total_frames)

    for item in annotations_list:
        label = item['label']
        start_time = (item['start_frame'] - 1) / fps
        end_time = (item['end_frame'] - 1) / fps
        df.loc[(df["Time (s)"] >= start_time) & (df["Time (s)"] <= end_time), "Primitive"] = label

        mask = (df["Time (s)"] >= start_time) & (df["Time (s)"] <= end_time)
        indices = df.index[mask]

        if not indices.empty:
            df.loc[indices[0], "Primitive"] = "Start"
            df.loc[indices[-1], "Primitive"] = "End"
    
    df.to_csv(trial_folder + "labeled.csv", index=False)   



def get_fps(df, frames):
    start = df.iloc[0]['Time (s)']
    end = df.iloc[-1]['Time (s)']
    dt = end - start
    return frames / dt