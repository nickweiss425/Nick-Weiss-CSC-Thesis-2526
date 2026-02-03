import numpy as np
import pandas as pd 
import json
import os

def label_data(trial_folder: str, participant_id: str, fps: float=120.0):
    json_path = os.path.join(trial_folder, "labels.json")
    df = pd.read_csv(os.path.join(trial_folder, "trimmed_synced.csv"))

    # load JSON labels
    with open(json_path, "r") as f:
        data = json.load(f)

    # collect start/end frames per annotation
    annotations_list = []
    for task in data:
        
        video_path = task["data"]["video"]

        # only use annotations for this participant
        if participant_id not in video_path:
            continue
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


    for item in annotations_list:
        label = item['label']
        start_time = (item['start_frame'] - 1) / fps
        end_time = (item['end_frame'] - 1) / fps
        df.loc[(df["Time (s)"] >= start_time) & (df["Time (s)"] < end_time), "Primitive"] = label

        # mask = (df["Time (s)"] >= start_time) & (df["Time (s)"] < end_time)
        # indices = df.index[mask]

        # if not indices.empty:
        #     df.loc[indices[0], "Primitive"] = "Start"
        #     df.loc[indices[-1], "Primitive"] = "End"
    
    df.to_csv(os.path.join(trial_folder, "labeled.csv"), index=False)   



