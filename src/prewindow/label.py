import numpy as np
import pandas as pd 
import json
import os

def label_data(trial_folder: str, participant_id: str, fps: float=120.0):
    """
    Apply Label Studio timeline annotations to synchronized sensor data.

    Converts 1-indexed frame ranges from labels.json into time ranges using the
    provided FPS and assigns primitive labels to rows in trimmed_synced.csv
    based on the 'Time (s)' column. Unlabeled regions are left as NaN by design.

    Assumptions:
      - Label Studio frame indices are 1-indexed.
      - Only annotations whose video path contains `participant_id` are used.
      - Overlapping labeled ranges are not resolved automatically; a warning
        is printed if overlaps are detected.
    """
    
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


    # quick sanity check to detect any overlap in labels in json:
    annotations_list.sort(key=lambda x: x["start_frame"])
    for i in range(1, len(annotations_list)):
        prev = annotations_list[i - 1]
        cur = annotations_list[i]
        if cur["start_frame"] < prev["end_frame"]:
            print(
                f"WARNING: overlapping labels detected: "
                f"{prev['label']} [{prev['start_frame']},{prev['end_frame']}) and "
                f"{cur['label']} [{cur['start_frame']},{cur['end_frame']})"
            )

    
    df.to_csv(os.path.join(trial_folder, "labeled.csv"), index=False)   



