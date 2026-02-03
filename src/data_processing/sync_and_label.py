from merge_sensors import merge_and_clean_sensor_data
from trim import sync_data_with_video, drop_initial_setup_time
from label import label_data
from video import get_video_length

import os
import argparse


def main(participant_id, data_root, sensors_used):
    # define where data is held
    trial_path = os.path.join(data_root, participant_id)

    trimmed_duration = get_video_length(trial_path, "trimmed_stage1.mp4")
    trimmed_finalized_duration = get_video_length(trial_path, "trimmed_stage2.mp4")

    # video should start when event marker goes high, should end when trial finishes
    print(f"Stage 1 Video Length: {trimmed_duration}")
    print(f"Stage 2 Video Length: {trimmed_finalized_duration}")

    # merge the two data files into one merged file
    merge_and_clean_sensor_data(
        trial_path,
        sensors_used,
        trial_path
    )

    # use the event marker and video length to synchronize video and data file
    sync_data_with_video(trial_path, trimmed_duration)

    # after event marker goes high, there is a period of setup time --> trim this off of video and data file
    setup_buffer_s = trimmed_duration - trimmed_finalized_duration
    print(f"Setup Time: {setup_buffer_s}")
    drop_initial_setup_time(trial_path, setup_buffer_s)

    # use json of annotations to label data file
    label_data(trial_path, participant_id, fps=120.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge, sync, trim, and label sensor data for one participant.")
    parser.add_argument(
        "--participant",
        required=True,
        help="Participant ID (e.g., P32)"
    )
    parser.add_argument(
        "--data_root",
        default="C:/Users/nicho/Desktop/Nick-Weiss-CSC-Thesis-2526/data/",
        help="Root directory containing participant folders"
    )
    parser.add_argument(
        "--sensors",
        nargs="+",
        default=["A5F2", "A19E"],
        help="Sensor IDs to merge (e.g., A5F2 A19E)"
    )

    args = parser.parse_args()
    main(args.participant, args.data_root, args.sensors)
