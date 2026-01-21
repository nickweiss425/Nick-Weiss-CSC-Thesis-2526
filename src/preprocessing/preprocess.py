from merge_sensors import merge_and_clean_sensor_data
from trim import sync_data_with_video, drop_initial_setup_time
from label import label_data
from video import get_video_length
import os 
import cv2

def main():
    participant_id = "P32_V2"

    # define where data is held and total trimmed video length
    trial_path = os.path.join("C:/Users/nicho/Desktop/Nick-Weiss-CSC-Thesis-2526/data/", participant_id)

    trimmed_duration = get_video_length(trial_path, "trimmed_stage1.mp4")
    trimmed_finalized_duration = get_video_length(trial_path, "trimmed_stage2.mp4")

    # video should start when event marker goes high, should end when trial finishes
    print(f"Trimmed Video Length: {trimmed_duration}")
    print(f"Trimmed Finalized Length: {trimmed_finalized_duration}")

    # define which sensor IDs were used
    sensors_used = ["A5F2", "A19E"]

    # merge the two data files into one merged file
    merge_and_clean_sensor_data(trial_path, 
                                sensors_used,
                                trial_path)

    # use the event marker and video length to synchronize video and data file
    sync_data_with_video(trial_path, trimmed_duration)

    # after event marker goes high, there is a period of setup time --> trim this off of video and data file
    setup_buffer_s = trimmed_duration - trimmed_finalized_duration
    print(f"Setup Time: {setup_buffer_s}")
    drop_initial_setup_time(trial_path, setup_buffer_s)

    # use json of annotations to label data file
    label_data(trial_path, participant_id, fps=120.0)


if __name__ == "__main__":
    main()