from merge_sensors import merge_and_clean_sensor_data
from trim import sync_data_with_video, drop_initial_setup_time
from label import label_data
from feature_eng import apply_feature_eng
def main():
    # define where data is held and total trimmed video length
    trial_path = "C:/Users/nicho/Desktop/Nick-Weiss-CSC-Thesis-2526/"

    # video should start when event marker goes high, should end when trial finishes
    trimmed_video_length = 492.71

    # define which sensor IDs were used
    sensors_used = ["A5F2", "A19E"]

    # merge the two data files into one merged file
    merge_and_clean_sensor_data(trial_path, 
                                sensors_used,
                                trial_path)

    # use the event marker and video length to synchronize video and data file
    sync_data_with_video(trial_path, trimmed_video_length)

    # after event marker goes high, there is a period of setup time --> trim this off of video and data file
    setup_buffer_s = 41.40
    drop_initial_setup_time(trial_path, setup_buffer_s)

    # use json of annotations to label data file
    total_frames = 54149
    label_data(trial_path, total_frames)

    apply_feature_eng(trial_path, sensors_used)

if __name__ == "__main__":
    main()