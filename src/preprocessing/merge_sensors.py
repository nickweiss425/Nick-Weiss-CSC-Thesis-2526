import numpy as np
import pandas as pd

def merge_and_clean_sensor_data(data_path: str, sensor_names: list, output_path: str):
    sensor_dict = get_master_nonmaster_sensors(data_path, sensor_names)
    master_sensor = sensor_dict["master"]
    nonmaster_sensor = sensor_dict["nonmaster"]

    sensor_dfs = sensor_dict["dataframes"]
    df_master = sensor_dfs[master_sensor]
    df_nonmaster = sensor_dfs[nonmaster_sensor]
    merged = merge_sensor_data(df_master=df_master, 
                        t_master=f'Shimmer_{master_sensor}_TimestampSync_Unix_CAL',
                        master_name=master_sensor, 
                        df_other=df_nonmaster,
                        t_other=f'Shimmer_{nonmaster_sensor}_TimestampSync_Unix_CAL',
                        other_name=nonmaster_sensor,
                        tolerance=10,
                        keep_other_time=False
                        )

    # convert timestamp to relative time (seconds)
    if f'Shimmer_{master_sensor}_TimestampSync_Unix_CAL' in merged.columns:
        merged['Time (s)'] = (merged[f'Shimmer_{master_sensor}_TimestampSync_Unix_CAL'] - merged[f'Shimmer_{master_sensor}_TimestampSync_Unix_CAL'].iloc[0]) / 1000

    # drop unwanted columns
    drop_cols = []
    for sensor_id in sensor_names:
        drop_cols.append(f"Shimmer_{sensor_id}_Battery_CAL")
        drop_cols.append(f"Shimmer_{sensor_id}_ECG_EMG_Status1_CAL")
        drop_cols.append(f'Shimmer_{sensor_id}_TimestampSync_Unix_CAL')
    merged = merged.drop(columns=[c for c in drop_cols if c in merged.columns], errors='ignore')

    rename_map = {}
    for sensor_id in sensor_names:
        rename_map.update({
            f'Shimmer_{sensor_id}_Accel_LN_X_CAL': f'{sensor_id}_AccelX',
            f'Shimmer_{sensor_id}_Accel_LN_Y_CAL': f'{sensor_id}_AccelY',
            f'Shimmer_{sensor_id}_Accel_LN_Z_CAL': f'{sensor_id}_AccelZ',
            f'Shimmer_{sensor_id}_EMG_CH1_24BIT_CAL': f'{sensor_id}_EMG1',
            f'Shimmer_{sensor_id}_EMG_CH2_24BIT_CAL': f'{sensor_id}_EMG2',
            f'Shimmer_{sensor_id}_Gyro_X_CAL': f'{sensor_id}_GyroX',
            f'Shimmer_{sensor_id}_Gyro_Y_CAL': f'{sensor_id}_GyroY',
            f'Shimmer_{sensor_id}_Gyro_Z_CAL': f'{sensor_id}_GyroZ',
            f'Shimmer_{sensor_id}_Mag_X_CAL': f'{sensor_id}_MagX',
            f'Shimmer_{sensor_id}_Mag_Y_CAL': f'{sensor_id}_MagY',
            f'Shimmer_{sensor_id}_Mag_Z_CAL': f'{sensor_id}_MagZ',
        })
    merged = merged.rename(columns=rename_map)

    merged.to_csv(output_path + "merged.csv", index=False)



def merge_sensor_data(
    df_master: pd.DataFrame,
    t_master: str,
    master_name: str,
    df_other: pd.DataFrame,
    t_other: str,
    other_name: str,
    tolerance: float = 10,   # milliseconds
    keep_other_time: bool = False
    ) -> pd.DataFrame:
        """
    Align (snap) df_other to df_master's timeline using nearest-neighbor on timestamps,
    then return a single DataFrame on the master time grid.

    Parameters
    ----------
    df_master : DataFrame
        The 'master' sensor data. Its time column defines the output timeline.
    t_master : str
        Name of the master time column.
    df_other : DataFrame
        The other sensor's data to be aligned to the master timeline.
    t_other : str
        Name of the other sensor's time column
    tolerance : float, default 10ms
        Max allowed absolute time difference  for a match. Rows without a
        nearby sample within this window remain NaN for the other sensor's columns.
        For 50 Hz (20 ms period), ±10 ms is a good default.
    keep_other_time : bool, default False
        If True, keep df_other's time column (it will be renamed with the prefix).
        If False, drop it (you'll have a single, master time column).

    Returns
    -------
    DataFrame
        df_master columns (unchanged) + df_other columns (prefixed), aligned to the
        master timeline, with a single master time column unless keep_other_time=True.
    """
        # Safety: merge_asof requires sorted keys
        left = df_master.sort_values(t_master).reset_index(drop=True)
        left = left.rename(columns={f'Shimmer_{master_name}_Event_Marker_CAL': 'Event_Marker'})
        right = df_other.sort_values(t_other).reset_index(drop=True)
        right = right.drop(f'Shimmer_{other_name}_Event_Marker_CAL', axis=1)
        
        merged = pd.merge_asof(
            left,
            right,
            left_on=t_master,
            right_on=t_other,
            direction='nearest',
            tolerance=tolerance,
        )

        if not keep_other_time and t_other in merged.columns:
            merged = merged.drop(columns=[t_other])


        return merged



def get_master_nonmaster_sensors(data_path: str, sensor_names: list):
    # dictionary to hold sensor name and dataframe associated
    dfs = {}
    latest_starting_time = None
    earliest_ending_time = None

    for sensor in sensor_names:
        # read the raw data from each sensor
        df_raw = pd.read_csv(data_path + sensor + ".csv", sep='\t', skiprows=1)
        df_raw = df_raw.drop(columns=[col for col in df_raw.columns if "Unnamed" in col], errors='ignore')

        # first row contains units - remove it and reset index
        df = df_raw.iloc[1:].reset_index(drop=True)

        # convert all columns to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # add df to dictionary
        dfs[sensor] = df

        # get starting and ending time for sensor
        starting_time = df.iloc[0][f'Shimmer_{sensor}_TimestampSync_Unix_CAL']
        ending_time = df.iloc[-1][f'Shimmer_{sensor}_TimestampSync_Unix_CAL']
        
        # keep track of which sensor has the latest starting time and earliest ending time, update if necessary
        if latest_starting_time == None or starting_time > latest_starting_time:
            latest_starting_time = starting_time
            latest_start_sensor = sensor
        
        if earliest_ending_time == None or ending_time < earliest_ending_time:
            earliest_ending_time = ending_time
            earliest_end_sensor = sensor
    
    # we now want to define one sensor as the reference point (master)
    master_sensor = None

    # if one sensor has both the latest start and earliest end (tightest window), define it as master
    if earliest_end_sensor == latest_start_sensor:
        master_sensor = earliest_end_sensor
    # otherwise, the sensor with the latest starting time is the master sensor clipped to match the earliest ending time
    else:
        latest_start_df = dfs[latest_start_sensor]
        dfs[latest_start_sensor] = latest_start_df[latest_start_df[f'Shimmer_{latest_start_sensor}_TimestampSync_Unix_CAL'] <= earliest_ending_time]
        master_sensor = latest_start_sensor

    # define the other sensor to be the nonmaster
    nonmaster_sensor = None
    for sensor in sensor_names:
        if sensor != master_sensor:
            nonmaster_sensor = sensor
    
    return {
        "dataframes": dfs,
        "master": master_sensor,
        "nonmaster": nonmaster_sensor
    }