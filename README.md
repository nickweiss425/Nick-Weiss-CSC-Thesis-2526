# Automated Upper Extremity Primitive Classification Pipeline

This repository contains the full data processing and modeling pipeline used to
construct a labeled, synchronized, and feature-engineered dataset for upper
extremity functional primitive classification using wearable sensors and video.

---

## Table of Contents

- [Video Trimming and Labeling Pipeline](#video-trimming-and-labeling-pipeline)
- [Sensor Synchronization, Labeling, and Pre-Windowing Pipeline](#sensor-synchronization-labeling-and-pre-windowing-pipeline)
- [Windowing and Dataset Construction](#windowing-and-dataset-construction)
- [Model Training and Evaluation](#model-training-and-evaluation)

---

---


# Video Trimming and Labeling Pipeline

This pipeline produces a frame-accurate video time base for manual labeling and subsequent alignment with wearable sensor data. Raw recordings may exhibit variable frame rates (VFR), which can cause drift between video timestamps and frame indices. To avoid this issue, all labeling is performed on a derived constant-frame-rate (CFR) video that serves as the authoritative reference for frame-level annotation.

The procedure below defines the experiment time window, removes non-task setup time, and converts the video to a fixed 120 fps labeling master.

---

### 1. Identify Event Marker in Raw Video (Manual)

Open the raw video and identify the first frame at which the external event marker is visibly active. This marker defines the start of the experiment time window.

    ffplay -hide_banner -stats untrimmed.mp4

Controls:
- `p` — pause / play
- `s` — step forward one frame (while paused)
- Arrow keys — coarse seek

Record the following timestamps (via visual inspection):
- `T_start`: timestamp of the first frame where the event marker is ON
- `T_end`: timestamp corresponding to the end of the experiment window

These timestamps are determined manually and recorded per participant.

---

### 2. First Trim: Isolate Experiment Window (Lossless)

Trim the raw video to the experiment window defined by the event marker. This step is performed without re-encoding to preserve the original frames.

    ffmpeg -ss <T_start> -to <T_end> -i untrimmed.mp4 -c copy trimmed_stage1.mp4

Output:
- `trimmed_stage1.mp4` — full experiment window (lossless)

---

### 3. Measure Setup Buffer (Manual)

Open `trimmed_stage1.mp4` and determine the duration between the start of the video (event marker onset) and the moment the participant begins performing task-related activity.

Record:
- `setup_buffer_s`: setup buffer duration in seconds

This value is estimated manually by visual inspection and will vary by participant.

---

### 4. Second Trim: Remove Setup Buffer (Lossless)

Remove the setup buffer from the beginning of the experiment window. This produces a video that contains only task-relevant activity.

    ffmpeg -ss <setup_buffer_s> -i trimmed_stage1.mp4 -c copy trimmed_stage2.mp4

(Optional: include `-to <T2_end>` if trimming the end of the video is also required.)

Output:
- `trimmed_stage2.mp4` — activity-only window (lossless)

---

### 5. Create Constant Frame Rate (CFR) Labeling Master

Convert the activity-only video to an exact constant frame rate of 120 fps. This video serves as the authoritative time base for all frame-level labeling.

    ffmpeg -i trimmed_stage2.mp4 -vf "fps=120" -vsync cfr -pix_fmt yuv420p -an labeling_master_CFR120.mp4

Output:
- `labeling_master_CFR120.mp4` — CFR video used for labeling

---

### 6. Verify Frame Rate and Frame Count

Verify that the labeling master is truly constant frame rate and obtain the authoritative frame count.

    ffprobe -count_frames -select_streams v:0 \
      -show_entries stream=nb_read_frames,avg_frame_rate,duration \
      -of default=noprint_wrappers=1 labeling_master_CFR120.mp4

Verify that:
- `avg_frame_rate = 120/1`
- `nb_read_frames ≈ duration × 120` (allowing ±1 frame due to rounding)

---

### 7. Upload to Label Studio

Upload `labeling_master_CFR120.mp4` to Label Studio and configure:

# Sensor Synchronization, Labeling, and Pre-Windowing Pipeline

After video trimming and frame-accurate labeling, the following scripts construct a synchronized, labeled, and feature-engineered sensor dataset suitable for windowing and model training. These steps operate on participant-level folders within the `data/` directory.

The pipeline is modular: each step can be run independently for debugging and iteration, or executed sequentially using a wrapper script.

---

## Overview of Pre-Windowing Steps

For each participant, the pre-windowing pipeline consists of:

1. **Synchronize and label sensor data**
   - Merge raw Shimmer sensor files
   - Align sensors using timestamps
   - Synchronize sensor data to video via the event marker
   - Apply frame-level primitive labels from Label Studio

2. **Label quality control (QC)**
   - Detect missing primitive labels (NaNs)
   - Flag incomplete labeling for manual inspection

3. **Feature engineering**
   - Apply signal processing (filtering, envelopes, dynamic components)
   - Produce a cleaned, engineered time series for windowing

---

### Step 1: Synchronize and Label Sensor Data (Single Participant)

This step produces `labeled.csv` for a given participant. It is typically run per participant after labeling is completed in Label Studio.

**Script**
- `sync_and_label.py` (or equivalent step-1 entry point)

**Input**
- Raw Shimmer CSV files
- Trimmed video metadata
- `labels.json` from Label Studio

**Output**
- `labeled.csv` (continuous, time-aligned, labeled sensor data)

**Example**

    python sync_and_label.py --data_root "C:/Users/nicho/Desktop/Nick-Weiss-CSC-Thesis-2526/data" --participant P32 --sensors A5F2 A19E

---

### Step 2: Check for Missing Primitive Labels (QC)

This step scans `labeled.csv` files for missing (`NaN`) primitive labels. Missing values indicate incomplete or inconsistent labeling and should be inspected in Label Studio before proceeding.

This script supports single-participant and batch operation.

**Script**
- `check_missing_labels.py`

#### Single Participant
    python check_missing_labels.py --data_root "C:/Users/nicho/Desktop/Nick-Weiss-CSC-Thesis-2526/data" --participant P32

#### Batch (All Participants)
    python check_missing_labels.py --data_root "C:/Users/nicho/Desktop/Nick-Weiss-CSC-Thesis-2526/data"

**Output**
- Console warnings indicating:
  - which participants contain missing labels
  - how many samples are affected
  - example timestamps for inspection

No files are modified by this step.

---

### Step 3: Feature Engineering (Pre-Windowing)

This step applies signal processing to the labeled continuous data and produces an engineered representation suitable for windowing and model input.

Operations include:
- EMG bandpass filtering and envelope extraction
- Accelerometer and gyroscope denoising
- Dynamic (motion-only) component extraction
- Magnitude features

This script supports single-participant and batch operation.

**Script**
- `feature_engineer.py`

**Input**
- `labeled.csv`

**Output**
- `engineered.csv`

#### Single Participant
    python feature_engineer.py --data_root "C:/Users/nicho/Desktop/Nick-Weiss-CSC-Thesis-2526/data" --participant P32 --sensors A5F2 A19E

#### Batch (All Participants)
    python feature_engineer.py --data_root "C:/Users/nicho/Desktop/Nick-Weiss-CSC-Thesis-2526/data"

---

## Running the Full Pre-Windowing Pipeline (Wrapper)

For convenience, a wrapper script is provided to run all three pre-windowing steps sequentially for a single participant:

1. Synchronize and label sensor data  
2. Check for missing primitive labels  
3. Apply feature engineering  

This is the recommended entry point once labeling is complete.

**Script**
- `run_prewindow_pipeline.py`

**Example**

    python run_prewindow_pipeline.py --data_root "C:/Users/nicho/Desktop/Nick-Weiss-CSC-Thesis-2526/data" --participant P32 --sensors A5F2 A19E

The wrapper halts if any step fails, ensuring that feature engineering is only applied to valid, labeled data.

---

## Output Summary (Per Participant)

After successful execution, each participant directory contains:

- `labeled.csv` — synchronized, labeled continuous sensor data
- `engineered.csv` — feature-engineered time series (input to windowing)

These files serve as the input to the subsequent windowing and model training stages.


- `frameRate = 120`

Confirm that:
- The total frame count displayed in Label Studio matches the frame count reported by `ffprobe` (±1 frame).

Once this check passes, the video is considered safe for frame-accurate labeling and downstream alignment with sensor data.




# Windowing and Dataset Construction

This stage converts continuous, feature-engineered sensor data into fixed-length, labeled windows suitable for machine learning. The procedure also defines participant-level train/validation/test splits and applies normalization in a manner that avoids data leakage.

All operations are performed at the **participant level** and use only statistics derived from the training participants for normalization.

---

## Overview of Windowing and Dataset Construction

For each fold (defined by a held-out participant), the following steps are performed:

1. **Determine window parameters**
   - Estimate a common sampling frequency across participants
   - Convert window duration and stride from seconds to samples

2. **Define participant splits**
   - Leave-one-participant-out (LOPO) testing
   - One participant from the training pool used for validation

3. **Apply EMG amplitude normalization (p95)**
   - Compute per-participant EMG envelope scaling from training data only
   - Normalize continuous data prior to windowing

4. **Window the normalized time series**
   - Fixed-length, overlapping windows
   - One label assigned per window using the center sample

5. **Assemble datasets and apply global standardization**
   - Concatenate windows across participants
   - Fit a StandardScaler on training data only

---

## Step 1: Sampling Rate Estimation and Window Parameters

Each participant’s sampling rate is estimated from the `Time (s)` column in `engineered.csv` by computing the median inter-sample interval. A single global sampling rate (`fs_used`) is then defined as the **median across participants**.

Window parameters are specified in seconds and converted to samples:

- `win = round(win_s × fs_used)`
- `stride = round(stride_s × fs_used)`

Typical values:
- `win_s = 1.0` s
- `stride_s = 0.05` s

---

## Step 2: Participant-Level Dataset Split (LOPO)

Dataset construction follows a **leave-one-participant-out (LOPO)** strategy:

- **Test set**: one held-out participant (`--held_out`)
- **Training pool**: all remaining participants
- **Validation set**: one participant selected from the training pool
- **Training set**: remaining participants after removing validation

This ensures that all windows from a participant belong to exactly one split.

---

## Step 3: EMG Robust Normalization (p95 Scaling)

To reduce between-subject EMG amplitude variability while avoiding test leakage, EMG normalization is performed **before windowing** using participant-level statistics.

### Per-Participant p95 Computation
For each training participant:
- EMG envelope channels (`*_EMG*_ENV`) are extracted
- Rows with `Primitive == "Unknown"` are excluded
- The 95th percentile (p95) is computed per EMG channel

### Scaling Strategy
- Training and validation participants are normalized using their **own** p95 values
- The held-out test participant is normalized using the **median p95 across training participants**

This approach ensures that no information from the held-out participant influences normalization statistics.

---

## Step 4: Windowing Procedure

After EMG normalization, each participant’s time series is segmented into overlapping windows.

For each window:
- Length: `win` samples
- Step size: `stride` samples
- Feature tensor shape: `(win, num_features)`
- Label assignment: the **center sample’s** `Primitive` label

Windows whose center label is `"Unknown"` are discarded.

This results in:
- `X`: shape `(num_windows, win, num_features)`
- `y`: shape `(num_windows,)` (string labels)

Windowing is performed independently for each participant before concatenation.

---

## Step 5: Dataset Assembly and Global Standardization

Windowed data are concatenated across participants to form the final datasets:

- `X_train`, `y_train`
- `X_val`, `y_val`
- `X_test`, `y_test`

A `StandardScaler` is then:
- fit **only** on the training windows (flattened across time)
- applied to training, validation, and test sets

Primitive labels are mapped to integer class indices using a fixed, sorted class list shared across all splits.

---

## Outputs (Per Fold)

Each fold is saved to: runs/prep/fold_<HELD_OUT_PID>/


Contents include:
- `X_train.npy`, `y_train.npy`
- `X_val.npy`, `y_val.npy`
- `X_test.npy`, `y_test.npy`
- `scaler.joblib`
- EMG normalization artifacts (`p95_train_by_pid.npy`, `p95_median_train.npy`)
- `meta.json` containing split details, window parameters, feature lists, and normalization descriptions

---

## Example Command

    python prepare_fold.py --data_root "C:/Users/nicho/Desktop/Nick-Weiss-CSC-Thesis-2526/data" --out_root runs/prep --held_out P32 --win_s 1.0 --stride_s 0.05

This command generates a complete LOPO fold with participant `P32` held out for testing.
