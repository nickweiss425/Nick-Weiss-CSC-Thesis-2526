## Sensor Synchronization, Labeling, and Pre-Windowing Pipeline

After video trimming and frame-accurate labeling, the following scripts construct a synchronized, labeled, and feature-engineered sensor dataset suitable for windowing and model training. These steps operate on participant-level folders within the `data/` directory.

The pipeline is modular: each step can be run independently for debugging and iteration, or executed sequentially using a wrapper script.

---

### Overview of Pre-Windowing Steps

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

## Step 1: Synchronize and Label Sensor Data (Single Participant)

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

## Step 2: Check for Missing Primitive Labels (QC)

This step scans `labeled.csv` files for missing (`NaN`) primitive labels. Missing values indicate incomplete or inconsistent labeling and should be inspected in Label Studio before proceeding.

This script supports single-participant and batch operation.

**Script**
- `check_missing_labels.py`

### Single Participant
    python check_missing_labels.py --data_root "C:/Users/nicho/Desktop/Nick-Weiss-CSC-Thesis-2526/data" --participant P32

### Batch (All Participants)
    python check_missing_labels.py --data_root "C:/Users/nicho/Desktop/Nick-Weiss-CSC-Thesis-2526/data"

**Output**
- Console warnings indicating:
  - which participants contain missing labels
  - how many samples are affected
  - example timestamps for inspection

No files are modified by this step.

---

## Step 3: Feature Engineering (Pre-Windowing)

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

### Single Participant
    python feature_engineer.py --data_root "C:/Users/nicho/Desktop/Nick-Weiss-CSC-Thesis-2526/data" --participant P32 --sensors A5F2 A19E

### Batch (All Participants)
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


## Video Trimming and Labeling Pipeline

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

- `frameRate = 120`

Confirm that:
- The total frame count displayed in Label Studio matches the frame count reported by `ffprobe` (±1 frame).

Once this check passes, the video is considered safe for frame-accurate labeling and downstream alignment with sensor data.
