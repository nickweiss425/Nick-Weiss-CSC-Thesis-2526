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

This value is estimated manually by visual inspection and may vary by participant.

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
