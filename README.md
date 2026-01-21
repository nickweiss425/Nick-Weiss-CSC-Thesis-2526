## Video Trimming and Labeling Pipeline

This pipeline ensures frame-accurate alignment between video labels and sensor data by enforcing a constant frame rate and consistent time base.

### 1. Identify Event Marker in Raw Video
Open the raw video and find the exact frame where the event marker first goes high.

    ffplay -hide_banner -stats untrimmed.mp4

Controls:
- p — pause / play
- s — step forward one frame (while paused)
- Arrow keys — coarse seek

Record:
- T_start: timestamp of the first frame where the event marker is ON
- T_end: timestamp of the end of the experiment window

---

### 2. First Trim: Isolate Experiment Window (Lossless)

Trim the raw video to the full experiment window without re-encoding.

    ffmpeg -ss <T_start> -to <T_end> -i untrimmed.mp4 -c copy trimmed_stage1.mp4

---

### 3. Measure Setup Buffer
Open trimmed_stage1.mp4 and determine the time between the start of the video (event marker) and when the participant actually begins performing activities.

- setup_buffer_s: setup buffer duration in seconds

---

### 4. Second Trim: Remove Setup Buffer (Lossless)
Remove the setup buffer from the beginning of the trimmed video.

    ffmpeg -ss <setup_buffer_s> -i trimmed_stage1.mp4 -c copy trimmed_stage2.mp4

(Optional: include -to <T2_end> if trimming the end as well.)

---

### 5. Create Constant Frame Rate (CFR) Labeling Master
Convert the final trimmed video to an exact constant frame rate (120 fps). This is the video used for labeling.

    ffmpeg -i trimmed_stage2.mp4 -vf "fps=120" -vsync cfr -pix_fmt yuv420p -an labeling_master_CFR120.mp4

---

### 6. Verify Frame Rate and Frame Count
Confirm that the labeling master is truly constant frame rate and obtain the authoritative frame count.

    ffprobe -count_frames -select_streams v:0 \
      -show_entries stream=nb_read_frames,avg_frame_rate,duration \
      -of default=noprint_wrappers=1 labeling_master_CFR120.mp4

Verify:
- avg_frame_rate = 120/1
- Frame count is consistent with duration × 120

---

### 7. Upload to Label Studio
Upload labeling_master_CFR120.mp4 to Label Studio and configure:

- frameRate = 120

Confirm that:
- The total frame count shown in Label Studio matches the frame count reported by ffprobe (±1 frame).

Once this check passes, the video is safe for frame-accurate labeling.
