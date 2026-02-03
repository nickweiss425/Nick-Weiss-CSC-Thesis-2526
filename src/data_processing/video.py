import os
import subprocess

def get_video_length(trial_path: str, video_name: str) -> float:
    """
    Return video duration in seconds using ffprobe (authoritative).
    Assumes ffprobe is installed and on PATH.
    """
    video_path = os.path.join(trial_path, video_name)

    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]

    out = subprocess.check_output(cmd, universal_newlines=True)
    return float(out.strip())
