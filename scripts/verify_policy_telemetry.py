#!/usr/bin/env python3
"""
Diagnostic tool for policy evaluation and visual inspection pre-checks.
Computes:
1. Video pixel-diff FFT & pelvis ROI motion frequency (Hz)
2. Peak-to-peak motion energy and standard deviation
3. Nyquist aliasing check against 1 fps sampling
4. Multi-tiered rating recommendation (Tier 0 to Tier 3)
"""

import sys
import os
import re
import subprocess
import numpy as np

def analyze_video(video_path: str):
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} does not exist.")
        sys.exit(1)

    # Locate ffmpeg
    ffmpeg_bin = "/home/ubuntu/.local/bin/ffmpeg"
    if not os.path.exists(ffmpeg_bin):
        ffmpeg_bin = "ffmpeg"

    # Get video info via ffmpeg
    p = subprocess.Popen([ffmpeg_bin, "-i", video_path], stderr=subprocess.PIPE, stdout=subprocess.DEVNULL)
    _, err = p.communicate()
    err_str = err.decode("utf-8", errors="ignore")
    
    dim_m = re.search(r"(\d{3,4})x(\d{3,4})", err_str)
    w, h = (int(dim_m.group(1)), int(dim_m.group(2))) if dim_m else (640, 480)
    
    fps_m = re.search(r"(\d+(?:\.\d+)?) fps", err_str)
    fps = float(fps_m.group(1)) if fps_m else 50.0

    # Extract raw grayscale frames
    cmd = [
        ffmpeg_bin, "-i", video_path,
        "-f", "rawvideo", "-pix_fmt", "gray", "-"
    ]
    pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    raw, _ = pipe.communicate()

    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, h, w)
    total_frames = len(frames)
    duration_s = total_frames / fps

    print("=" * 60)
    print(f"POLICY EVALUATION & MOTION TELEMETRY AUDIT")
    print(f"File: {os.path.basename(video_path)}")
    print(f"Resolution: {w}x{h} | Total Frames: {total_frames} | FPS: {fps:.1f} | Duration: {duration_s:.2f}s")
    print("=" * 60)

    # Frame-to-frame full differences
    diffs = np.array([np.mean(np.abs(frames[i].astype(float) - frames[i-1].astype(float))) for i in range(1, total_frames)])
    
    # Pelvis crop (center 40% box)
    h_start, h_end = int(h * 0.3), int(h * 0.7)
    w_start, w_end = int(w * 0.3), int(w * 0.7)
    pelvis_frames = frames[:, h_start:h_end, w_start:w_end]
    pelvis_diffs = np.array([np.mean(np.abs(pelvis_frames[i].astype(float) - pelvis_frames[i-1].astype(float))) for i in range(1, total_frames)])

    # FFT computation
    fft_pelvis = np.abs(np.fft.rfft(pelvis_diffs - np.mean(pelvis_diffs)))
    freqs = np.fft.rfftfreq(len(pelvis_diffs), 1.0 / fps)
    
    # In pixel diffs, physical oscillation frequency is half of the pixel diff frequency (swings back & forth)
    top_indices = np.argsort(fft_pelvis)[::-1][:3]
    top_freq = freqs[top_indices[0]]
    physical_freq = top_freq / 2.0
    top_amplitude = fft_pelvis[top_indices[0]]

    print("\n[Kinematic Motion Metrics]")
    print(f"- Pelvis Pixel Motion Energy (Mean ± Std): {np.mean(pelvis_diffs):.2f} ± {np.std(pelvis_diffs):.2f}")
    print(f"- Max Instantaneous Pixel Diff: {np.max(pelvis_diffs):.2f}")
    print(f"- Dominant Pixel Diff Frequency: {top_freq:.2f} Hz (Peak Power: {top_amplitude:.1f})")
    print(f"- Inferred Physical Gesture Frequency: {physical_freq:.2f} Hz")

    # Stroboscopic Nyquist Aliasing Audit
    print("\n[Stroboscopic Nyquist Aliasing Audit (1 fps Sampling)]")
    aliased_f = abs(physical_freq - round(physical_freq))
    print(f"- Physical Gesture Frequency: {physical_freq:.2f} Hz")
    print(f"- Beat / Aliased Frequency at 1 fps: {aliased_f:.2f} Hz")
    if aliased_f < 0.25:
        print("  ⚠️ HIGH RISK OF STROBOSCOPIC FREEZE ILLUSION! 1 fps sampling catches robot at nearly identical cycle phase.")
        print("  -> Multi-rate telemetry or intra-cycle burst frames REQUIRED.")
    else:
        print("  ✓ Apparent motion visible across discrete 1-second intervals.")

    # Multi-tiered classification recommendation
    print("\n[Automated Tier Recommendation]")
    if np.mean(pelvis_diffs) < 0.5:
        tier = "Tier 1: FAIL (TRUE STATIC FREEZE)"
        desc = "No cyclic motion detected (< 0.5 mean diff). Policy is stationary."
    elif 2.2 <= physical_freq <= 3.8:
        if np.mean(pelvis_diffs) >= 4.0:
            tier = "Tier 3: PASS (EXEMPLARY TARGET BEHAVIOR)"
            desc = f"Optimal target frequency ({physical_freq:.2f} Hz) with high amplitude motion energy ({np.mean(pelvis_diffs):.2f})."
        else:
            tier = "Tier 2: PASS (CONSERVATIVE FUNCTIONAL)"
            desc = f"Target frequency achieved ({physical_freq:.2f} Hz) with stable conservative amplitude ({np.mean(pelvis_diffs):.2f})."
    else:
        tier = "Tier 0 / Unclassified: REVIEW NEEDED"
        desc = f"Frequency {physical_freq:.2f} Hz outside nominal 2.5–3.5 Hz band."

    print(f"Classification: {tier}")
    print(f"Details: {desc}")
    print("=" * 60)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_policy_telemetry.py <path_to_video.mp4>")
        sys.exit(1)
    analyze_video(sys.argv[1])
