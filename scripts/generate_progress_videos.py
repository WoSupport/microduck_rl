#!/usr/bin/env python3
"""
Generate standardized honest progress animations for all Microduck Butt-Wiggle training attempts.
- Exactly 12.00 seconds (600 frames at 50 fps) for EVERY attempt.
- Unmasked failure trajectories: if a policy fails or falls over, it stays down.
- Clean raw versions in raw/ and professionally annotated versions with sleek overlays.
- 3x3 Synchronized Comparison Grid Video showing all 9 attempts side-by-side for 12.0s.
- Chronological Evolution Reel Video.
"""

import os
import glob
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio.v3 as iio

OUTPUT_DIR = "/home/ubuntu/vibeduck/progress_animations"
HONEST_DIR = os.path.join(OUTPUT_DIR, "honest_12s")
RAW_DIR = os.path.join(OUTPUT_DIR, "raw")
os.makedirs(RAW_DIR, exist_ok=True)

BOLD_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
REG_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

# Attempt metadata for honest 12.0s rollouts
ATTEMPTS = [
    {
        "id": 1,
        "code": "v1",
        "source_file": "v1_honest_12s.mp4",
        "output_name": "attempt_01_v1_baseline_exploration.mp4",
        "title": "Attempt 1 (v1): Baseline Exploration",
        "subtitle": "Additive Posture Reward | Uncontrolled Pitch, Falls at t=7.2s",
        "badge_text": "FALLS OVER",
        "badge_color": (239, 68, 68),  # Red
        "summary": "Initial baseline exploration with additive posture terms. Oscillates erratically and flips over backward at 7.2s.",
        "status": "failed_fall"
    },
    {
        "id": 2,
        "code": "v2",
        "source_file": "v2_honest_12s.mp4",
        "output_name": "attempt_02_v2_anti_freeze_quadrature.mp4",
        "title": "Attempt 2 (v2): Anti-Freeze Quadrature",
        "subtitle": "Quadrature Velocity Term | Unstable Yaw Spin, Falls at t=3.7s",
        "badge_text": "FALLS OVER",
        "badge_color": (239, 68, 68),  # Red
        "summary": "Quadrature velocity penalty prevents freeze, but unanchored yaw degrees of freedom cause severe spin and fall at 3.7s.",
        "status": "failed_fall"
    },
    {
        "id": 3,
        "code": "v3",
        "source_file": "v3_honest_12s.mp4",
        "output_name": "attempt_03_v3_fixed_quaternion_gaze.mp4",
        "title": "Attempt 3 (v3): Fixed Quaternion Gaze",
        "subtitle": "Forward Gaze Lock | Upright Stance, Low Amplitude Roll",
        "badge_text": "UNDER-OSCILLATING",
        "badge_color": (234, 179, 8),  # Amber
        "summary": "Forward quaternion head-lock stabilizes gaze and robot stays upright for all 12s, but roll oscillation amplitude is weak.",
        "status": "low_amplitude"
    },
    {
        "id": 4,
        "code": "v4",
        "source_file": "v4_honest_12s.mp4",
        "output_name": "attempt_04_v4_pythagorean_high_amplitude_drift.mp4",
        "title": "Attempt 4 (v4): High-Amplitude Pythagorean",
        "subtitle": "Vigorous 3 Hz Butt-Wiggle | Stays Upright, Drifts Over 12s",
        "badge_text": "VIGOROUS (DRIFTING)",
        "badge_color": (249, 115, 22),  # Orange
        "summary": "Multiplicative Pythagorean projection unlocks vigorous 3 Hz butt-wiggling; stays upright for 12s, but feet drift unanchored.",
        "status": "drifting"
    },
    {
        "id": 5,
        "code": "v5",
        "source_file": "v5_honest_12s.mp4",
        "output_name": "attempt_05_v5_tight_composite_freeze.mp4",
        "title": "Attempt 5 (v5): Tight Multiplicative Multipliers",
        "subtitle": "Over-Constrained Product | Collapses & Flops at t=2.1s",
        "badge_text": "COLLAPSED",
        "badge_color": (239, 68, 68),  # Red
        "summary": "Multiplying tight drift and sagittal terms directly into composite reward crushed exploration gradient, causing immediate collapse at 2.1s.",
        "status": "collapsed"
    },
    {
        "id": 6,
        "code": "v6",
        "source_file": "v6_honest_12s.mp4",
        "output_name": "attempt_06_v6_sagittal_posture_anchor.mp4",
        "title": "Attempt 6 (v6): Sagittal Posture Anchor",
        "subtitle": "Nominal Joint Regularizer | Upright Stance, Damped Butt Motion",
        "badge_text": "DAMPED MOTION",
        "badge_color": (234, 179, 8),  # Amber
        "summary": "Tested nominal joint posture constraints to anchor sagittal plane; duck remains upright for 12s, but butt motion is heavily damped.",
        "status": "damped"
    },
    {
        "id": 7,
        "code": "v7",
        "source_file": "v7_honest_12s.mp4",
        "output_name": "attempt_07_v7_push_shock_perturbation.mp4",
        "title": "Attempt 7 (v7): Push Shock Perturbation",
        "subtitle": "High Amplitude Restored | Falls at t=9.4s Under Push Shock",
        "badge_text": "PERTURBATION FALL",
        "badge_color": (239, 68, 68),  # Red
        "summary": "High amplitude restored, but inherited velocity push shocks (±0.4 m/s) knock the duck down at 9.4s where it stays down.",
        "status": "perturbation_fall"
    },
    {
        "id": 8,
        "code": "v8",
        "source_file": "v8_honest_12s.mp4",
        "output_name": "attempt_08_v8_additive_l1_compromise_basin.mp4",
        "title": "Attempt 8 (v8): Additive Tracking Policy",
        "subtitle": "Additive Joint Target Tracking | Stable 3.2 Hz Wiggle, Planted Stance",
        "badge_text": "CLEAN WIGGLE (3.2 Hz)",
        "badge_color": (16, 185, 129),  # Emerald / Teal
        "summary": "Policy successfully overcomes additive posture tracking to achieve a stable 3.2 Hz butt-wiggle with upright balance and planted feet across all 12.0s.",
        "status": "clean_wiggle"
    },
    {
        "id": 9,
        "code": "v9",
        "source_file": "v9_honest_12s.mp4",
        "output_name": "attempt_09_v9_approved_preen_shake_perfect.mp4",
        "title": "Attempt 9 (v9): Approved Preen Shake (SUCCESS)",
        "subtitle": "Multiplicative Core (5.0) + Additive Drift (-1.0) | Zero Falls",
        "badge_text": "APPROVED (THUMBS UP) ✅",
        "badge_color": (34, 197, 94),  # Green
        "summary": "Vigorous 3 Hz roll oscillation, locked forward gaze, planted feet, zero falls/resets across all 12.0s (600 frames). Approved!",
        "status": "approved"
    }
]

TARGET_FRAMES = 600  # Exactly 12.0s @ 50fps
FPS = 50

def create_overlay(frame_np, attempt_info, frame_idx, total_frames):
    """Render sleek modern HUD overlay onto frame."""
    h, w, c = frame_np.shape
    img = Image.fromarray(frame_np).convert("RGBA")
    
    # Overlay layer
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    font_title = ImageFont.truetype(BOLD_FONT_PATH, 16)
    font_sub = ImageFont.truetype(REG_FONT_PATH, 12)
    font_badge = ImageFont.truetype(BOLD_FONT_PATH, 11)
    font_mono = ImageFont.truetype(BOLD_FONT_PATH, 11)
    
    # Header translucent bar
    header_h = 58
    draw.rectangle([(0, 0), (w, header_h)], fill=(15, 23, 42, 220))  # Dark slate with opacity
    badge_c = attempt_info["badge_color"]
    draw.line([(0, header_h), (w, header_h)], fill=(badge_c[0], badge_c[1], badge_c[2], 255), width=2)
    
    # Status Pill Badge
    badge_text = attempt_info["badge_text"]
    badge_w = int(draw.textlength(badge_text, font=font_badge)) + 16
    badge_h = 22
    bx, by = 12, 10
    draw.rounded_rectangle([(bx, by), (bx + badge_w, by + badge_h)], radius=4, fill=(badge_c[0], badge_c[1], badge_c[2], 220))
    draw.text((bx + 8, by + 4), badge_text, font=font_badge, fill=(255, 255, 255, 255))
    
    # Title & Subtitle
    tx = bx + badge_w + 12
    draw.text((tx, 8), attempt_info["title"], font=font_title, fill=(255, 255, 255, 255))
    draw.text((tx, 32), attempt_info["subtitle"], font=font_sub, fill=(203, 213, 225, 240))
    
    # Bottom HUD Bar
    footer_h = 28
    draw.rectangle([(0, h - footer_h), (w, h)], fill=(15, 23, 42, 200))
    
    # Left Footer Info
    time_sec = frame_idx / float(FPS)
    time_str = f"Time: {time_sec:4.2f}s / 12.00s  |  Frame: {frame_idx:03d}/600  |  50 Hz (No Auto-Reset)"
    draw.text((12, h - footer_h + 7), time_str, font=font_mono, fill=(148, 163, 184, 255))
    
    # Right Footer microduck tag
    tag_str = "Microduck RL: Butt-Wiggle Task"
    tw = int(draw.textlength(tag_str, font=font_mono))
    draw.text((w - tw - 12, h - footer_h + 7), tag_str, font=font_mono, fill=(148, 163, 184, 255))
    
    # Composite overlay
    composed = Image.alpha_composite(img, overlay).convert("RGB")
    return np.array(composed)


def process_individual_videos():
    print("=== Processing Individual Standardized Honest Videos (12.0s / 600 frames) ===")
    loaded_runs = {}
    
    for att in ATTEMPTS:
        src_path = os.path.join(HONEST_DIR, att["source_file"])
        print(f"\nLoading {att['code']} from {src_path}...")
        frames = list(iio.imiter(src_path))
        
        # Standardize to exactly TARGET_FRAMES (600 frames / 12.0s)
        if len(frames) >= TARGET_FRAMES:
            std_frames = frames[:TARGET_FRAMES]
        else:
            pad = [frames[-1]] * (TARGET_FRAMES - len(frames))
            std_frames = frames + pad
            
        loaded_runs[att["code"]] = std_frames
        
        # Save raw 12.0s version in raw/
        raw_out = os.path.join(RAW_DIR, f"{att['code']}_12s_raw.mp4")
        iio.imwrite(raw_out, std_frames, fps=FPS, codec="libx264")
        print(f"  -> Saved raw: {raw_out}")
        
        # Render overlay on each frame
        annotated_frames = []
        for idx, f in enumerate(std_frames):
            ann_f = create_overlay(f, att, idx, TARGET_FRAMES)
            annotated_frames.append(ann_f)
            
        annotated_out = os.path.join(OUTPUT_DIR, att["output_name"])
        iio.imwrite(annotated_out, annotated_frames, fps=FPS, codec="libx264")
        print(f"  -> Saved annotated: {annotated_out}")
        
    return loaded_runs


def generate_3x3_grid(loaded_runs):
    print("\n=== Generating 3x3 Synchronized Comparison Grid Video (12.0s / 600 frames) ===")
    grid_out = os.path.join(OUTPUT_DIR, "progress_attempts_3x3_grid.mp4")
    
    cell_w, cell_h = 640, 480
    header_h = 90
    total_w = cell_w * 3
    total_h = cell_h * 3 + header_h
    
    font_main_title = ImageFont.truetype(BOLD_FONT_PATH, 28)
    font_main_sub = ImageFont.truetype(REG_FONT_PATH, 16)
    font_cell_title = ImageFont.truetype(BOLD_FONT_PATH, 15)
    font_cell_sub = ImageFont.truetype(REG_FONT_PATH, 12)
    font_cell_badge = ImageFont.truetype(BOLD_FONT_PATH, 11)
    
    grid_frames = []
    t0 = time.time()
    
    for f_idx in range(TARGET_FRAMES):
        if f_idx % 60 == 0:
            print(f"  Grid rendering frame {f_idx}/{TARGET_FRAMES} ({time.time()-t0:.1f}s)...")
            
        canvas = Image.new("RGB", (total_w, total_h), color=(10, 15, 26))
        draw = ImageDraw.Draw(canvas)
        
        # Draw Header
        draw.rectangle([(0, 0), (total_w, header_h)], fill=(15, 23, 42))
        draw.line([(0, header_h - 1), (total_w, header_h - 1)], fill=(51, 65, 85), width=2)
        
        title_text = "Microduck RL Policy Evolution: Butt-Wiggle Progress Across 9 Attempts (Honest 12.0s, No Resets)"
        draw.text((24, 16), title_text, font=font_main_title, fill=(255, 255, 255))
        
        time_sec = f_idx / float(FPS)
        sub_text = f"Synchronized 12.00s Rollout  |  T={time_sec:4.2f}s (Frame {f_idx:03d}/600)  |  50 Hz Control Loop  |  Unmasked Failures"
        draw.text((24, 54), sub_text, font=font_main_sub, fill=(148, 163, 184))
        
        # Draw each cell
        for i, att in enumerate(ATTEMPTS):
            row = i // 3
            col = i % 3
            cx = col * cell_w
            cy = header_h + row * cell_h
            
            cell_frame = loaded_runs[att["code"]][f_idx]
            cell_img = Image.fromarray(cell_frame)
            canvas.paste(cell_img, (cx, cy))
            
            draw.rectangle([(cx, cy), (cx + cell_w - 1, cy + cell_h - 1)], outline=(30, 41, 59), width=1)
            
            bar_h = 36
            cell_box = Image.new("RGBA", (cell_w, bar_h), (15, 23, 42, 210))
            box_draw = ImageDraw.Draw(cell_box)
            
            badge_c = att["badge_color"]
            badge_t = att["badge_text"]
            bw = int(box_draw.textlength(badge_t, font=font_cell_badge)) + 12
            box_draw.rounded_rectangle([(8, 7), (8 + bw, 27)], radius=3, fill=(badge_c[0], badge_c[1], badge_c[2], 230))
            box_draw.text((14, 10), badge_t, font=font_cell_badge, fill=(255, 255, 255))
            
            lbl = f"Attempt {att['id']} ({att['code']})"
            box_draw.text((16 + bw, 9), lbl, font=font_cell_title, fill=(255, 255, 255))
            
            sub_lbl = att["subtitle"].split("|")[0].strip()
            sw = int(box_draw.textlength(sub_lbl, font=font_cell_sub))
            box_draw.text((cell_w - sw - 12, 11), sub_lbl, font=font_cell_sub, fill=(203, 213, 225))
            
            canvas.paste(cell_box, (cx, cy), cell_box)
            
            if att["id"] == 9:
                draw.rectangle([(cx, cy), (cx + cell_w - 1, cy + cell_h - 1)], outline=(34, 197, 94), width=3)
            elif att["id"] == 8:
                draw.rectangle([(cx, cy), (cx + cell_w - 1, cy + cell_h - 1)], outline=(16, 185, 129), width=2)
            elif "FALL" in att["badge_text"] or "COLLAPSE" in att["badge_text"]:
                draw.rectangle([(cx, cy), (cx + cell_w - 1, cy + cell_h - 1)], outline=(239, 68, 68), width=1)
                
        grid_frames.append(np.array(canvas))
        
    print(f"Writing 3x3 grid video to {grid_out}...")
    iio.imwrite(grid_out, grid_frames, fps=FPS, codec="libx264")
    print(f"3x3 grid video saved successfully: {grid_out} ({os.path.getsize(grid_out) / 1e6:.2f} MB)")


def generate_timelapse_reel(loaded_runs):
    print("\n=== Generating Chronological Evolution Reel Video ===")
    reel_out = os.path.join(OUTPUT_DIR, "progress_evolution_timelapse.mp4")
    
    w, h = 640, 480
    font_big = ImageFont.truetype(BOLD_FONT_PATH, 24)
    font_med = ImageFont.truetype(BOLD_FONT_PATH, 16)
    font_reg = ImageFont.truetype(REG_FONT_PATH, 13)
    font_badge = ImageFont.truetype(BOLD_FONT_PATH, 12)
    
    reel_frames = []
    
    for att in ATTEMPTS:
        print(f"  Adding Attempt {att['id']} to timelapse reel...")
        
        # Title Card (35 frames = 0.7s)
        card_img = Image.new("RGB", (w, h), color=(15, 23, 42))
        cdraw = ImageDraw.Draw(card_img)
        
        bc = att["badge_color"]
        bt = f"PHASE {att['id']} OF 9: {att['badge_text']}"
        bw = int(cdraw.textlength(bt, font=font_badge)) + 20
        cdraw.rounded_rectangle([(30, 80), (30 + bw, 110)], radius=4, fill=(bc[0], bc[1], bc[2]))
        cdraw.text((40, 87), bt, font=font_badge, fill=(255, 255, 255))
        
        cdraw.text((30, 130), att["title"], font=font_big, fill=(255, 255, 255))
        cdraw.text((30, 175), att["subtitle"], font=font_med, fill=(203, 213, 225))
        
        words = att["summary"].split()
        lines = []
        cur = []
        for word in words:
            if len(" ".join(cur + [word])) > 55:
                lines.append(" ".join(cur))
                cur = [word]
            else:
                cur.append(word)
        if cur:
            lines.append(" ".join(cur))
            
        for li, line in enumerate(lines):
            cdraw.text((30, 225 + li * 24), line, font=font_reg, fill=(148, 163, 184))
            
        card_np = np.array(card_img)
        for _ in range(35):
            reel_frames.append(card_np)
            
        # Play 180 frames (3.6s) of rollout with overlay
        for f_idx in range(60, 240):
            frame_np = loaded_runs[att["code"]][f_idx]
            ann_f = create_overlay(frame_np, att, f_idx, TARGET_FRAMES)
            reel_frames.append(ann_f)
            
    print(f"Writing evolution reel video to {reel_out}...")
    iio.imwrite(reel_out, reel_frames, fps=FPS, codec="libx264")
    print(f"Evolution reel saved successfully: {reel_out} ({os.path.getsize(reel_out) / 1e6:.2f} MB)")


if __name__ == "__main__":
    t_start = time.time()
    loaded_runs = process_individual_videos()
    generate_3x3_grid(loaded_runs)
    generate_timelapse_reel(loaded_runs)
    print(f"\nAll honest progress videos generated successfully in {time.time()-t_start:.1f}s!")
