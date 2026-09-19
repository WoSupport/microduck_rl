#!/usr/bin/env python3
"""
Generate high-production YouTube Shorts video (1080x1920 vertical, 50 fps)
chronicling the progression of training Microduck to do a tail wag trick.
- Beat-matched scene cuts synchronized with Kevin MacLeod's 'Fluffing a Duck' (122 BPM).
- High-contrast visual cards, progress indicators, zoom crops, and side-by-side outro.
- Explicit focus on v8 ('almost good') and v9 ('final outcome').
"""

import os
import time
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import imageio.v3 as iio

OUTPUT_DIR = "/home/ubuntu/vibeduck"
VIDEO_HONEST_DIR = "/home/ubuntu/vibeduck/progress_animations/honest_12s"
AUDIO_FILE = "/home/ubuntu/vibeduck/shorts_audio_track.aac"
OUTPUT_VIDEO = "/home/ubuntu/vibeduck/youtube_shorts_microduck_tail_wag.mp4"
TEMP_VIDEO = "/home/ubuntu/vibeduck/shorts_visual_stream.mp4"

BOLD_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
REG_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

FPS = 50
BPM = 122.0
BEAT_DUR = 60.0 / BPM  # ~0.4918s per beat

# Pre-load fonts
f_badge = ImageFont.truetype(BOLD_FONT, 26)
f_title = ImageFont.truetype(BOLD_FONT, 48)
f_card_title = ImageFont.truetype(BOLD_FONT, 38)
f_card_desc = ImageFont.truetype(REG_FONT, 28)
f_meta = ImageFont.truetype(BOLD_FONT, 22)
f_split = ImageFont.truetype(BOLD_FONT, 24)

# Scenes definition aligned with 64 musical beats (16 bars @ 122 BPM)
SCENES = [
    {
        "id": 0,
        "name": "Intro Hook",
        "beats": 4,  # Beats 1-4 (Bar 1)
        "att_num": 0,
        "source": "v9",
        "start_frame": 100,
        "badge_text": "AI ROBOT TRAINING  •  9 ATTEMPTS ★",
        "badge_bg": (234, 179, 8),
        "headline": "★ GOAL: TEACH ROBOT DUCK TO TAIL WAG!",
        "desc": [
            "• Reinforcement learning (PPO) in MuJoCo sim",
            "• Target: Rapid 3 Hz lateral hip-roll wiggle",
            "• Must lock forward gaze & balance on 2 feet",
            "• Watch the AI struggle, fail, and triumph!",
            "• Wait until Attempt 8 and 9... it gets crazy!"
        ],
        "glow_color": (250, 204, 21),
        "is_split": False
    },
    {
        "id": 1,
        "name": "Attempt 1",
        "beats": 4,  # Beats 5-8 (Bar 2)
        "att_num": 1,
        "source": "v1",
        "start_frame": 360,  # Crashes at 427
        "badge_text": "ATTEMPT 1/9: FAILED ✕",
        "badge_bg": (239, 68, 68),
        "headline": "✕ ATTEMPT 1: BACKWARD SOMERSAULT!",
        "desc": [
            "• Initial exploration with additive reward stack",
            "• Discovered hip roll... but forgot pitch balance!",
            "• Flipped over backward onto its head at t=8.5s",
            "• Result: Total wipeout on the floor"
        ],
        "glow_color": (239, 68, 68),
        "is_split": False
    },
    {
        "id": 2,
        "name": "Attempt 2",
        "beats": 4,  # Beats 9-12 (Bar 3)
        "att_num": 2,
        "source": "v2",
        "start_frame": 100,  # Spins and falls
        "badge_text": "ATTEMPT 2/9: FAILED ✕",
        "badge_bg": (239, 68, 68),
        "headline": "✕ ATTEMPT 2: SPIN TO WIN... INTO FLOOR!",
        "desc": [
            "• Added velocity penalties to stop freeze",
            "• AI exploited unanchored yaw degrees of freedom",
            "• Began spinning like a helicopter until it crashed!",
            "• Classic reward hacking: high speed, zero balance"
        ],
        "glow_color": (239, 68, 68),
        "is_split": False
    },
    {
        "id": 3,
        "name": "Attempt 3",
        "beats": 4,  # Beats 13-16 (Bar 4)
        "att_num": 3,
        "source": "v3",
        "start_frame": 50,
        "badge_text": "ATTEMPT 3/9: TOO TIMID ✕",
        "badge_bg": (234, 179, 8),
        "headline": "▲ ATTEMPT 3: PARALYZED WITH FEAR!",
        "desc": [
            "• Penalized head orientation to stop spinning",
            "• Forward gaze firmly locked... but robot froze!",
            "• Scared to move hips without disturbing head",
            "• Standing upright, but timid 0.5 Hz tremble"
        ],
        "glow_color": (234, 179, 8),
        "is_split": False
    },
    {
        "id": 4,
        "name": "Attempt 4",
        "beats": 4,  # Beats 17-20 (Bar 5)
        "att_num": 4,
        "source": "v4",
        "start_frame": 80,
        "badge_text": "ATTEMPT 4/9: DRIFTING ✕",
        "badge_bg": (59, 130, 246),
        "headline": "▲ ATTEMPT 4: GREAT WAG, BUT MOONWALKING!",
        "desc": [
            "• Switched to Pythagorean multiplicative reward",
            "• Huge energy! Vigorous 3 Hz tail wag unlocked!",
            "• BUT no translation penalty was enforced",
            "• Robot happily danced sideways out of frame"
        ],
        "glow_color": (59, 130, 246),
        "is_split": False
    },
    {
        "id": 5,
        "name": "Attempt 5",
        "beats": 4,  # Beats 21-24 (Bar 6)
        "att_num": 5,
        "source": "v5",
        "start_frame": 50,
        "badge_text": "ATTEMPT 5/9: COLLAPSED ✕",
        "badge_bg": (239, 68, 68),
        "headline": "✕ ATTEMPT 5: KNEES SAID ABSOLUTELY NOT!",
        "desc": [
            "• Added heavy drift penalties and nominal constraints",
            "• Multiplied too many criteria directly together",
            "• Exploration gradient vanished: knees buckled",
            "• Slumped into a deep squat collapse at t=2.1s"
        ],
        "glow_color": (239, 68, 68),
        "is_split": False
    },
    {
        "id": 6,
        "name": "Attempt 6",
        "beats": 4,  # Beats 25-28 (Bar 7)
        "att_num": 6,
        "source": "v6",
        "start_frame": 75,
        "badge_text": "ATTEMPT 6/9: DAMPED ✕",
        "badge_bg": (234, 179, 8),
        "headline": "▲ ATTEMPT 6: STABLE, BUT DAMPED ROLL!",
        "desc": [
            "• Anchored sagittal plane to prevent buckling",
            "• Duck stays perfectly upright for the full rollout",
            "• But excessive roll damping squashed amplitude",
            "• Looks more like shivering than tail wagging"
        ],
        "glow_color": (234, 179, 8),
        "is_split": False
    },
    {
        "id": 7,
        "name": "Attempt 7",
        "beats": 4,  # Beats 29-32 (Bar 8)
        "att_num": 7,
        "source": "v7",
        "start_frame": 260,
        "badge_text": "ATTEMPT 7/9: TRIPPED ✕",
        "badge_bg": (239, 68, 68),
        "headline": "✕ ATTEMPT 7: TRIPPED ON ITS WEBBED FOOT!",
        "desc": [
            "• Introduced velocity push disturbances",
            "• High amplitude returned, but feet crossed over",
            "• Webbed foot caught the floor and tripped at 5.8s",
            "• Needs contact stability and drift suppression"
        ],
        "glow_color": (239, 68, 68),
        "is_split": False
    },
    {
        "id": 8,
        "name": "Attempt 8",
        "beats": 8,  # Beats 33-40 (Bars 9-10) -> ~3.93s
        "att_num": 8,
        "source": "v8",
        "start_frame": 60,
        "badge_text": "ATTEMPT 8/9: ALMOST GOOD! ★",
        "badge_bg": (16, 185, 129),
        "headline": "★ ATTEMPT 8: THE 3.2 Hz BREAKTHROUGH!",
        "desc": [
            "• Overcame compromise basin with joint tracking",
            "• Clean, rhythmic 3.2 Hz pelvic roll discovered!",
            "• Upright balance maintained with zero falls",
            "• Feet firmly planted on the ground",
            "• ALMOST THERE... just needs more amplitude!"
        ],
        "glow_color": (16, 185, 129),
        "is_split": False
    },
    {
        "id": 9,
        "name": "Attempt 9",
        "beats": 16,  # Beats 41-56 (Bars 11-14) -> ~7.87s (The Climax!)
        "att_num": 9,
        "source": "v9",
        "start_frame": 50,
        "badge_text": "ATTEMPT 9/9: FINAL OUTCOME ✓",
        "badge_bg": (34, 197, 94),
        "headline": "✓ ATTEMPT 9: THE ULTIMATE TAIL WAG!",
        "desc": [
            "✓ Multiplicative core + drift suppression = SUCCESS!",
            "✓ Vigorous 3.0 Hz high-amplitude tail wag / preen shake",
            "✓ Locked gaze forward (vestibulo-ocular reflex)",
            "✓ Both feet planted firmly with zero drift & zero falls",
            "✓ Fully verified, approved & ready for hardware robotd!"
        ],
        "glow_color": (34, 197, 94),
        "is_split": False
    },
    {
        "id": 10,
        "name": "Outro Comparison",
        "beats": 8,  # Beats 57-64 (Bars 15-16) -> ~3.93s (Cadence)
        "att_num": 9,
        "source": "split",  # Left: v1, Right: v9
        "start_frame": 360,
        "badge_text": "MISSION COMPLETE: RL SUCCESS ★",
        "badge_bg": (250, 204, 21),
        "headline": "★ FROM CRASHING TO TAIL WAGGING!",
        "desc": [
            "• Left: Attempt 1 (Day 1 face-plant somersault)",
            "• Right: Attempt 9 (Final approved tail wag perfection)",
            "• 9 iterative reward recipes on NVIDIA L4 GPU",
            "• 776 KB standalone ONNX policy ready for hardware!",
            "• Follow for more biped robot AI experiments!"
        ],
        "glow_color": (250, 204, 21),
        "is_split": True
    }
]


def pre_render_template(scene):
    """Pre-render static layout template for a scene to achieve 100+ fps compositing."""
    W, H = 1080, 1920
    canvas = Image.new("RGBA", (W, H), (10, 14, 26, 255))
    draw = ImageDraw.Draw(canvas)
    
    # Top Tag
    tag = "MICRODUCK BIPED  •  REINFORCEMENT LEARNING"
    tw = int(draw.textlength(tag, font=f_meta))
    draw.rounded_rectangle([(W//2 - tw//2 - 20, 75), (W//2 + tw//2 + 20, 118)], radius=18, fill=(30, 41, 59, 230), outline=(71, 85, 105), width=2)
    draw.text((W//2 - tw//2, 85), tag, font=f_meta, fill=(226, 232, 240))
    
    # Hook Title
    t1 = "TRAINING A ROBOT DUCK"
    t2 = "TO DO A TAIL WAG"
    tw1 = int(draw.textlength(t1, font=f_title))
    tw2 = int(draw.textlength(t2, font=f_title))
    draw.text((W//2 - tw1//2, 145), t1, font=f_title, fill=(255, 255, 255))
    draw.text((W//2 - tw2//2, 210), t2, font=f_title, fill=(250, 204, 21))
    
    # 9-Segment Progress Indicator
    prog_w = 900
    prog_x = (W - prog_w) // 2
    prog_y = 295
    seg_w = prog_w // 9
    att_num = scene["att_num"]
    glow_color = scene["glow_color"]
    
    for i in range(9):
        sx = prog_x + i * seg_w
        if att_num == 0:
            c = (234, 179, 8)  # Intro
        elif i + 1 < att_num:
            c = (34, 197, 94) if i+1 >= 8 else (239, 68, 68)
        elif i + 1 == att_num:
            c = glow_color
        else:
            c = (51, 65, 85)
        draw.rounded_rectangle([(sx + 5, prog_y), (sx + seg_w - 5, prog_y + 12)], radius=6, fill=c)
        
    # Main Video Card bounds (1000 x 750)
    card_w, card_h = 1000, 750
    card_x = (W - card_w) // 2
    card_y = 340
    
    # Border & Shadow placeholder
    draw.rounded_rectangle([(card_x - 3, card_y - 3), (card_x + card_w + 3, card_y + card_h + 3)], radius=18, outline=glow_color, width=6)
    
    # Badge placeholder
    badge_text = scene["badge_text"]
    badge_bg = scene["badge_bg"]
    bw = int(draw.textlength(badge_text, font=f_badge))
    draw.rounded_rectangle([(card_x + 24, card_y + 24), (card_x + bw + 56, card_y + 76)], radius=10, fill=(badge_bg[0], badge_bg[1], badge_bg[2], 240))
    draw.text((card_x + 38, card_y + 35), badge_text, font=f_badge, fill=(255, 255, 255))
    
    # Bottom Commentary Card
    bot_y = 1130
    bot_h = 580
    draw.rounded_rectangle([(card_x, bot_y), (card_x + card_w, bot_y + bot_h)], radius=24, fill=(15, 23, 42, 240), outline=(51, 65, 85), width=2)
    
    draw.text((card_x + 40, bot_y + 36), scene["headline"], font=f_card_title, fill=glow_color)
    
    for li, line in enumerate(scene["desc"]):
        draw.text((card_x + 40, bot_y + 105 + li * 52), line, font=f_card_desc, fill=(241, 245, 249))
        
    # Footer
    ft = "Microduck Biped Robot  •  Sim2Real Training Progression"
    ftw = int(draw.textlength(ft, font=f_meta))
    draw.text((W//2 - ftw//2, 1780), ft, font=f_meta, fill=(148, 163, 184))
    
    return canvas, (card_x, card_y, card_w, card_h)


def main():
    print("=== Generating YouTube Shorts: Microduck Tail Wag Progression ===")
    t_start = time.time()
    
    # 1. Load source raw video frames for all attempts
    print("\nLoading source video frames...")
    sources = {}
    for att_id in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
        f_path = os.path.join(VIDEO_HONEST_DIR, f"v{att_id}_honest_12s.mp4")
        sources[f"v{att_id}"] = list(iio.imiter(f_path))
        print(f"  Loaded v{att_id}: {len(sources[f'v{att_id}'])} frames")
        
    card_w, card_h = 1000, 750
    card_x = (1080 - card_w) // 2
    card_y = 340
    
    # 2. Build scene frames list
    print("\nCompositing 1080x1920 video frames synchronized to 122 BPM...")
    all_video_frames = []
    total_beats = sum(s["beats"] for s in SCENES)
    total_frames = int(total_beats * BEAT_DUR * FPS)
    print(f"Total Beats: {total_beats} (16 bars) | Total Target Frames: {total_frames} (~31.48s)")
    
    frame_counter = 0
    
    for s_idx, scene in enumerate(SCENES):
        scene_frames_count = int(scene["beats"] * BEAT_DUR * FPS)
        template, (cx, cy, cw, ch) = pre_render_template(scene)
        
        print(f"  Scene {s_idx} ({scene['name']}): {scene_frames_count} frames ({scene['beats']} beats, {scene_frames_count/FPS:.2f}s)")
        
        src_key = scene["source"]
        start_f = scene["start_frame"]
        
        for f in range(scene_frames_count):
            canvas = template.copy()
            
            if not scene["is_split"]:
                # Normal single attempt card
                raw_frame_idx = (start_f + f) % len(sources[src_key])
                raw_frame = sources[src_key][raw_frame_idx]
                
                # Crop center 1.15x for tight, punchy presentation
                img = Image.fromarray(raw_frame)
                w, h = img.size
                zoom = 1.15
                crop_w, crop_h = int(w / zoom), int(h / zoom)
                crop_cx, crop_cy = w // 2, int(h * 0.52)
                cropped_video = img.crop((crop_cx - crop_w//2, crop_cy - crop_h//2, crop_cx + crop_w//2, crop_cy + crop_h//2)).resize((cw, ch), Image.Resampling.BILINEAR)
                
                canvas.paste(cropped_video, (cx, cy))
            else:
                # Scene 10: Side-by-side Outro (Left: v1 fail, Right: v9 success)
                v1_idx = (start_f + f) % len(sources["v1"])
                v9_idx = (100 + f) % len(sources["v9"])
                
                img1 = Image.fromarray(sources["v1"][v1_idx])
                img9 = Image.fromarray(sources["v9"][v9_idx])
                
                # Each half is 500x750
                half_w = cw // 2
                crop1 = img1.resize((half_w, ch), Image.Resampling.BILINEAR)
                crop9 = img9.resize((half_w, ch), Image.Resampling.BILINEAR)
                
                canvas.paste(crop1, (cx, cy))
                canvas.paste(crop9, (cx + half_w, cy))
                
                # Divider & Labels
                draw = ImageDraw.Draw(canvas)
                draw.line([(cx + half_w, cy), (cx + half_w, cy + ch)], fill=(255, 255, 255), width=4)
                
                # Left Tag: ATTEMPT 1
                draw.rounded_rectangle([(cx + 12, cy + ch - 50), (cx + 220, cy + ch - 12)], radius=8, fill=(239, 68, 68, 220))
                draw.text((cx + 24, cy + ch - 44), "ATTEMPT 1 (FAIL)", font=f_split, fill=(255, 255, 255))
                
                # Right Tag: ATTEMPT 9
                draw.rounded_rectangle([(cx + half_w + 12, cy + ch - 50), (cx + half_w + 250, cy + ch - 12)], radius=8, fill=(34, 197, 94, 220))
                draw.text((cx + half_w + 24, cy + ch - 44), "ATTEMPT 9 (FINAL)", font=f_split, fill=(255, 255, 255))
                
            # Re-draw the badge on top of video card
            draw = ImageDraw.Draw(canvas)
            badge_text = scene["badge_text"]
            badge_bg = scene["badge_bg"]
            bw = int(draw.textlength(badge_text, font=f_badge))
            draw.rounded_rectangle([(cx + 20, cy + 20), (cx + bw + 52, cy + 72)], radius=10, fill=(badge_bg[0], badge_bg[1], badge_bg[2], 240))
            draw.text((cx + 34, cy + 31), badge_text, font=f_badge, fill=(255, 255, 255))
            
            # Subtle Beat Flash / Pulse on the exact downbeat (first 3 frames of each scene)
            if f < 3:
                flash = Image.new("RGBA", (1080, 1920), (255, 255, 255, int(60 * (1.0 - f/3.0))))
                canvas = Image.alpha_composite(canvas, flash)
                
            all_video_frames.append(np.array(canvas.convert("RGB")))
            frame_counter += 1
            
    print(f"\nRendered {len(all_video_frames)} total frames in {time.time()-t_start:.1f}s.")
    
    # 3. Write video stream
    print(f"\nEncoding visual stream to {TEMP_VIDEO}...")
    iio.imwrite(TEMP_VIDEO, all_video_frames, fps=FPS, codec="libx264")
    print(f"Visual stream encoded: {TEMP_VIDEO} ({os.path.getsize(TEMP_VIDEO)/1e6:.2f} MB)")
    
    # 4. Multiplex video with audio track using ffmpeg
    print(f"\nMuxing audio and video to final YouTube Shorts MP4...")
    mux_cmd = [
        "/home/ubuntu/.local/bin/ffmpeg", "-y",
        "-i", TEMP_VIDEO,
        "-i", AUDIO_FILE,
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        OUTPUT_VIDEO
    ]
    subprocess.run(mux_cmd, check=True)
    
    # Also copy to root web directory so it is directly downloadable
    web_dest = "/home/ubuntu/vibeduck/youtube_shorts_microduck_tail_wag.mp4"
    if os.path.exists(OUTPUT_VIDEO):
        final_size_mb = os.path.getsize(OUTPUT_VIDEO) / 1e6
        print(f"\n============================================================")
        print(f"🎉 YOUTUBE SHORTS VIDEO GENERATION COMPLETE!")
        print(f"Output File: {OUTPUT_VIDEO} ({final_size_mb:.2f} MB)")
        print(f"Resolution: 1080x1920 (9:16 Vertical Shorts)")
        print(f"Duration: {len(all_video_frames)/FPS:.2f} seconds ({len(all_video_frames)} frames @ 50 fps)")
        print(f"Music: Kevin MacLeod - 'Fluffing a Duck' (122 BPM, 16 bars)")
        print(f"Cuts: 10 Beat-matched scenes synchronized to the musical downbeat")
        print(f"============================================================")

if __name__ == "__main__":
    main()
