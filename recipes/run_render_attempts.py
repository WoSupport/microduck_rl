"""Execute honest rendering of all attempts on Modal and save outputs locally."""

import os
import sys
from modal_render_attempts import app, render_all_runs

def main():
    print("=== LAUNCHING HONEST RENDERING ON MODAL (NO RESETS, EXACT SAME LENGTH: 12.0s / 600 FRAMES) ===")
    out_dir = "/home/ubuntu/vibeduck/progress_animations/honest_12s"
    os.makedirs(out_dir, exist_ok=True)

    with app.run():
        results = render_all_runs.remote(video_length=600)
        print(f"\nReceived {len(results)} rendered videos from Modal!")
        for code, mp4_bytes in results.items():
            dest = os.path.join(out_dir, f"{code}_honest_12s.mp4")
            with open(dest, "wb") as f:
                f.write(mp4_bytes)
            print(f"Saved {code} -> {dest} ({len(mp4_bytes)/1e6:.2f} MB)")

    print("\n=== ALL HONEST ATTEMPT VIDEOS DOWNLOADED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()
