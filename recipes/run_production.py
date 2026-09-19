"""Launch production training of Microduck Butt-Wiggle policy on Modal GPU."""

import os
import sys
from modal_train_butt_wiggle import app, train_production

def main():
    print("=== CONNECTING TO MODAL APP AND LAUNCHING PRODUCTION TRAINING ===")
    with app.run():
        result = train_production.remote(
            num_envs=4096,
            max_iterations=1200,
            save_interval=100,
            run_name="v9_butt_wiggle_preen_shake_perfect",
        )
        print("=== PRODUCTION TRAINING AND EXPORT COMPLETE ===")
        print("Run Name:", result.get("run_name"))
        print("Latest Checkpoint:", result.get("latest_checkpoint"))
        print("Modal Volume Dest:", result.get("volume_dest"))
        print("ONNX size:", result.get("onnx_size"), "bytes")
        print("Video size:", result.get("video_size"), "bytes")

        if result.get("video_bytes"):
            video_path = "/home/ubuntu/vibeduck/butt_wiggle_eval.mp4"
            with open(video_path, "wb") as f:
                f.write(result["video_bytes"])
            print(f"Successfully saved evaluation video to {video_path}")

        if result.get("onnx_bytes"):
            onnx_path = "/home/ubuntu/vibeduck/butt_wiggle_policy.onnx"
            with open(onnx_path, "wb") as f:
                f.write(result["onnx_bytes"])
            print(f"Successfully saved ONNX policy to {onnx_path}")

if __name__ == "__main__":
    main()
