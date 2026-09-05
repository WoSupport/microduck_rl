"""Batch render perspective and front rollouts for duck walk policies."""

from pathlib import Path
import subprocess
import sys

JOBS = [
    {
        "checkpoint": "logs/rsl_rl/duck_walk/duck_walk/2026-09-04_22-15-54_v1_gentle_waddle/model_1499.pt",
        "mp4": "renders/v1_front.mp4",
        "gif": "renders/v1_front.gif",
        "keyframes_dir": "renders/keyframes_v1_front",
        "camera": "front",
        "vx": 0.25,
        "duration": 4.0,
    },
    {
        "checkpoint": "logs/rsl_rl/duck_walk/duck_walk/2026-09-04_22-15-43_v3_deep_crouch/model_1499.pt",
        "mp4": "renders/v3_perspective.mp4",
        "gif": "renders/v3_perspective.gif",
        "keyframes_dir": "renders/keyframes_v3_perspective",
        "camera": "perspective",
        "vx": 0.25,
        "duration": 4.0,
    },
    {
        "checkpoint": "logs/rsl_rl/duck_walk/duck_walk/2026-09-04_22-15-43_v3_deep_crouch/model_1499.pt",
        "mp4": "renders/v3_front.mp4",
        "gif": "renders/v3_front.gif",
        "keyframes_dir": "renders/keyframes_v3_front",
        "camera": "front",
        "vx": 0.25,
        "duration": 4.0,
    },
]

def main():
    for i, job in enumerate(JOBS, 1):
        print(f"[{i}/{len(JOBS)}] Rendering {job['mp4']} ({job['camera']} view)...", flush=True)
        cmd = [
            "uv", "run", "python", "scripts/render_velocity_rollout.py",
            "--checkpoint", job["checkpoint"],
            "--mp4", job["mp4"],
            "--gif", job["gif"],
            "--keyframes-dir", job["keyframes_dir"],
            "--camera", job["camera"],
            "--vx", str(job["vx"]),
            "--duration", str(job["duration"]),
        ]
        res = subprocess.run(cmd)
        if res.returncode != 0:
            print(f"Failed rendering {job['mp4']}", file=sys.stderr)
            sys.exit(res.returncode)
    print("All batch renders completed successfully!", flush=True)

if __name__ == "__main__":
    main()
