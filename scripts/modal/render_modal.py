"""Modal offscreen renderer for MicroDuck Duck Walk rollout animations."""

import os
import subprocess
from pathlib import Path
import modal

app = modal.App("microduck-render")
checkpoints_volume = modal.Volume.from_name("microduck-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "git",
        "curl",
        "ca-certificates",
        "build-essential",
        "libgl1",
        "libgl1-mesa-glx",
        "libosmesa6-dev",
        "libglew-dev",
        "ffmpeg",
    )
    .run_commands(
        "curl -LsSf https://astral.sh/uv/0.11.30/install.sh | sh",
        "echo 'export PATH=\"/root/.local/bin:$PATH\"' >> /root/.bashrc",
    )
    .add_local_dir(
        "/home/jimmy/ducky/microduck_rl",
        remote_path="/root/microduck_rl",
        copy=True,
        ignore=[".venv", ".git", "logs", "__pycache__", "*.pt", "*.onnx", "*.npy"],
    )
    .run_commands(
        "cd /root/microduck_rl && /root/.local/bin/uv sync --no-dev",
    )
)


@app.function(
    image=image,
    gpu="L4",
    timeout=600,
    volumes={"/root/checkpoints": checkpoints_volume},
)
def render_variant_rollout(
    ckpt_rel_path: str,
    output_prefix: str,
    cmd_vx: float = 0.25,
    camera_view: str = "perspective",
):
    output_dir = "/root/checkpoints/renders"
    os.makedirs(output_dir, exist_ok=True)
    keyframes_dir = os.path.join(output_dir, f"keyframes_{output_prefix}")
    os.makedirs(keyframes_dir, exist_ok=True)

    mp4_path = os.path.join(output_dir, f"{output_prefix}_{camera_view}.mp4")
    gif_path = os.path.join(output_dir, f"{output_prefix}_{camera_view}.gif")
    ckpt_path = os.path.join("/root/checkpoints", ckpt_rel_path)

    cmd = [
        "/root/.local/bin/uv", "run", "python", "scripts/render_velocity_rollout.py",
        "--checkpoint", ckpt_path,
        "--mp4", mp4_path,
        "--gif", gif_path,
        "--keyframes-dir", keyframes_dir,
        "--vx", str(cmd_vx),
        "--camera", camera_view,
    ]
    print(f"[{output_prefix}] Executing: {' '.join(cmd)}")
    subprocess.run(cmd, cwd="/root/microduck_rl", check=True)
    checkpoints_volume.commit()
    print(f"[{output_prefix}] Render complete: {mp4_path}")
    return mp4_path


@app.local_entrypoint()
def main():
    VARIANTS_TO_RENDER = [
        ("duck_walk/2026-09-04_22-15-54_v1_gentle_waddle/model_1499.pt", "v1_gentle_waddle", "perspective"),
        ("duck_walk/2026-09-04_22-15-54_v1_gentle_waddle/model_1499.pt", "v1_gentle_waddle", "front"),
        ("duck_walk/2026-09-04_22-15-43_v3_deep_crouch/model_1499.pt", "v3_deep_crouch", "perspective"),
        ("duck_walk/2026-09-04_22-15-43_v3_deep_crouch/model_1499.pt", "v3_deep_crouch", "front"),
    ]

    for ckpt_rel, prefix, view in VARIANTS_TO_RENDER:
        render_variant_rollout.remote(
            ckpt_rel_path=ckpt_rel,
            output_prefix=prefix,
            cmd_vx=0.25,
            camera_view=view,
        )
