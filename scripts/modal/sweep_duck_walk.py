"""Modal Cloud Parallel Hyperparameter Sweep for MicroDuck Duck Walk.

Runs multiple Duck Walk variations concurrently across NVIDIA L4 GPUs on Modal:
  - Variant 1: Gentle waddle (8° roll sway, mild crouch)
  - Variant 2: Classic waddle (11° roll sway, classic crouch)
  - Variant 3: Deep crouch waddle (11° roll sway, deep knee bend)
  - Variant 4: Pronounced waddle (14° roll sway, wide sway)

Usage:
    uvx modal run scripts/modal/sweep_duck_walk.py
"""

import os
import shutil
import subprocess
import time
import modal

app = modal.App("microduck-duck-walk-sweep")

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
    timeout=14400,  # 4 hours
    volumes={"/root/checkpoints": checkpoints_volume},
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train_variant(
    task_id: str,
    num_envs: int = 4096,
    max_iterations: int = 1500,
    save_interval: int = 200,
    run_name: str = "v1_gentle_waddle",
    wandb_project: str = "ducky_duck_walk",
):
    """Executes training for one Duck Walk variant and syncs checkpoints to Modal Volume."""
    print(f"=== Starting Modal Training for Variant: {run_name} (Task: {task_id}) ===")
    print(f"Num Envs: {num_envs}, Max Iterations: {max_iterations}, WandB Project: {wandb_project}")

    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["WANDB_PROJECT"] = wandb_project

    cmd = [
        "/root/.local/bin/uv",
        "run",
        "train",
        task_id,
        "--env.scene.num-envs", str(num_envs),
        "--agent.max-iterations", str(max_iterations),
        "--agent.save-interval", str(save_interval),
        "--agent.run-name", run_name,
        "--agent.wandb-project", wandb_project,
    ]

    print(f"[{run_name}] Executing: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd,
        cwd="/root/microduck_rl",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    last_sync_time = 0
    for line in process.stdout:
        print(f"[{run_name}] {line}", end="")
        now = time.time()
        if "Saving model" in line or "Saved model" in line or "Model saved" in line or (now - last_sync_time > 120):
            last_sync_time = now
            local_logs = "/root/microduck_rl/logs/rsl_rl"
            if os.path.exists(local_logs):
                checkpoints_volume.reload()
                for root, _, files in os.walk(local_logs):
                    for f in files:
                        if f.endswith(".pt") or f.endswith(".onnx"):
                            src = os.path.join(root, f)
                            rel = os.path.relpath(src, local_logs)
                            dst = os.path.join("/root/checkpoints", rel)
                            os.makedirs(os.path.dirname(dst), exist_ok=True)
                            shutil.copy2(src, dst)
                checkpoints_volume.commit()

    process.wait()

    # Final sync
    local_logs = "/root/microduck_rl/logs/rsl_rl"
    if os.path.exists(local_logs):
        checkpoints_volume.reload()
        for root, _, files in os.walk(local_logs):
            for f in files:
                if f.endswith(".pt") or f.endswith(".onnx"):
                    src = os.path.join(root, f)
                    rel = os.path.relpath(src, local_logs)
                    dst = os.path.join("/root/checkpoints", rel)
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(src, dst)
        checkpoints_volume.commit()

    if process.returncode != 0:
        raise RuntimeError(f"[{run_name}] Training failed with exit code {process.returncode}")

    print(f"=== [{run_name}] Training Complete ===")
    return {"run_name": run_name, "task_id": task_id, "status": "COMPLETED"}


@app.local_entrypoint()
def main(
    max_iterations: int = 1500,
    num_envs: int = 4096,
    save_interval: int = 200,
):
    SWEEP_VARIANTS = [
        ("Mjlab-DuckWalk-V1-Flat-MicroDuck", "v1_gentle_waddle"),
        ("Mjlab-DuckWalk-V2-Flat-MicroDuck", "v2_classic_waddle"),
        ("Mjlab-DuckWalk-V3-Flat-MicroDuck", "v3_deep_crouch"),
        ("Mjlab-DuckWalk-V4-Flat-MicroDuck", "v4_pronounced_waddle"),
    ]

    print(f"Launching parallel sweep across {len(SWEEP_VARIANTS)} Modal L4 GPU containers...")
    args = [
        (task_id, num_envs, max_iterations, save_interval, run_name, "ducky_duck_walk")
        for task_id, run_name in SWEEP_VARIANTS
    ]

    for result in train_variant.starmap(args):
        print(f"Result: {result}")
