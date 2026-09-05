"""Modal Cloud Training Runner for MicroDuck Reinforcement Learning.

Usage:
    uvx modal run scripts/modal/train_modal.py --task-id Mjlab-LegLift-Flat-MicroDuck --max-iterations 15000 --gpu L4
"""

import os
import shutil
import subprocess
import modal

app = modal.App("microduck-rl")

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
    timeout=28800,  # 8 hours
    volumes={"/root/checkpoints": checkpoints_volume},
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train_task(
    task_id: str = "Mjlab-LegLift-Flat-MicroDuck",
    num_envs: int = 4096,
    max_iterations: int = 15000,
    save_interval: int = 200,
    run_name: str = "v1_phased_baseline",
    wandb_project: str = "ducky_leg_lift",
    resume_checkpoint: str = None,
):
    """Executes PPO training inside the Modal GPU container and syncs checkpoints."""
    print(f"=== Starting Modal RL Training on GPU for Task: {task_id} ===")
    print(f"Num Envs: {num_envs}, Max Iterations: {max_iterations}, Run: {run_name}, WandB: {wandb_project}")

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
    if resume_checkpoint:
        cmd.extend(["--agent.load-checkpoint", resume_checkpoint])

    print(f"Executing: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd,
        cwd="/root/microduck_rl",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    last_sync_time = 0
    import time
    for line in process.stdout:
        print(line, end="")
        now = time.time()
        if "Saving model" in line or "Saved model" in line or "Model saved" in line or (now - last_sync_time > 120):
            last_sync_time = now
            local_logs = "/root/microduck_rl/logs/rsl_rl"
            if os.path.exists(local_logs):
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
        raise RuntimeError(f"Training failed with exit code {process.returncode}")

    print("=== Training Complete and Volume Checkpoints Committed ===")


@app.local_entrypoint()
def main(
    task_id: str = "Mjlab-LegLift-Flat-MicroDuck",
    num_envs: int = 4096,
    max_iterations: int = 15000,
    save_interval: int = 200,
    run_name: str = "v1_phased_baseline",
    wandb_project: str = "ducky_leg_lift",
    gpu: str = "L4",
    resume_checkpoint: str = None,
):
    train_task.remote(
        task_id=task_id,
        num_envs=num_envs,
        max_iterations=max_iterations,
        save_interval=save_interval,
        run_name=run_name,
        wandb_project=wandb_project,
        resume_checkpoint=resume_checkpoint,
    )
