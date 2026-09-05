"""`train` entry point: mjlab's trainer, plus `--hf-jobs` remote submission.

This project's [project.scripts] `train` shadows mjlab's so the everyday
command grows one flag:

    uv run train Mjlab-Kick-Flat-MicroDuck --env.scene.num-envs 4096 \
        --agent.max_iterations 4000              # local, exactly as before
    uv run train Mjlab-Kick-Flat-MicroDuck --env.scene.num-envs 4096 \
        --agent.max_iterations 4000 --hf-jobs    # same run, on HF Jobs

Without --hf-jobs, argv is passed to mjlab.scripts.train untouched. With it,
the submission flags (--flavor, --namespace, --detach, ... see hf_jobs.py)
are consumed here and everything else is forwarded to `uv run train` inside
the job.
"""

from __future__ import annotations

import sys

# WandB SDK 0.28+ compatibility patch
try:
    import wandb
    _orig_settings = wandb.Settings
    def _safe_settings(**kwargs):
        kwargs.pop("start_method", None)
        return _orig_settings(**kwargs)
    wandb.Settings = _safe_settings
except Exception:
    pass


def main() -> int | None:
    argv = sys.argv[1:]
    if "--hf-jobs" in argv:
        from mjlab_microduck.hf_jobs import submit

        return submit([a for a in argv if a != "--hf-jobs"])

    from mjlab.scripts.train import main as mjlab_train_main

    return mjlab_train_main()


if __name__ == "__main__":
    sys.exit(main())
