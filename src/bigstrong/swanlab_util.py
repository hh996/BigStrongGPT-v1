"""SwanLab 实验跟踪（与 legacy path_setup 行为一致）。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_SWANLAB_ENABLED = False


def load_repo_dotenv(repo_root: Path) -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_file = repo_root / ".env"
    if env_file.is_file():
        load_dotenv(env_file)


def init_swanlab(
    repo_root: Path,
    project: str,
    experiment_name: str,
    config: dict,
    workspace: str | None = None,
) -> None:
    global _SWANLAB_ENABLED
    load_repo_dotenv(repo_root)
    key = os.environ.get("SWANLAB_API_KEY")
    if not key:
        _SWANLAB_ENABLED = False
        print("[SwanLab] 未设置 SWANLAB_API_KEY，跳过实验跟踪")
        return
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
    import swanlab

    swanlab.login(api_key=key)
    kwargs = {"project": project, "experiment_name": experiment_name, "config": config}
    if workspace:
        kwargs["workspace"] = workspace
    swanlab.init(**kwargs)
    _SWANLAB_ENABLED = True


def log_swanlab(metrics: dict) -> None:
    if not _SWANLAB_ENABLED:
        return
    import swanlab

    swanlab.log(metrics)
