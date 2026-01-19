from pathlib import Path

import pytest

from scripts.evaluate_domain import load_model_and_adapter


def test_load_model_missing_checkpoint(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing"
    with pytest.raises(RuntimeError, match="Checkpoint directory not found"):
        load_model_and_adapter(
            base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
            checkpoint_dir=missing_path,
            load_in_4bit=False,
            device_map="cpu",
        )


def test_load_model_missing_adapter_files(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="No adapter model found"):
        load_model_and_adapter(
            base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
            checkpoint_dir=tmp_path,
            load_in_4bit=False,
            device_map="cpu",
        )
