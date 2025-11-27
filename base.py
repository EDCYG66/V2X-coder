# -*- coding: utf-8 -*-
from __future__ import annotations
import os


class BaseModel:
    """
    轻量占位的 BaseModel，兼容当前工程的使用方式（Agent 仅继承，不依赖其中方法）。
    去除 TF1 的 Saver/Session 等旧接口，避免在 TF2 环境中报错与 retracing。
    """

    def __init__(self, config=None):
        self.config = config

    def save_model(self, step=None):
        print("BaseModel.save_model called (no-op).")

    def load_model(self):
        print("BaseModel.load_model called (no-op).")
        return False

    @property
    def checkpoint_dir(self) -> str:
        return os.path.join('checkpoints', self.model_dir)

    @property
    def model_dir(self) -> str:
        try:
            env_name = getattr(self.config, 'env_name', 'default')
        except Exception:
            env_name = 'default'
        return f"{env_name}/"