# src/__init__.py

import os
import sys

# 确保项目根目录在路径中
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 初始化配置 (这步很重要，确保 cfg 在任何地方导入时都已加载)
from .config import cfg, update_config

__all__ = [
    "cfg", "update_config"
]
