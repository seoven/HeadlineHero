# src/label_map.py
from pathlib import Path
from loguru import logger
from src import cfg

# 全局缓存，避免重复读取文件
_LABEL_MAP = None
_LABEL_TO_ID = None


def _load_labels():
    """
    内部函数：从 labels.txt 读取并生成映射字典。
    使用单例模式缓存结果。
    """
    global _LABEL_MAP, _LABEL_TO_ID

    if _LABEL_MAP is not None:
        return _LABEL_MAP, _LABEL_TO_ID

    # 构建路径：data/raw_data/labels.txt
    labels_file = Path(cfg.path.raw_data) / "labels.txt"

    if not labels_file.exists():
        raise FileNotFoundError(f"❌ 关键文件缺失：{labels_file}\n请确保原始数据目录中包含 labels.txt")

    with open(labels_file, 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f.readlines() if line.strip()]

    if not labels:
        raise ValueError("labels.txt 内容为空")

    # 生成映射
    _LABEL_MAP = {idx: label for idx, label in enumerate(labels)}
    _LABEL_TO_ID = {label: idx for idx, label in enumerate(labels)}

    logger.info(f"✅ 标签映射已加载：{len(_LABEL_MAP)} 个类别 (来源：{labels_file.name})")
    return _LABEL_MAP, _LABEL_TO_ID


def get_label_mapping():
    """
    返回完整的 id2label 和 label2id 字典。
    供 Trainer 用于更新 Config。
    """
    id2label, label2id = _load_labels()
    return id2label, label2id


def get_label_name(label_id):
    """
    安全地获取标签名称。
    """
    id2label, _ = _load_labels()
    return id2label.get(label_id, f"未知类别_{label_id}")


def get_label_id(label_name):
    """
    安全地获取标签 ID。
    """
    _, label2id = _load_labels()
    return label2id.get(label_name, -1)


def get_num_labels():
    """动态获取类别数量"""
    id2label, _ = _load_labels()
    return len(id2label)


def get_all_labels():
    """返回所有标签名称列表 (按 ID 排序)"""
    id2label, _ = _load_labels()
    return [id2label[i] for i in range(len(id2label))]

# 初始化触发（可选，确保导入时即加载，或按需加载）
# _load_labels()