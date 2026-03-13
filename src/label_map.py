# src/label_map.py

# 定义标签映射 (ID -> 中文名称)
LABEL_MAP = {
    0: "财经",
    1: "房产",
    2: "股票",
    3: "教育",
    4: "科技",
    5: "社会",
    6: "国际",
    7: "体育",
    8: "游戏",
    9: "娱乐"
}

# 反向映射 (中文名称 -> ID)，方便在数据预处理阶段使用
# 自动根据 LABEL_MAP 生成，防止手动写错
LABEL_TO_ID = {v: k for k, v in LABEL_MAP.items()}

def get_label_name(label_id):
    """
    安全地获取标签名称，防止 ID 越界。
    如果 ID 不存在，返回 'Unknown_ID' 格式。
    """
    return LABEL_MAP.get(label_id, f"未知类别_{label_id}")

def get_label_id(label_name):
    """
    安全地获取标签 ID。
    如果名称不存在，返回 -1。
    """
    return LABEL_TO_ID.get(label_name, -1)

def get_num_labels():
    """动态获取类别数量"""
    return len(LABEL_MAP)

def get_all_labels():
    """返回所有标签名称列表 (按 ID 排序)"""
    return [LABEL_MAP[i] for i in range(len(LABEL_MAP))]