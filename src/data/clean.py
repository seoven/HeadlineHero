import re
import pandas as pd
import opencc
from loguru import logger
from collections import Counter
from pathlib import Path
from src import cfg

# ==========================================
# 全局配置与初始化
# ==========================================

# 初始化 OpenCC (繁体转简体)，放在全局避免重复初始化影响性能
cc_converter = opencc.OpenCC('t2s')

# 配置 Loguru 输出格式
logger.add(
    f"{cfg.path.log_dir}/data_cleaning.log",  # 日志文件路径
    rotation="10 MB",  # 日志轮转大小
    retention="7 days",  # 保留天数
    level="INFO"
)


# ==========================================
# 核心函数定义
# ==========================================

def clean_text(text):
    """
    清洗单条文本
    包含：去空、去前缀、去重、繁转简、去噪声、去符号、去HTML
    """
    if not isinstance(text, str):
        return ""

    text = text.strip()

    # 1. 空值检查
    if len(text) == 0:
        return ""

    # 2. 平台前缀类（如“新浪新闻：…”）
    prefix_pattern = r"^(新浪新闻|网易资讯|知乎热榜|B站热评|抖音头条)[：:]?"
    text = re.sub(prefix_pattern, "", text)

    # 3. 重复文本类：如“新闻内容 新闻内容” (局部重复)
    text = re.sub(r'(.+?)\s+\1+', r'\1', text)

    # 4. 繁体字转换 (使用全局初始化好的 converter)
    text = cc_converter.convert(text)

    # 5. 明显异常样本：如“噪声数据1”、“测试文本”等
    noise_pattern = r'^(噪声数据\d+|测试文本|示例数据|待补充|暂无内容|[\d\W]+)$'
    if re.match(noise_pattern, text):
        return ""

    # 6. 特殊符号清理
    text = re.sub(r'[★※◆→◎#￥%]+', "", text)

    # 7. HTML 标签清理
    text = re.sub(r'</?[a-zA-Z]+[^>]*>', "", text)

    # 8. 样本完全重复 (整句重复，如 "你好你好" -> "你好")
    text = re.sub(r'^(.+)\1+$', r'\1', text)

    # 9. 最终再次清洗可能产生的多余空格
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def analyze_files_distribution(file_paths):
    """
    统计每个文件的类别分布
    """
    logger.info("=" * 60)
    logger.info("【阶段 1: 数据分布统计】")
    logger.info("=" * 60)

    global_counter = Counter()
    error_count = 0

    for file_path in file_paths:
        try:
            # 只读取 label 列
            df_temp = pd.read_csv(file_path, usecols=['label'], encoding='utf-8')

            # 清洗 label 列的空格，防止 " 1" 和 "1" 被算作不同类
            df_temp['label'] = df_temp['label'].astype(str).str.strip()

            counts = df_temp['label'].value_counts().sort_index()
            global_counter.update(counts)

            # 格式化输出当前文件统计
            label_str = ", ".join([f"类{k}: {v}" for k, v in counts.items()])
            logger.info(f"文件: {file_path.name:<25} | {label_str}")

        except Exception as e:
            logger.warning(f"无法统计文件 {file_path.name}: {e}")
            error_count += 1

    logger.info("-" * 40)
    logger.info(f"全局总计: {sum(global_counter.values())} 条数据")

    return global_counter


def process_raw_data():
    """
    主流程：查找 -> 统计 -> 清洗 -> 保存
    """
    # 1. 路径初始化
    raw_dir = Path(cfg.path.raw_data)
    processed_dir = Path(cfg.path.processed_data)
    output_file = processed_dir / "processed_data.csv"

    # 确保输出目录存在
    processed_dir.mkdir(parents=True, exist_ok=True)

    # 2. 查找文件 (使用 pathlib.glob)
    pattern = "*.csv"
    files = list(raw_dir.glob(pattern))

    if not files:
        logger.error(f"在 {raw_dir.absolute()} 下未找到任何 CSV 文件 (模式: {pattern})")
        return

    logger.success(f"🔍 发现 {len(files)} 个数据文件。")

    # 3. 统计分布
    analyze_files_distribution(files)

    # 4. 加载、清洗并合并
    logger.info("【阶段 2: 加载与清洗】")
    all_dfs = []
    total_dropped = 0

    for file_path in files:
        try:
            df = pd.read_csv(file_path, encoding='utf-8')

            # 列名校正
            if 'title' not in df.columns:
                df.rename(columns={df.columns[0]: 'title'}, inplace=True)
            if 'label' not in df.columns:
                df.rename(columns={df.columns[1]: 'label'}, inplace=True)

            original_count = len(df)

            # --- 文本清洗 ---
            # 使用 apply 调用清洗函数
            df['title'] = df['title'].apply(clean_text)

            # 过滤空文本
            df = df[df['title'].str.len() > 0]

            # --- 标签清洗 (针对数字标签的鲁棒处理) ---
            # 转为 numeric，无法转换的变 NaN (例如 "未知", "1.5" 等)
            df['label'] = pd.to_numeric(df['label'], errors='coerce')

            # 记录清洗前的数量用于计算丢弃数
            count_after_text_drop = len(df)

            # 丢弃 label 为 NaN 的行
            df = df.dropna(subset=['label'])

            # 转为 int
            df['label'] = df['label'].astype(int)

            final_count = len(df)
            dropped_count = original_count - final_count
            total_dropped += dropped_count

            all_dfs.append(df)

            # 日志详情
            if dropped_count > 0:
                reason = []
                if count_after_text_drop != final_count:
                    reason.append(f"标签格式错误:{count_after_text_drop - final_count}")
                if original_count != count_after_text_drop:
                    reason.append(f"文本无效:{original_count - count_after_text_drop}")
                logger.info(
                    f"✅ {file_path.name}: 读取{original_count} -> 保留{final_count} (丢弃: {', '.join(reason)})")
            else:
                logger.info(f"✅ {file_path.name}: 读取{original_count} -> 保留{final_count}")

        except Exception as e:
            logger.error(f"❌ 处理文件 {file_path.name} 失败: {e}")
            continue

    if not all_dfs:
        logger.critical("没有成功处理任何数据，程序终止。")
        return

    # 5. 合并与保存
    logger.info("【阶段 3: 合并与保存】")
    final_df = pd.concat(all_dfs, ignore_index=True)

    # 最终统计
    logger.info(f"总有效数据量: {len(final_df)} 条")
    logger.info(f"总丢弃数据量: {total_dropped} 条")

    final_dist = final_df['label'].value_counts().sort_index()
    logger.info(f"最终类别分布:\n{final_dist.to_string()}")

    # 保存
    final_df.to_csv(output_file, index=False, encoding='utf-8')
    logger.success(f"🎉 清洗完成！文件已保存至: {output_file.absolute()}")


if __name__ == '__main__':
    # 可以在这里运行一个小测试，或者直接运行主流程
    # 测试用例
    test_text = "新浪新闻:這裡是一段繁體中文 中文<div>jiu是的</div> 噪声数据1"
    res = clean_text(test_text)
    logger.debug(f"测试清洗结果: '{res}'")

    # 执行主流程
    process_raw_data()