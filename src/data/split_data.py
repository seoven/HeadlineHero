import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from loguru import logger
from collections import Counter
from src import cfg

# ==========================================
# 日志配置
# ==========================================
logger.add(
    f"{cfg.path.log_dir}/data_splitting.log",
    rotation="10 MB",
    retention="7 days",
    level="INFO"
)


def split_dataset():
    """
    读取清洗后的数据，进行分层划分 (Train/Val/Test)，并保存。
    默认比例: Train 80%, Val 10%, Test 10%
    """

    # 1. 路径准备
    input_file = Path(cfg.path.processed_data) / "processed_data.csv"
    output_dir = Path(cfg.path.divided_data)

    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_file.exists():
        logger.error(f"❌ 未找到清洗后的数据文件：{input_file}")
        logger.info("💡 请先运行 clean.py 生成 processed_data.csv")
        return

    logger.info("=" * 50)
    logger.info("【开始数据划分】")
    logger.info(f"输入文件：{input_file}")
    logger.info(f"输出目录：{output_dir}")
    logger.info("=" * 50)

    # 2. 加载数据
    try:
        df = pd.read_csv(input_file, encoding='utf-8')
        logger.success(f"✅ 成功加载数据：{len(df)} 条")
    except Exception as e:
        logger.error(f"❌ 加载数据失败：{e}")
        return

    if len(df) == 0:
        logger.warning("⚠️ 数据为空，无法划分。")
        return

    # 检查 label 列
    if 'label' not in df.columns:
        logger.error("❌ 数据中缺少 'label' 列，无法进行分层划分。")
        return

    # 3. 分层划分逻辑
    # 第一步：分出测试集 (Test set) - 10%
    # stratify=df['label'] 保证各类别比例一致
    train_val_df, test_df = train_test_split(
        df,
        test_size=0.1,
        random_state=cfg.seed,
        stratify=df['label'],
        shuffle=True
    )

    # 第二步：从剩余数据中分出验证集 (Val set) - 约 11.1% (即总体的 10%)
    # 此时 train_val_df 占总体的 90%，我们要从中取出 1/9 作为验证集，剩下 8/9 作为训练集
    # 计算比例：10% / 90% ≈ 0.1111
    val_ratio = 0.1 / 0.9

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio,
        random_state=cfg.seed,
        stratify=train_val_df['label'],
        shuffle=True
    )

    logger.info(f"\n📊 划分结果概览:")
    logger.info(f"   训练集 (Train): {len(train_df)} 条 ({len(train_df) / len(df) * 100:.1f}%)")
    logger.info(f"   验证集 (Val):   {len(val_df)} 条 ({len(val_df) / len(df) * 100:.1f}%)")
    logger.info(f"   测试集 (Test):  {len(test_df)} 条 ({len(test_df) / len(df) * 100:.1f}%)")

    # 4. 打印各类别分布详情
    def print_distribution(name, data_df):
        logger.info(f"\n--- {name} 类别分布 ---")
        counts = data_df['label'].value_counts().sort_index()
        total = len(data_df)

        distribution_str = []
        for label, count in counts.items():
            pct = (count / total) * 100
            distribution_str.append(f"类{label}:{count}({pct:.1f}%)")

        logger.info(" | ".join(distribution_str))
        return counts

    dist_train = print_distribution("训练集", train_df)
    dist_val = print_distribution("验证集", val_df)
    dist_test = print_distribution("测试集", test_df)

    # 5. 保存文件
    output_files = {
        "train": output_dir / "train.csv",
        "val": output_dir / "val.csv",
        "test": output_dir / "test.csv"
    }

    try:
        train_df.to_csv(output_files["train"], index=False, encoding='utf-8')
        val_df.to_csv(output_files["val"], index=False, encoding='utf-8')
        test_df.to_csv(output_files["test"], index=False, encoding='utf-8')

        logger.success("\n" + "=" * 60)
        logger.success("🎉 数据划分完成！文件已保存:")
        for name, path in output_files.items():
            logger.success(f"   {name.upper():<6} -> {path}")
        logger.success("=" * 60)

    except Exception as e:
        logger.error(f"❌ 保存文件失败：{e}")


if __name__ == '__main__':
    split_dataset()