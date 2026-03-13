import torch
from torch.utils.data import DataLoader
from pathlib import Path
from loguru import logger
from transformers import AutoTokenizer, AutoConfig, BertForSequenceClassification

from src import cfg
from src.data.dataset import TextClassificationDataset
from src.data.loader import get_collate_fn
from src.trainer import Trainer

# 配置日志
logger.add(f"{cfg.path.log_dir}/training.log", rotation="10 MB", retention="7 days", level="INFO")


def main():
    # ================= 1. 设备准备 =================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🖥️  使用设备：{device}")
    if device.type == 'cuda':
        logger.info(f"🚀 GPU: {torch.cuda.get_device_name(0)}")

    # 定义模型基础路径
    model_path = Path(cfg.path.pretrained_model) / cfg.model.model_name

    if not model_path.exists():
        raise FileNotFoundError(
            f"❌ 预训练模型目录未找到：{model_path}\n"
            f"请检查 cfg.path.pretrained_model 和 cfg.model.model_name 配置是否正确。"
        )

    # ================= 2. 初始化分词器 (内联逻辑) =================
    logger.info("🔧 初始化分词器...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    logger.success(f"✅ 分词器加载成功：{model_path.name}")

    # ================= 3. 构建数据集 (Dataset) =================
    logger.info("📂 构建数据集 (Dataset)...")

    # 训练集
    train_path = cfg.path.divided_data / "train.csv"
    train_dataset = TextClassificationDataset(
        csv_path=train_path,
        tokenizer=tokenizer,
        max_length=cfg.model.max_length
    )
    logger.info(f"✅ 训练集加载完成：{len(train_dataset)} 条")

    # 验证集
    val_path = cfg.path.divided_data / "val.csv"
    val_dataset = TextClassificationDataset(
        csv_path=val_path,
        tokenizer=tokenizer,
        max_length=cfg.model.max_length
    )
    logger.info(f"✅ 验证集加载完成：{len(val_dataset)} 条")

    # ================= 4. 构建数据加载器 (DataLoader) =================
    logger.info("🔄 构建数据加载器 (DataLoader)...")

    collate_fn = get_collate_fn()

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=False
    )

    logger.info(f"✅ DataLoader 准备就绪")

    # ================= 5. 构建模型 (内联逻辑，核心修改点) =================
    logger.info("🤖 构建模型架构...")

    # 5.1 加载配置并注入超参数
    config = AutoConfig.from_pretrained(
        str(model_path),
        num_labels=cfg.finetune.num_labels,
        hidden_dropout_prob=getattr(cfg.finetune, 'hidden_dropout_prob', 0.1),
        attention_probs_dropout_prob=getattr(cfg.finetune, 'attention_probs_dropout_prob', 0.1),
        output_attentions=False,
        output_hidden_states=False
    )
    logger.debug(f"模型配置：隐藏层={config.hidden_size}, 类别数={config.num_labels}")

    # 5.2 实例化官方模型
    # ignore_mismatched_sizes=True: 允许分类头维度不匹配 (预训练通常是2类，我们是N类)
    model = BertForSequenceClassification.from_pretrained(
        str(model_path),
        config=config,
        ignore_mismatched_sizes=True,
        local_files_only=True  # 强制本地加载，防止联网
    )

    model.to(device)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.success(f"✅ 模型加载成功 | 总参数: {total_params:,} | 可训练: {trainable_params:,}")

    # ================= 6. 初始化训练器并运行 =================
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )

    try:
        logger.info("🚀 开始训练...")
        trainer.run()
        logger.success("🎉 训练圆满结束！")
    except KeyboardInterrupt:
        logger.warning("⛔ 用户中断训练。")
    except Exception as e:
        logger.error(f"❌ 训练出错：{e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()