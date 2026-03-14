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

    # 定义模型基础路径 (预训练模型位置)
    model_path = Path(cfg.path.pretrained_model) / cfg.model.model_name

    if not model_path.exists():
        raise FileNotFoundError(
            f"❌ 预训练模型目录未找到：{model_path}\n"
            f"请检查 cfg.path.pretrained_model 和 cfg.model.model_name 配置。"
        )

    # ================= 2. 初始化分词器 =================
    logger.info("🔧 初始化分词器...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    logger.success(f"✅ 分词器加载成功：{model_path.name}")

    # ================= 3. 构建数据集 =================
    logger.info("📂 构建数据集...")
    train_dataset = TextClassificationDataset(
        csv_path=cfg.path.divided_data / "train.csv",
        tokenizer=tokenizer,
        max_length=cfg.model.max_length
    )
    val_dataset = TextClassificationDataset(
        csv_path=cfg.path.divided_data / "val.csv",
        tokenizer=tokenizer,
        max_length=cfg.model.max_length
    )
    logger.info(f"✅ 数据集准备就绪 (Train: {len(train_dataset)}, Val: {len(val_dataset)})")

    # ================= 4. 构建 DataLoader =================
    collate_fn = get_collate_fn()
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=cfg.train.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.train.eval_batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=cfg.train.num_workers, pin_memory=True, drop_last=False)

    # ================= 5. 构建模型 =================
    logger.info("🤖 构建模型架构...")
    config = AutoConfig.from_pretrained(
        str(model_path),
        num_labels=cfg.finetune.num_labels,
        hidden_dropout_prob=getattr(cfg.finetune, 'hidden_dropout_prob', 0.1),
        output_attentions=False
    )

    model = BertForSequenceClassification.from_pretrained(
        str(model_path),
        config=config,
        ignore_mismatched_sizes=True,
        local_files_only=True
    )
    model.to(device)
    logger.success(f"✅ 模型加载成功")

    # ================= 6. 初始化训练器 (关键修改) =================
    # ✅ 将 tokenizer 传入 Trainer，以便保存
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        tokenizer=tokenizer  # <--- 新增参数
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