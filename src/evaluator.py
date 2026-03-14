# src/evaluator.py
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from loguru import logger
from transformers import AutoTokenizer, AutoConfig, BertForSequenceClassification
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

from src import cfg
from src.data.dataset import TextClassificationDataset
from src.data.loader import get_collate_fn


class ModelEvaluator:
    def __init__(self, model_path, checkpoint_path, device):
        """
        :param model_path: 预训练模型基础目录 (用于补全底层权重，若 saved_model 无 tokenizer 则用于加载 tokenizer)
        :param checkpoint_path: 微调后的权重文件路径 (e.g., ./model/saved_model/best_model.pth)
        :param device: 运行设备
        """
        self.device = device
        self.pretrained_dir = Path(model_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.saved_dir = self.checkpoint_path.parent  # ✅ 核心：微调保存目录 (包含 config, tokenizer, weights)

        self.tokenizer = None
        self.model = None
        self.config = None

    def load_resources(self):
        logger.info("🔧 加载模型资源...")

        # ================= 步骤 1: 强制加载 Saved_Model 的配置 =================
        config_file = self.saved_dir / "config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"❌ 致命错误：未找到微调后的 config.json: {config_file}")

        logger.info(f"📥 [1/4] 加载微调配置 (含标签映射): {config_file}")
        self.config = AutoConfig.from_pretrained(str(self.saved_dir), local_files_only=True)

        if hasattr(self.config, 'id2label') and self.config.id2label:
            logger.info(f"✅ 标签映射已就绪：{list(self.config.id2label.values())}")
        else:
            logger.warning("⚠️ Config 中缺少 id2label，评估报告将仅显示数字 ID")

        # ================= 步骤 2: 智能加载 Tokenizer (核心修改) =================
        logger.info("📥 [2/4] 加载分词器...")

        # 策略：优先检查 saved_dir 是否有 tokenizer 文件 (实现模型独立交付)
        tokenizer_files_exist = (self.saved_dir / "vocab.txt").exists() or \
                                (self.saved_dir / "tokenizer_config.json").exists()

        if tokenizer_files_exist:
            logger.info(f"   ✅ 发现独立 Tokenizer，从 {self.saved_dir} 加载 (推荐)")
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.saved_dir), local_files_only=True)
        else:
            logger.warning(f"   ⚠️ 未在 {self.saved_dir} 找到 Tokenizer，回退到基础模型目录 {self.pretrained_dir}")
            if not self.pretrained_dir.exists():
                raise FileNotFoundError(f"基础模型目录也不存在：{self.pretrained_dir}")
            if not (self.pretrained_dir / "vocab.txt").exists():
                raise FileNotFoundError(f"基础模型目录缺少 Tokenizer 文件：{self.pretrained_dir}")
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.pretrained_dir), local_files_only=True)

        # ================= 步骤 3: 构建模型架构 =================
        # 策略：使用 pretrained_dir 的底层权重 + 强制注入 saved_dir 的 config
        logger.info("📥 [3/4] 构建模型架构...")

        if not self.pretrained_dir.exists():
            raise FileNotFoundError(f"❌ 预训练模型目录不存在 (需要底层权重): {self.pretrained_dir}")

        try:
            self.model = BertForSequenceClassification.from_pretrained(
                str(self.pretrained_dir),  # 底层权重来源 (Embeddings, Encoder)
                config=self.config,  # ✅ 强制使用微调后的 Config (num_labels=10)
                local_files_only=True,
                ignore_mismatched_sizes=True  # ✅ 忽略预训练分类头的大小不匹配
            )
        except Exception as e:
            logger.error(f"❌ 模型构建失败：{e}")
            raise e

        self.model.to(self.device)
        self.model.eval()

        # ================= 步骤 4: 加载微调后的完整权重 =================
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"❌ 权重文件不存在：{self.checkpoint_path}")

        logger.info(f"📥 [4/4] 加载微调权重：{self.checkpoint_path.name}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)

        # 此时模型架构已完全对齐 (10 类)，应能严格匹配
        load_result = self.model.load_state_dict(state_dict, strict=True)

        if load_result.missing_keys or load_result.unexpected_keys:
            critical_errors = [k for k in (load_result.missing_keys + load_result.unexpected_keys)
                               if 'classifier' in k or 'pooler' in k]
            if critical_errors:
                logger.error(f"❌ 关键层级权重不匹配：{critical_errors}")
                raise RuntimeError("模型权重与架构严重不匹配")
            else:
                logger.debug(f"ℹ️ 忽略非关键权重差异")

        logger.success(
            "✅ 模型资源加载完毕 (Tokenizer 独立加载模式)" if tokenizer_files_exist else "✅ 模型资源加载完毕 (兼容模式)")

    def build_dataloader(self, csv_path, batch_size=None):
        if batch_size is None:
            batch_size = cfg.train.eval_batch_size

        dataset = TextClassificationDataset(
            csv_path=csv_path,
            tokenizer=self.tokenizer,
            max_length=cfg.model.max_length
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=get_collate_fn(),
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )

    def evaluate(self, data_path, output_report=True):
        if self.model is None:
            self.load_resources()

        loader = self.build_dataloader(data_path)
        logger.info(f"🚀 开始在 {Path(data_path).name} 上进行深度评估...")

        all_preds = []
        all_labels = []
        total_loss = 0.0

        with torch.no_grad():
            pbar = tqdm(loader, desc="Evaluating")
            for batch in pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels
                )

                loss = outputs.loss
                logits = outputs.logits
                total_loss += loss.item() * labels.size(0)

                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = total_loss / len(loader.dataset)
        accuracy = accuracy_score(all_labels, all_preds)

        logger.info("-" * 30)
        logger.info(f"📊 基础指标")
        logger.info(f"Loss: {avg_loss:.4f}")
        logger.info(f"Accuracy: {accuracy:.4f}")

        if output_report:
            logger.info("\n📋 详细分类报告 (Precision/Recall/F1):")
            target_names = None
            if hasattr(self.config, 'id2label') and self.config.id2label:
                # 确保按 ID 顺序排列 (0, 1, 2...)
                max_id = max(self.config.id2label.keys())
                target_names = [self.config.id2label.get(i, f"Label_{i}") for i in range(max_id + 1)]
                logger.info(f"   🏷️ 使用标签映射：{target_names}")

            report = classification_report(all_labels, all_preds, target_names=target_names, digits=4)
            logger.info("\n" + report)

            logger.info("\n🧩 混淆矩阵:")
            cm = confusion_matrix(all_labels, all_preds)
            logger.info("\n" + np.array2string(cm, separator=', '))

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'predictions': all_preds,
            'labels': all_labels
        }


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 路径定义
    model_base = Path(cfg.path.pretrained_model) / cfg.model.model_name
    best_ckpt = Path(cfg.path.saved_model) / "best_model.pth"
    eval_data = cfg.path.divided_data / "val.csv"

    # 检查文件是否存在
    if not best_ckpt.exists():
        logger.error(f"❌ 未找到权重文件：{best_ckpt}")
        logger.info("💡 请先运行训练脚本生成模型。")
    else:
        evaluator = ModelEvaluator(model_base, best_ckpt, device)
        evaluator.evaluate(eval_data)