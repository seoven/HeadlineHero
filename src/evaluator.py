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
    """
    专门用于模型训练完成后的深度评估。
    功能：加载最佳模型，在验证集/测试集上运行，输出详细报告 (F1, Confusion Matrix)。
    """

    def __init__(self, model_path, checkpoint_path, device):
        self.device = device
        self.model_path = Path(model_path)
        self.checkpoint_path = Path(checkpoint_path)

        self.tokenizer = None
        self.model = None
        self.config = None

    def load_resources(self):
        """加载分词器、配置和模型权重"""
        logger.info("🔧 加载模型资源...")

        # 1. Tokenizer & Config
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型基础路径不存在：{self.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self.config = AutoConfig.from_pretrained(
            str(self.model_path),
            num_labels=cfg.finetune.num_labels,
            local_files_only=True
        )

        # 2. Model Architecture
        self.model = BertForSequenceClassification.from_pretrained(
            str(self.model_path),
            config=self.config,
            local_files_only=True
        )
        self.model.to(self.device)
        self.model.eval()

        # 3. Weights
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"权重文件不存在：{self.checkpoint_path}")

        logger.info(f"📥 加载权重：{self.checkpoint_path.name}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        logger.success("✅ 模型资源加载完毕")

    def build_dataloader(self, csv_path, batch_size=None):
        """构建评估专用的 DataLoader"""
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
            num_workers=0,  # 评估时通常设为 0 保证稳定性
            pin_memory=True if self.device.type == 'cuda' else False
        )

    def evaluate(self, data_path, output_report=True):
        """执行评估并生成详细报告"""
        if self.model is None:
            self.load_resources()

        loader = self.build_dataloader(data_path)
        logger.info(f"🚀 开始在 {Path(data_path).name} 上进行深度评估...")

        all_preds = []
        all_labels = []
        total_loss = 0.0
        criterion = torch.nn.CrossEntropyLoss()

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

        # 计算指标
        avg_loss = total_loss / len(loader.dataset)
        accuracy = accuracy_score(all_labels, all_preds)

        logger.info("-" * 30)
        logger.info(f"📊 基础指标")
        logger.info(f"Loss: {avg_loss:.4f}")
        logger.info(f"Accuracy: {accuracy:.4f}")

        if output_report:
            logger.info("\n📋 详细分类报告 (Precision/Recall/F1):")
            # target_names 需要根据你的 label_map 来定，这里假设是 0,1,2...
            # 如果你有 label_map，可以传入 target_names=list(label_map.values())
            report = classification_report(all_labels, all_preds, digits=4)
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


# 方便直接运行的主函数
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 路径配置
    model_base = Path(cfg.path.pretrained_model) / cfg.model.model_name
    best_ckpt = Path(cfg.path.saved_model) / "best_model.pth"
    eval_data = cfg.path.divided_data / "val.csv"  # 或者 test.csv

    evaluator = ModelEvaluator(model_base, best_ckpt, device)
    evaluator.evaluate(eval_data)