import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, AutoConfig
from tqdm import tqdm
from loguru import logger
from src import cfg
import json
from pathlib import Path


class Trainer:
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.optimizer = AdamW(self.model.parameters(), lr=float(cfg.train.lr), weight_decay=cfg.train.weight_decay)

        total_steps = len(train_loader) * cfg.train.epochs
        warmup_steps = int(total_steps * cfg.finetune.warmup_ratio)

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        self.criterion = nn.CrossEntropyLoss()
        self.best_val_loss = float('inf')

        # 保存路径
        self.save_dir = cfg.path.saved_model
        self.save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📂 模型及配置将保存至：{self.save_dir}")

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss, correct, total = 0, 0, 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1} [Train]")
        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            token_type_ids = batch['token_type_ids'].to(self.device)
            labels = batch['labels'].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                 token_type_ids=token_type_ids, labels=labels)

            loss = outputs['loss']
            logits = outputs['logits']

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.finetune.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)

            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        return total_loss / len(self.train_loader), correct / total

    def validate(self, epoch):
        self.model.eval()
        total_loss, correct, total = 0, 0, 0

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch + 1} [Val] ")
        with torch.no_grad():
            for batch in pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                     token_type_ids=token_type_ids, labels=labels)

                loss = outputs['loss']
                logits = outputs['logits']

                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                correct += torch.sum(preds == labels).item()
                total += labels.size(0)

        return total_loss / len(self.val_loader), correct / total

    def _save_config(self):
        """
        单独保存模型的 config 为 JSON 文件
        确保推理时能加载到正确的 num_labels
        """
        config_path = self.save_dir / "config.json"
        # 获取模型当前的 config
        config = self.model.config
        # 序列化为 JSON
        # transformers 的 PretrainedConfig 有 save_pretrained 方法，但为了精准控制，我们手动存或调用它
        config.save_pretrained(str(self.save_dir))
        logger.info(f"💾 模型配置已保存至：{config_path} (num_labels={config.num_labels})")

    def save_checkpoint(self, epoch, val_loss, is_best):
        # 1. 先保存 config.json (无论是不是 best，config 都是一样的，但为了保险每次更新或仅首次保存均可)
        # 这里选择在每次保存 checkpoint 时都覆盖一下 config，确保万无一失
        self._save_config()

        # 2. 保存权重 .pth
        filename = "best_model.pth" if is_best else f"checkpoint_epoch_{epoch + 1}.pth"
        save_path = self.save_dir / filename

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'num_labels': cfg.finetune.num_labels, # 冗余备份，主要依赖 config.json
            'config_info': {
                'hidden_size': self.model.config.hidden_size,
                'num_hidden_layers': self.model.config.num_hidden_layers
            }
        }
        torch.save(checkpoint, save_path)

        if is_best:
            logger.success(f"🏆 新最佳模型已保存：{filename} (Val Loss: {val_loss:.4f})")
        else:
            logger.info(f"💾 检查点已保存：{filename}")

    def run(self):
        logger.info("🚀 开始训练循环...")
        for epoch in range(cfg.train.epochs):
            train_loss, train_acc = self.train_one_epoch(epoch)
            val_loss, val_acc = self.validate(epoch)

            logger.info(f"\n--- Epoch {epoch + 1}/{cfg.train.epochs} ---")
            logger.info(f"Train | Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
            logger.info(f"Val   | Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            self.save_checkpoint(epoch, val_loss, is_best)

        logger.success("🎉 训练全部完成！")