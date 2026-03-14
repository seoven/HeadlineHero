import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from loguru import logger
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter  # ✅ 导入 TensorBoard

from src import cfg
from src.label_map import get_label_mapping


class Trainer:
    def __init__(self, model, train_loader, val_loader, device, tokenizer):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.tokenizer = tokenizer

        self.optimizer = AdamW(self.model.parameters(), lr=float(cfg.train.lr), weight_decay=cfg.train.weight_decay)

        total_steps = len(train_loader) * cfg.train.epochs
        warmup_steps = int(total_steps * cfg.finetune.warmup_ratio)

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        self.criterion = nn.CrossEntropyLoss()
        self.best_val_loss = float('inf')

        # 保存路径
        self.save_dir = Path(cfg.path.saved_model)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # ✅ TensorBoard 初始化
        self.log_dir = Path(cfg.path.log_dir) / "tensorboard"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        logger.info(f"📊 TensorBoard 日志将保存至：{self.log_dir}")

        # ✅ 早停机制参数
        es_cfg = getattr(cfg.finetune, 'early_stopping', {})
        self.patience = es_cfg.get('patience', 3)
        self.min_delta = es_cfg.get('min_delta', 0.0001)
        self.early_stop_counter = 0
        self.should_stop = False
        logger.info(f"⏳ 早停机制已启用 (Patience: {self.patience}, Min Delta: {self.min_delta})")

        logger.info(f"📂 模型全套文件将保存至：{self.save_dir}")

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss, correct, total = 0, 0, 0
        global_step = epoch * len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1} [Train]")
        for step, batch in enumerate(pbar):
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

            # ✅ 记录 TensorBoard (每个 step 记录一次 loss)
            self.writer.add_scalar('Loss/Train_Step', loss.item(), global_step + step)

        avg_loss = total_loss / len(self.train_loader)
        avg_acc = correct / total

        # ✅ 记录 TensorBoard (每个 epoch 记录一次平均指标)
        self.writer.add_scalar('Loss/Train_Epoch', avg_loss, epoch)
        self.writer.add_scalar('Accuracy/Train', avg_acc, epoch)

        return avg_loss, avg_acc

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

        avg_loss = total_loss / len(self.val_loader)
        avg_acc = correct / total

        # ✅ 记录 TensorBoard
        self.writer.add_scalar('Loss/Val_Epoch', avg_loss, epoch)
        self.writer.add_scalar('Accuracy/Val', avg_acc, epoch)

        return avg_loss, avg_acc

    def _save_config(self):
        try:
            id2label, label2id = get_label_mapping()
            self.model.config.id2label = id2label
            self.model.config.label2id = label2id
            self.model.config.num_labels = len(id2label)
            logger.debug("✅ 内存中 Config 已注入标签映射")
        except Exception as e:
            logger.warning(f"⚠️ 标签映射注入失败：{e}")

        self.model.config.save_pretrained(str(self.save_dir))
        logger.debug(f"💾 Config 已写入：{self.save_dir}/config.json")

    def _save_tokenizer(self):
        if self.tokenizer is None:
            logger.warning("⚠️ Tokenizer 为空，跳过保存")
            return
        self.tokenizer.save_pretrained(str(self.save_dir))
        logger.debug(f"💾 Tokenizer 已写入：{self.save_dir}")

    def save_checkpoint(self, epoch, val_loss, is_best, is_final=False):
        """
        :param is_final: 是否为早停或正常结束时的最终保存
        """
        # 如果是最佳模型或最终模型，保存全套
        if is_best or is_final:
            logger.info("🔄 正在保存模型全套文件...")
            try:
                self._save_tokenizer()
                self._save_config()

                filename = "best_model.pth"
                # 如果是早停触发的最终保存，且不是最佳（极少见情况），也可以覆盖或另存
                # 这里逻辑简化：只要触发了保存全套，都存为 best_model.pth
                # 或者如果是早停且非最佳，可以存为 final_model.pth

                save_path = self.save_dir / filename

                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'num_labels': self.model.config.num_labels,
                }
                torch.save(checkpoint, save_path)

                status = "🏆 最佳模型" if is_best else "🛑 早停最终模型"
                logger.success(f"{status} 全套保存成功：{filename}")
                logger.success(f"   📁 位置：{self.save_dir}")
                logger.success(f"   📦 包含：weights, config.json, tokenizer files")

            except Exception as e:
                logger.error(f"❌ 模型保存失败：{e}")
                raise

        # 如果不是最佳也不是最终，仅保存中间检查点 (可选)
        if not is_best and not is_final:
            filename = f"checkpoint_epoch_{epoch + 1}.pth"
            save_path = self.save_dir / filename
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_loss': val_loss,
            }, save_path)
            logger.info(f"💾 中间检查点已保存：{filename}")

    def run(self):
        logger.info("🚀 开始训练循环...")
        # 预先保存初始配置
        if self.tokenizer:
            self._save_tokenizer()
        self._save_config()

        for epoch in range(cfg.train.epochs):
            train_loss, train_acc = self.train_one_epoch(epoch)
            val_loss, val_acc = self.validate(epoch)

            logger.info(f"\n--- Epoch {epoch + 1}/{cfg.train.epochs} ---")
            logger.info(f"Train | Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
            logger.info(f"Val   | Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

            # === 早停判断逻辑 ===
            # 如果 val_loss 下降了 (超过 min_delta)
            if val_loss < self.best_val_loss - self.min_delta:
                self.best_val_loss = val_loss
                self.early_stop_counter = 0
                is_best = True
                logger.info("✨ 验证集 Loss 下降，更新最佳模型")
            else:
                self.early_stop_counter += 1
                is_best = False
                logger.warning(f"⚠️ 验证集 Loss 未改善 (连续 {self.early_stop_counter}/{self.patience} 次)")

            # 保存当前状态
            self.save_checkpoint(epoch, val_loss, is_best)

            # 检查是否触发早停
            if self.early_stop_counter >= self.patience:
                logger.success(f"🛑 触发早停机制！连续 {self.patience} 个 epoch 未改善。")
                # 即使不是最佳（理论上不可能，因为至少有一次初始化），也再保存一次以防万一
                # 实际上 best_model.pth 已经在之前更新过了，这里主要是为了结束循环
                self.should_stop = True
                break

        # 训练结束处理
        if self.should_stop:
            logger.info("🏁 训练因早停提前结束。")
        else:
            logger.info("🏁 训练按计划完成所有 Epoch。")

        self.writer.close()
        logger.success("🎉 训练全部完成！请检查 saved_model 目录及 TensorBoard 日志。")
        logger.info(f"💡 查看 TensorBoard 命令：tensorboard --logdir={self.log_dir}")