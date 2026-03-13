# src/api_service.py
import torch
from transformers import AutoTokenizer, AutoConfig, BertForSequenceClassification
from pathlib import Path
from loguru import logger
from typing import List, Dict, Any

from src import cfg
from src.label_map import get_label_name  # 引入映射函数


class PredictionService:
    """
    模型预测服务单例类
    流程：
      1. 加载 saved_model/config.json (优先) 或 pretrained_model/config.json
      2. 加载 pretrained_model 的 tokenizer
      3. 构建模型架构
      4. 加载 saved_model/*.pth 权重
      5. 推理并映射标签名称
    """
    _instance = None
    _model = None
    _tokenizer = None
    _device = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PredictionService, cls).__new__(cls)
        return cls._instance

    def initialize(self, base_model_path: str, checkpoint_path: str):
        if self._model is not None:
            logger.warning("模型已加载，跳过初始化")
            return

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"🚀 正在组装模型... (设备：{self._device})")

        base_dir = Path(base_model_path)
        ckpt_file = Path(checkpoint_path)
        saved_model_dir = ckpt_file.parent

        if not base_dir.exists():
            raise FileNotFoundError(f"预训练模型目录不存在：{base_dir}")
        if not ckpt_file.exists():
            raise FileNotFoundError(f"微调权重文件不存在：{ckpt_file}")

        # 1. 确定 Config 来源
        # 优先使用 saved_model 目录下的 config.json (由 trainer.py 新生成，包含正确的 num_labels)
        config_path = saved_model_dir / "config.json"
        if config_path.exists():
            logger.info(f"📥 发现微调后的 config: {config_path}")
            config = AutoConfig.from_pretrained(str(saved_model_dir), local_files_only=True)
        else:
            logger.warning("⚠️ 未在 saved_model 找到 config.json，回退到 pretrained_model (可能导致维度错误)")
            config = AutoConfig.from_pretrained(str(base_dir), local_files_only=True)

        # 2. 加载 Tokenizer (始终来自 base_model)
        logger.info("📥 加载分词器...")
        self._tokenizer = AutoTokenizer.from_pretrained(str(base_dir), local_files_only=True)

        # 3. 构建模型
        self._model = BertForSequenceClassification.from_pretrained(
            str(base_dir),
            config=config,
            local_files_only=True,
            ignore_mismatched_sizes=False  # 因为用了微调后的 config，理论上应该完全匹配
        )

        # 4. 加载权重
        logger.info(f"📥 加载微调权重：{ckpt_file.name}")
        checkpoint = torch.load(ckpt_file, map_location=self._device)

        state_dict = checkpoint.get('model_state_dict', checkpoint)
        load_result = self._model.load_state_dict(state_dict, strict=True)

        if load_result.missing_keys or load_result.unexpected_keys:
            logger.error(f"❌ 权重加载失败 - 缺失：{load_result.missing_keys}, 多余：{load_result.unexpected_keys}")
            raise RuntimeError("模型权重加载不匹配，请检查 config 和权重文件是否对应")

        self._model.to(self._device)
        self._model.eval()

        logger.success("✅ 模型服务初始化完成 (含 Label Map)")

    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        if self._model is None:
            raise RuntimeError("模型未初始化")

        results = []

        # 1. 编码
        encodings = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=cfg.model.max_length,
            return_tensors="pt"
        ).to(self._device)

        # 2. 推理
        with torch.no_grad():
            outputs = self._model(**encodings)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            confidences, predicted_ids = torch.max(probs, dim=-1)

        # 3. 构建结果 (包含 ID 和 Name)
        for i, text in enumerate(texts):
            pred_id = int(predicted_ids[i].item())
            confidence = float(confidences[i].item())

            results.append({
                "text": text,
                "label_id": pred_id,
                "label_name": get_label_name(pred_id),  # 核心：映射为中文
                "confidence": round(confidence, 4)
            })

        return results


# 全局单例
predictor = PredictionService()