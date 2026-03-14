# src/api_service.py
import torch
from transformers import AutoTokenizer, AutoConfig, BertForSequenceClassification
from pathlib import Path
from loguru import logger
from typing import List, Dict, Any

from src import cfg


class PredictionService:
    _instance = None
    _model = None
    _tokenizer = None
    _device = None
    _id2label = {}

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
        saved_dir = ckpt_file.parent  # 微调保存目录

        if not base_dir.exists():
            raise FileNotFoundError(f"预训练模型目录不存在：{base_dir}")
        if not ckpt_file.exists():
            raise FileNotFoundError(f"微调权重文件不存在：{ckpt_file}")

        # ================= 步骤 1: 强制加载 Saved_Model 的 Config =================
        config_path = saved_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"❌ 致命错误：未找到微调后的 config.json: {config_path}")

        logger.info(f"📥 [1/4] 加载微调配置 (含标签映射): {config_path}")
        config = AutoConfig.from_pretrained(str(saved_dir), local_files_only=True)

        if hasattr(config, 'id2label') and config.id2label:
            self._id2label = config.id2label
            logger.info(f"✅ 标签映射已就绪：{list(self._id2label.values())}")
        else:
            raise RuntimeError("模型配置不完整，缺少 id2label 字段！请重新训练。")

        # ================= 步骤 2: 智能加载 Tokenizer (核心修改) =================
        logger.info("📥 [2/4] 加载分词器...")

        # 策略：优先检查 saved_dir 是否有 tokenizer 文件 (实现模型独立交付)
        tokenizer_files_exist = (saved_dir / "vocab.txt").exists() or \
                                (saved_dir / "tokenizer_config.json").exists()

        if tokenizer_files_exist:
            logger.info(f"   ✅ 发现独立 Tokenizer，从 {saved_dir} 加载 (推荐)")
            self._tokenizer = AutoTokenizer.from_pretrained(str(saved_dir), local_files_only=True)
        else:
            logger.warning(f"   ⚠️ 未在 {saved_dir} 找到 Tokenizer，回退到基础模型目录 {base_dir}")
            if not (base_dir / "vocab.txt").exists():
                raise FileNotFoundError(f"基础模型目录也缺少 Tokenizer 文件：{base_dir}")
            self._tokenizer = AutoTokenizer.from_pretrained(str(base_dir), local_files_only=True)

        # ================= 步骤 3: 构建模型 =================
        logger.info("📥 [3/4] 构建模型架构...")
        self._model = BertForSequenceClassification.from_pretrained(
            str(base_dir),  # 骨架权重依然来自 base_dir (除非你保存了完整 pytorch_model.bin)
            config=config,
            local_files_only=True,
            ignore_mismatched_sizes=True
        )

        # ================= 步骤 4: 加载微调权重 =================
        logger.info(f"📥 [4/4] 加载微调权重：{ckpt_file.name}")
        checkpoint = torch.load(ckpt_file, map_location=self._device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)

        load_result = self._model.load_state_dict(state_dict, strict=True)
        if load_result.missing_keys or load_result.unexpected_keys:
            critical = [k for k in (load_result.missing_keys + load_result.unexpected_keys) if 'classifier' in k]
            if critical:
                logger.error(f"❌ 权重加载失败 - 关键层不匹配：{critical}")
                raise RuntimeError("模型权重与架构严重不匹配")
            else:
                logger.warning(f"⚠️ 忽略非关键权重差异")

        self._model.to(self._device)
        self._model.eval()
        logger.success("✅ 模型服务初始化完成 (Tokenizer 独立加载模式)")

    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        if self._model is None:
            raise RuntimeError("模型未初始化")

        results = []
        encodings = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=cfg.model.max_length,
            return_tensors="pt"
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**encodings)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            confidences, predicted_ids = torch.max(probs, dim=-1)

        for i, text in enumerate(texts):
            pred_id = int(predicted_ids[i].item())
            confidence = float(confidences[i].item())
            label_name = self._id2label.get(pred_id, f"Unknown_{pred_id}")

            results.append({
                "text": text,
                "label_id": pred_id,
                "label_name": label_name,
                "confidence": round(confidence, 4)
            })
        return results


predictor = PredictionService()