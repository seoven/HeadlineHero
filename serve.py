# serve.py
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from loguru import logger
import sys
from pathlib import Path

from src.api_service import predictor
from src import cfg

# 配置日志
logger.add(f"{cfg.path.log_dir}/api.log", rotation="10 MB", level="INFO")

# 临时占位，实际描述会在启动后由动态数据填充，或者在 startup 后更新
app = FastAPI(
    title="新闻分类模型 API (Config-Driven)",
    description="基于 BERT 的新闻文本分类服务。\n\n"
                "**特性**: \n"
                "- 自动返回 `label_id` (数字) 和 `label_name` (中文类别)\n"
                "- 支持批量预测\n"
                "- 标签映射集成于模型 config.json，无需额外文件",
    version="3.0.0"
)


class PredictRequest(BaseModel):
    texts: List[str] = Field(..., description="待预测的文本列表", min_items=1, max_items=32)

    class Config:
        schema_extra = {
            "example": {
                "texts": [
                    "高盛大摩唱多商品 上调原油目标20",
                    "英超-中场怪才连场入球 曼联绝杀朴茨茅斯夺首胜"
                ]
            }
        }


class PredictItem(BaseModel):
    text: str
    label_id: int
    label_name: str
    confidence: float


class PredictResponse(BaseModel):
    success: bool
    data: List[PredictItem]
    message: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    try:
        base_model_dir = Path(cfg.path.pretrained_model) / cfg.model.model_name
        saved_model_dir = Path(cfg.path.saved_model)

        checkpoint_file = saved_model_dir / "best_model.pth"
        if not checkpoint_file.exists():
            pth_files = list(saved_model_dir.glob("*.pth"))
            if pth_files:
                checkpoint_file = sorted(pth_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
                logger.warning(f"未找到 best_model.pth，使用最新权重：{checkpoint_file.name}")
            else:
                raise FileNotFoundError(f"在 {saved_model_dir} 下未找到任何 .pth 文件")

        logger.info(f"📂 基础模型：{base_model_dir}")
        logger.info(f"💾 权重文件：{checkpoint_file}")

        # 初始化预测服务 (此时会加载 config 并解析 id2label)
        predictor.initialize(str(base_model_dir), str(checkpoint_file))

        # 动态更新 API 文档中的描述，展示实际加载的标签
        labels = list(predictor._id2label.values())
        app.description += f"\n\n**当前支持类别**: {', '.join(labels)}"

        logger.info(f"🌐 API 服务启动成功 (加载类别：{len(labels)})")

    except Exception as e:
        logger.error(f"❌ 服务启动失败：{e}")
        sys.exit(1)


@app.get("/")
async def root():
    # 从已加载的 predictor 中获取标签
    labels = list(predictor._id2label.values()) if predictor._id2label else []
    return {
        "message": "新闻分类 API 运行中",
        "version": "3.0.0 (Config-Driven)",
        "docs": "/docs",
        "available_labels": labels
    }


@app.get("/health")
async def health_check():
    labels = list(predictor._id2label.values()) if predictor._id2label else []
    return {
        "status": "healthy",
        "model_loaded": predictor._model is not None,
        "device": str(predictor._device) if predictor._device else "None",
        "labels_count": len(labels),
        "labels": labels
    }


@app.post("/predict", response_model=PredictResponse)
async def predict_api(request: PredictRequest):
    try:
        valid_texts = [t.strip() for t in request.texts if t.strip()]
        if not valid_texts:
            raise HTTPException(status_code=400, detail="文本列表不能为空")

        results = predictor.predict(valid_texts)

        return PredictResponse(
            success=True,
            data=results,
            message="预测成功"
        )
    except Exception as e:
        logger.exception(f"预测错误：{e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "serve:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1
    )