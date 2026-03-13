# evaluate.py (在根目录)
import torch
from pathlib import Path
from loguru import logger
from src import cfg
from src.evaluator import ModelEvaluator

# 配置日志
logger.add(f"{cfg.path.log_dir}/evaluation.log", rotation="10 MB", level="INFO")


def main():
    logger.info("🎯 启动评估统筹程序")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 配置路径
    model_base = Path(cfg.path.pretrained_model) / cfg.model.model_name
    best_ckpt = Path(cfg.path.saved_model) / "best_model.pth"

    # 可以选择评估 val 或 test
    data_to_eval = cfg.path.divided_data / "test.csv"
    # data_to_eval = cfg.path.divided_data / "val.csv"

    if not best_ckpt.exists():
        logger.error(f"❌ 未找到模型权重：{best_ckpt}")
        return

    evaluator = ModelEvaluator(model_base, best_ckpt, device)
    results = evaluator.evaluate(data_to_eval)

    logger.success("🎉 评估完成！")


if __name__ == '__main__':
    main()