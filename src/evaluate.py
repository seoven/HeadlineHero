# src/evaluate.py
"""
评估脚本入口
用法：python -m src.evaluate 或 python src/evaluate.py
"""
import torch
from pathlib import Path
from loguru import logger

from src import cfg
from src.evaluator import ModelEvaluator  # ✅ 导入上面修改好的类

# 配置日志 (如果 evaluator 里没配，这里可以补配，或者共用 trainer 的日志配置)
logger.add(f"{cfg.path.log_dir}/evaluation.log", rotation="10 MB", retention="7 days", level="INFO")


def main():
    logger.info("🚀 启动模型评估流程...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🖥️  使用设备：{device}")

    # 1. 定义路径
    model_base = Path(cfg.path.pretrained_model) / cfg.model.model_name
    best_ckpt = Path(cfg.path.saved_model) / "best_model.pth"

    # 自动 fallback: 如果没有 best_model.pth，找最新的 .pth
    if not best_ckpt.exists():
        pth_files = list(Path(cfg.path.saved_model).glob("*.pth"))
        if pth_files:
            best_ckpt = sorted(pth_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
            logger.warning(f"⚠️ 未找到 best_model.pth，自动使用最新权重：{best_ckpt.name}")
        else:
            logger.error("❌ 在 saved_model 目录下未找到任何 .pth 文件，无法评估。")
            return

    # 默认评估验证集，也可以通过命令行参数扩展
    eval_data = cfg.path.divided_data / "val.csv"

    if not eval_data.exists():
        logger.error(f"❌ 评估数据文件不存在：{eval_data}")
        return

    # 2. 初始化评估器
    try:
        evaluator = ModelEvaluator(
            model_path=str(model_base),
            checkpoint_path=str(best_ckpt),
            device=device
        )

        # 3. 执行评估
        results = evaluator.evaluate(eval_data, output_report=True)

        logger.success("🎉 评估完成！")

    except Exception as e:
        logger.error(f"❌ 评估过程中发生错误：{e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()