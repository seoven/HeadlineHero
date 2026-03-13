没问题！根据你提供的真实项目结构（tree 输出），我为你重新生成了精准的 README.md。

主要调整点：
目录结构：完全匹配你的 TouManFen 文件夹布局，包括 data/raw_data (原始分片), data/divided_data, data/processed_data 等。
配置文件：指明使用 config.yml 而不是 Python 配置文件。
日志系统：增加了 logs/ 目录的说明，涵盖数据清洗、训练、评估和 API 日志。
模块划分：准确描述了 src/data/ 下的数据处理模块 (clean.py, split_data.py 等) 和核心逻辑 (trainer.py, evaluate.py)。
启动方式：修正为根目录下的 serve.py。

📰 TouManFen: 基于 BERT 的新闻文本自动分类系统

本项目是一个完整的端到端新闻文本分类解决方案。它使用 HuggingFace Transformers (BERT) 作为底座模型，在自定义新闻数据集上进行微调，并通过 FastAPI 提供高性能的 RESTful API 服务。

✨ 主要特性

🤖 核心模型: 基于 bert-base-chinese 进行迁移学习，针对 10 类新闻场景优化。
📊 完整数据流水线:
  支持原始分片数据 (data/raw_data/*.csv) 的自动清洗与合并。
  包含标准化的训练/验证/测试集划分 (data/divided_data/)。
  生成统一的处理后数据 (data/processed_data/)。
🎯 高精度训练:
  模块化训练器 (src/trainer.py)，支持断点续训。
  自动保存最佳权重 (best_model.pth) 及对应的模型配置 (config.json)。
  独立的评估模块 (src/evaluate.py) 生成详细报告。
📦 工程化部署:
  使用 FastAPI 构建高并发 API 服务 (serve.py)。
  自动生成 Swagger UI 文档 (/docs)。
  支持批量预测 (Batch Prediction)。
🏷️ 智能映射: 内置标签映射机制 (src/label_map.py)，推理结果直接返回中文类别名称。
⚙️ 配置驱动: 使用 config.yml 统一管理路径、超参数和日志设置。
📝 完善日志: 独立记录数据清洗、分割、训练、评估及 API 运行日志 (logs/)。

📂 项目结构

TouManFen/
├── config.yml              # [核心] 全局配置文件 (路径、超参数、日志配置)
├── requirements.txt        # Python 依赖列表
├── serve.py                # [入口] API 服务启动脚本
├── data/                   # 数据目录
│   ├── raw_data/              # 原始分片数据 (data_part_1.csv ... data_part_51.csv)
│   ├── divided_data/       # [输出] 划分后的数据集
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   └── processed_data/     # [输出] 清洗合并后的完整数据
│       └── processed_data.csv
├── logs/                   # 日志目录
│   ├── data_cleaning.log   # 数据清洗日志
│   ├── data_splitting.log  # 数据划分日志
│   ├── training.log        # 模型训练日志
│   ├── evaluation.log      # 模型评估日志
│   └── api.log             # API 服务运行日志
├── model/                  # 模型存储
│   ├── pretrained_model/   # HuggingFace 底座模型
│   │   └── bert_base_chinese/
│   │       ├── config.json
│   │       ├── pytorch_model.bin
│   │       ├── tokenizer_config.json
│   │       └── vocab.txt
│   └── saved_model/        # [输出] 微调成果
│       ├── best_model.pth          # 最佳模型权重
│       ├── checkpoint_epoch_x.pth  # 历史检查点
│       └── config.json             # 微调后的模型配置 (含 num_labels)
└── src/                    # 源代码包
    ├── init.py
    ├── config.py           # 配置加载工具 (读取 config.yml)
    ├── label_map.py        # 标签 ID  中文名称 映射
    ├── trainer.py          # 核心训练逻辑
    ├── train.py            # 训练入口脚本 (调用 trainer)
    ├── evaluate.py         # 评估入口脚本
    ├── evaluator.py        # 评估逻辑实现
    ├── api_service.py      # 推理服务封装 (单例模式)
    └── data/               # 数据处理子模块
        ├── clean.py        # 数据清洗逻辑
        ├── split_data.py   # 数据集划分逻辑
        ├── dataset.py      # PyTorch Dataset 定义
        └── loader.py       # DataLoader 封装

🚀 快速开始

模型下载

由于模型文件较大，已放在百度网盘，下载 HeadlineHero/ 文件夹下的 model.zip 压缩文件
（必须项）：将解压后的目录中的 pretrained_model/ 文件夹整个替代项目文件夹下的同名文件夹即可得到基座预训练模型
（可选项）：将解压后的 saved_model/ 文件夹整个替代项目文件夹下的同名文件夹即可获得原项目训练和微调好的模型。或不选择替换，则需要自己跑一遍训练和评估过程，即运行 train.py 文件
百度网盘下载链接：

环境准备

确保已安装 Python 3.10+，并安装依赖：

pip install -r requirements.txt

配置检查

打开 config.yml，确认以下路径与你实际环境一致（通常默认即可）：
path.pretrained_model: 指向 model/pretrained_model
path.saved_model: 指向 model/saved_model
path.data_raw: 指向 data/raw_data
path.data_divided: 指向 data/divided_data

数据预处理 (可选)

如果尚未生成 divided_data，可运行数据处理流程（具体命令视你的 train.py 或独立脚本而定，通常训练脚本会自动处理）：注：本项目的数据处理逻辑封装在 src/data/clean.py 和 src/data/split_data.py 中，通常在首次运行训练时自动触发。

模型训练

运行训练脚本。训练过程会记录到 logs/training.log，并在 model/saved_model/ 保存结果。

python src/train.py
(注意：训练入口在 src/train.py，而非根目录)

训练产出:
model/saved_model/best_model.pth: 最佳模型权重。
model/saved_model/config.json: 关键，包含微调后的架构配置。

模型评估

运行评估脚本，查看准确率、F1 分数等指标，日志输出至 logs/evaluation.log。

python src/evaluate.py

启动 API 服务

训练完成后，从项目根目录启动 FastAPI 服务：

python serve.py

服务默认运行在 http://0.0.0.0:8000。日志实时写入 logs/api.log。

🌐 API 使用指南

启动服务后，访问 Swagger UI 查看交互式文档：
👉 http://127.0.0.1:8000/docs

接口详情

健康检查
URL: GET /health
描述: 检查模型加载状态及设备信息。

新闻分类预测
URL: POST /predict
Content-Type: application/json
请求体:
    {
    "texts": [
      "高盛大摩唱多商品 上调原油目标20",
      "英超-中场怪才连场入球 曼联绝杀朴茨茅斯夺首胜"
    ]
  }
  
响应示例:
    {
    "success": true,
    "data": [
      {
        "text": "高盛大摩唱多商品 上调原油目标20",
        "label_id": 0,
        "label_name": "财经",
        "confidence": 0.9821
      },
      {
        "text": "英超-中场怪才连场入球 曼联绝杀朴茨茅斯夺首胜",
        "label_id": 7,
        "label_name": "体育",
        "confidence": 0.9905
      }
    ],
    "message": "预测成功"
  }
  

支持的分类标签
ID   类别名称   ID   类别名称
0   财经   5   社会

1   房产   6   国际

2   股票   7   体育

3   教育   8   游戏

4   科技   9   娱乐

⚙️ 配置说明 (config.yml)

项目通过 src/config.py 读取 config.yml。关键配置项示例：

path:
  pretrained_model: "./model/pretrained_model"
  saved_model: "./model/saved_model"
  data_raw: "./data/data1"
  data_divided: "./data/divided_data"
  log_dir: "./logs"

model:
  model_name: "bert_base_chinese"
  max_length: 128

finetune:
  num_labels: 10
  lr: 2e-5
  epochs: 5
  batch_size: 32

logging:
  level: "INFO"
  format: "%(asctime)s - %(levelname)s - %(message)s"

💡 技术实现

Config 同步机制:
    训练时利用 config.save_pretrained() 将 num_labels 等关键参数保存到 saved_model/config.json。
    推理时 (api_service.py) 优先加载该配置，彻底避免预训练模型默认配置与微调任务不匹配的问题。

模块化数据管道:
    src/data/clean.py: 负责去重、去噪。
    src/data/split_data.py: 负责按比例划分训练/验证/测试集。
    解耦设计使得数据预处理可独立运行或嵌入训练流程。

单例预测服务:
    PredictionService 采用单例模式，确保模型在内存中只加载一次，所有 HTTP 请求共享实例，极大降低延迟。

全方位日志监控:
    不同阶段（清洗、训练、推理）写入不同日志文件，便于问题追踪和性能分析。

📝 License

MIT License