import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
from loguru import logger


class TextClassificationDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=128):
        """
        纯净数据集类。
        :param csv_path: 数据文件路径
        :param tokenizer: 分词器实例
        :param max_length: 截断/填充长度
        """
        self.csv_path = Path(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

        if not self.csv_path.exists():
            raise FileNotFoundError(f"数据文件不存在：{self.csv_path}")

        logger.debug(f"正在读取数据：{self.csv_path}")
        self.data = pd.read_csv(str(self.csv_path))

        # 检查列名是否为 'title' 和 'label'
        if 'title' not in self.data.columns or 'label' not in self.data.columns:
            raise ValueError(f"CSV 缺少 'title' 或 'label' 列。当前列名：{list(self.data.columns)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = str(self.data.iloc[idx]['title'])
        label = int(self.data.iloc[idx]['label'])

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,  # BERT 需要这个
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }