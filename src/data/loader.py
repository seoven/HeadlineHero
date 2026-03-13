import torch
from loguru import logger


def get_collate_fn():
    """
    返回标准的 collate_fn 函数。
    将其封装在这里，方便 train.py 调用，保持 train.py 整洁。
    """

    def collate_fn(batch):
        """
        将 list of dicts 转换为 batched tensors。
        假设 __getitem__ 返回的是固定长度的 tensor，直接 stack 即可。
        """
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        token_type_ids = torch.stack([item['token_type_ids'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'labels': labels
        }

    return collate_fn