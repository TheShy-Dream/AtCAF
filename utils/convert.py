import numpy as np
import torch
def change_to_classify(y, output_size):
    if output_size == 2:
        y = y.squeeze()

        # 使用 torch.sign 来获取符号，0 将变为 1，其他保持不变
        # 然后使用 torch.where 来设置小于 0 的元素为 0
        return torch.where(y >= 0, torch.ones_like(y), torch.zeros_like(y)).long()

    if output_size == 7:
        y=y.squeeze()
        y = torch.clip(y, min=-3., max=3.)
        return torch.round(y).long() + 3