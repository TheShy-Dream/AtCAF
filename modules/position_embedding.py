import math

import torch
import torch.nn as nn

# Code adapted from the fairseq repo.

def make_positions(tensor, padding_idx, left_pad):
    """Replace non-padding symbols with their position numbers.
    Position numbers begin at padding_idx+1.
    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).

    tensor： torch.Size([32, 50])
    """
    max_pos = padding_idx + 1 + tensor.size(1)  # 51
    device = tensor.get_device()  # 0
    buf_name = f'range_buf_{device}'
    # if not hasattr(make_positions, buf_name):  # 进
    #     setattr(make_positions, buf_name, tensor.new())
        # tensor.new(): 创建一个新的Tensor，该Tensor的type和device都和原有Tensor一致，且无内容。
        # setattr(object, name, value):设置属性值，该属性不一定是存在的。

    # setattr(make_positions, buf_name, getattr(make_positions, buf_name).type_as(tensor))
    # ten1.type_as(ten2): 将1的数据类型转换为2的数据类型
    # getattr(object, name[, default])函数用于返回一个对象属性值。
    setattr(make_positions, buf_name, tensor.new())
    # print('*'*40)
    # print(getattr(make_positions, buf_name).numel())  # 0 50
    # print(getattr(make_positions, buf_name).size())  # 0 50
    # print(getattr(make_positions, buf_name))
    # print(torch.arange(padding_idx + 1, max_pos))  # 1-50
    # print('*'*40)

    if getattr(make_positions, buf_name).numel() < max_pos:  # numel(): 获取张量元素个数
        torch.arange(padding_idx + 1, max_pos, out=getattr(make_positions, buf_name))
        # eg: torch.arange(1,6)  # tensor([1, 2, 3, 4, 5])
    mask = tensor.ne(padding_idx)  # torch.Size([32, 50]) ne(): 将mask和padding_idx对应位置比较，不想等返回Ture
    positions = getattr(make_positions, buf_name)[:tensor.size(1)].expand_as(tensor)  # torch.Size([32, 50])
    if left_pad:
        positions = positions - mask.size(1) + mask.long().sum(dim=1).unsqueeze(1)
    new_tensor = tensor.clone()
    return new_tensor.masked_scatter_(mask, positions[mask]).long()


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.
    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    """

    def __init__(self, embedding_dim, padding_idx=0, left_pad=0, init_size=128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.left_pad = left_pad
        self.weights = dict()  # device --> actual weight; due to nn.DataParallel :-(
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        eg: num_embeddings:30  embedding_dim:51(变)  padding_idx:0
        """
        half_dim = embedding_dim // 2  # 15
        emb_c1 = math.log(10000) / (half_dim - 1)  # 0.6578814551411559

        emb_c2 = torch.arange(embedding_dim, dtype=torch.int32)  # shape: 30

        emb = torch.exp((emb_c2 // 2).to(torch.float) * -emb_c1)  # shape: 30 (embedding_dim,)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(
            0)  # torch.Size([51, 30]) (num_emb, embedding_dim)

        # assign sinusoidal positional embedding to correct positions 
        emb[:, emb_c2 % 2 == 0] = torch.sin(emb[:, emb_c2 % 2 == 0])
        emb[:, emb_c2 % 2 == 1] = torch.cos(emb[:, emb_c2 % 2 == 1])

        # emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1) # (num_emb, half_dim*2)

        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0

        return emb

    def forward(self, input):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input.size()
        max_pos = self.padding_idx + 1 + seq_len
        device = input.get_device()
        if device not in self.weights or max_pos > self.weights[device].size(0):
            # recompute/expand embeddings if needed
            self.weights[device] = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights[device] = self.weights[device].type_as(self._float_tensor)
        positions = make_positions(input, self.padding_idx, self.left_pad)
        return self.weights[device].index_select(0, positions.contiguous().view(-1)).view(bsz, seq_len, -1).detach()

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number
