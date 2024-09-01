import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import sys
import numpy as np


# Code adapted from the fairseq repo.

class MultiheadAttention(nn.Module):
    """Multi-headed attention.
    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, attn_dropout=0.,
                 bias=True, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.embed_dim = embed_dim  # 30
        self.num_heads = num_heads  # 5
        self.attn_dropout = attn_dropout  # 0
        self.head_dim = embed_dim // num_heads  # 30 // 6 =5
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        # assert（断言）用于判断一个表达式，在表达式条件为 false 的时候触发异常。
        self.scaling = self.head_dim ** -0.5  # ** ：乘方（指数）根号dk 收缩因子

        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))  # 使得in_proj_weight变得可优化
        self.register_parameter('in_proj_bias', None)
        # in_proj_bias 的意思就是一开始的线性变换的偏置。
        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn  # FALSE

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value, attn_mask=None,counterfactual_attention_type=None):
        """Input shape: Time x Batch x Channel
        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()  # false  # data_ptr():返回tensor第一个元素的地址
        kv_same = key.data_ptr() == value.data_ptr()  # false

        # key, value：500*8*30
        tgt_len, bsz, embed_dim = query.size()  # 50，8，30
        # 以下断言都是为了确认参数合法
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        aved_state = None

        if qkv_same:  # false
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:  # false
            # encoder-decoder attention
            q = self.in_proj_q(query)

            if key is None:
                assert value is None
                k = v = None
            else:
                k, v = self.in_proj_kv(key)
        else:  # 这里
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q = q * self.scaling  # 根号dk

        if self.bias_k is not None:  # 不进
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)  # 32*5,50, 30/5
        # contiguous() 返回开辟了一块新的存放q的连续内存，并且改变该值会改变原值
        # head_dim = embed_dim(30) // num_heads(5)
        # view(50, 8*5, 6)  -> 40*50*6
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)  # 32*5,147, 30/5
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)  # 32*5,147, 30/5

        src_len = k.size(1)  # 50

        if self.add_zero_attn:  # 没进
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))  # 32*5,l1,l2
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:  # 进
            try:
                # attn_weights += attn_mask.unsqueeze(0)  # attn_mask：50*500
                attn_mask = attn_mask.unsqueeze(1).unsqueeze(0).repeat(self.num_heads, 1, query.shape[0], 1)
                attn_weights = attn_weights.view(self.num_heads, query.shape[1], query.shape[0], -1).masked_fill(
                    attn_mask.cuda(), -np.inf)
                attn_weights = attn_weights.view(self.num_heads * query.shape[1], query.shape[0], -1)
            except:
                print(attn_weights.shape)
                print(attn_mask.unsqueeze(0).shape)
                assert False

        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)

        if counterfactual_attention_type=="random":
            non_zero_mask = attn_weights != 0

            # 生成随机值，均匀分布在[0, 1] 范围内
            random_values = torch.rand(attn_weights.size()).cuda()

            # 创建一个新的 attn_weights tensor，用随机值替代非零元素
            new_attn_weights = torch.where(non_zero_mask, random_values, attn_weights)

            # 计算new_attn_weights在dim=2维度上的L1范数
            l1_norm = torch.norm(new_attn_weights, p=1, dim=2, keepdim=True)

            # 使用L1范数来归一化new_attn_weights在dim=2维度上
            normalized_new_attn_weights = new_attn_weights / l1_norm

            # 更新 attn_weights 为归一化后的new_attn_weights
            attn_weights = normalized_new_attn_weights
        elif counterfactual_attention_type=="shuffle":
            #bs wise shuffle
            reshaped_tensor = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            # 创建一个随机的排列索引
            permutation_indices = torch.randperm(bsz).cuda()
            # 使用排列后的索引在第一个维度上进行 shuffle
            shuffled_tensor = reshaped_tensor[permutation_indices]
            # 将张量变回 (bs*head, size0, size1) 形状
            shuffled_tensor = shuffled_tensor.view(bsz*self.num_heads, tgt_len, src_len)
            # shuffled_tensor 现在包含了经过 shuffle 的张量，形状为 (bs*head, size0, size1)
            attn_weights=shuffled_tensor

        elif counterfactual_attention_type=="reversed":
            nonzero_mask = attn_weights != 0
            # 创建一个新的张量，初始化为零
            reciprocal_attn_weight = torch.zeros_like(attn_weights)
            # 对于非零元素，计算取倒数并赋值
            reciprocal_attn_weight[nonzero_mask] = 1.0 / attn_weights[nonzero_mask]
            # 2. 计算在 size2 维度上的和，但不包括零元素
            sum_along_size2 = torch.sum(
                torch.where(nonzero_mask, reciprocal_attn_weight, torch.zeros_like(attn_weights)), dim=2, keepdim=True)
            # 3. 归一化，确保零元素保持零
            normalized_reciprocal_attn_weight = torch.where(nonzero_mask, reciprocal_attn_weight / sum_along_size2,
                                                            attn_weights)
            attn_weights=normalized_reciprocal_attn_weight
        elif counterfactual_attention_type=="uniform":
            non_zero_mask = attn_weights != 0
            # 计算每个子序列的平均值
            non_zero_sum = attn_weights.sum(dim=2)  # 在 size1 维度上求和
            non_zero_count = non_zero_mask.sum(dim=2)  # 统计非零元素的数量
            non_zero_mean = non_zero_sum / non_zero_count
            # 创建一个新的 attn_weights tensor，用平均值替代非零元素
            new_attn_weights = torch.where(non_zero_mask, non_zero_mean.unsqueeze(2), attn_weights)
            attn_weights=new_attn_weights
        elif counterfactual_attention_type is None:
            pass
        else:
            print(counterfactual_attention_type)

        # attn_weights = F.relu(attn_weights)
        # attn_weights = attn_weights / torch.max(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)  # 32*5,50,30/5 attn_weights: 40*50*500  v:40*500*6
        # bmm: bnm 和 bmp 得到 bnp   -> 40*50*6   具体运算看：https://blog.csdn.net/weixin_45573525/article/details/108143684
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)  # 拼回去，50，32，30
        attn = self.out_proj(attn)

        # average attention weights over heads
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)  # 32,5,50,147
        attn_weights = attn_weights.sum(dim=1) / self.num_heads  # 对注意力分数的5个头求均值
        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query, **kwargs):
        # 以 q = self.in_proj_q(query)为例
        return self._in_proj(query, end=self.embed_dim, **kwargs)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None, **kwargs):  # input: query ,end: 30
        # 以 self._in_proj(query, end=self.embed_dim, **kwargs) 为例
        weight = kwargs.get('weight', self.in_proj_weight)  # 30*30
        bias = kwargs.get('bias', self.in_proj_bias)  # shape=90, size:1
        weight = weight[start:end, :]  # 30*30
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)

# if __name__=="__main__":
    # attn_weights=torch.arange(54,dtype=torch.double).view(6,3,3)
    # bs = 3
    # head = 2
    # size0 = 3
    # size1 = 3
    # attn_weights[1,1,1]=0

    # # Uniform Attention.
    # # 假设 attn_weights 是一个形状为 (bs*head, size0, size1) 的 PyTorch tensor
    # # 找到 attn_weights 中非零元素的位置
    # non_zero_mask = attn_weights != 0
    # # 计算每个子序列的平均值
    # non_zero_sum = attn_weights.sum(dim=2)  # 在 size1 维度上求和
    # non_zero_count = non_zero_mask.sum(dim=2)  # 统计非零元素的数量
    # non_zero_mean = non_zero_sum / non_zero_count
    # # 创建一个新的 attn_weights tensor，用平均值替代非零元素
    # new_attn_weights = torch.where(non_zero_mask, non_zero_mean.unsqueeze(2), attn_weights)
    # print(new_attn_weights)

    # Random Attention.
    # non_zero_mask = attn_weights != 0
    # # 从 U(0, 2) 均匀分布中采样随机值
    # random_values = torch.rand(attn_weights.size(),dtype=torch.double) * 2.0
    # # 创建一个新的 attn_weights tensor，用随机值替代非零元素
    # new_attn_weights = torch.where(non_zero_mask, random_values, attn_weights)
    # print(new_attn_weights)


    # Reversed Attention.
    # 假设 attn_weights 是一个形状为 (bs*head, size0, size1) 的 PyTorch tensor
    # 找到 attn_weights 中非零元素的位置
    # 假设 attn_weights 是一个形状为 (bs*head, size0, size1) 的 PyTorch tensor
    # 找到 attn_weights 中非零元素的位置
    # non_zero_mask = attn_weights != 0
    # # 计算 size1 维度上的最大值
    # max_value_size1, _ = attn_weights.max(dim=2, keepdim=True)
    # # 创建一个新的 attn_weights tensor，只对非零元素进行操作，保持零元素不变
    # new_attn_weights = torch.where(non_zero_mask, max_value_size1 - attn_weights, attn_weights)
    # # 现在 new_attn_weights 包含了在 size1 维度上将所有非零元素替换为 size1 维度上的最大值减去原有的 attn_weights 的结果，而零元素保持不变
    # print(new_attn_weights)

    # Shuffle Attention.
    # 创建一个随机的排列索引
    # 将张量变形为 (bs, head, size0, size1)
    # reshaped_tensor = attn_weights.view(bs, head, size0, size1)
    # # 创建一个随机的排列索引
    # permutation_indices = torch.randperm(bs)
    # # 使用排列后的索引在第一个维度上进行 shuffle
    # shuffled_tensor = reshaped_tensor[permutation_indices]
    # # 将张量变回 (bs*head, size0, size1) 形状
    # shuffled_tensor = shuffled_tensor.view(bs * head, size0, size1)
    # # shuffled_tensor 现在包含了经过 shuffle 的张量，形状为 (bs*head, size0, size1)
    # print(attn_weights)
    # print(shuffled_tensor)