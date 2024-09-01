import torch
import torch.nn.functional as F
import time
import math
import copy
from random import randint
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from transformers import XLNetModel,XLNetConfig,BertModel,BertConfig
from .transformer import TransformerEncoder,TransformerEncoderLayer
from torch.nn import TransformerEncoder as TransEncoder
from torch.nn import TransformerEncoderLayer as TransEncoderLayer

import numpy as np


def add_noise(x, intens=1e-7):
    return x + torch.rand(x.size()) * intens


class LanguageEmbeddingLayer(nn.Module):
    """Embed input text with "glove" or "Bert"
    """

    def __init__(self, hp):
        super(LanguageEmbeddingLayer, self).__init__()
        bertconfig = BertConfig.from_pretrained('/home/cjl/code/cjl/MMIM/bert-base-uncased/', output_hidden_states=True,output_attentions=True)
        self.bertmodel = BertModel.from_pretrained('/home/cjl/code/cjl/MMIM/bert-base-uncased/', config=bertconfig)

    def forward(self, sentences, bert_sent, bert_sent_type, bert_sent_mask):
        bert_output = self.bertmodel(input_ids=bert_sent,
                                     attention_mask=bert_sent_mask,
                                     token_type_ids=bert_sent_type)
        '''
        tensor_to_save = bert_output[-1]
        npy_stacked = np.stack([tensor.detach().cpu().numpy() for tensor in tensor_to_save], axis=0)
        np.save('output_array.npy', npy_stacked)
        '''
        bert_output= bert_output[0]
        return bert_output# return head (sequence representation)


class SubNet(nn.Module):  # 融合模块  可以更换！！！
    '''
    The subnetwork that is used in TFN for video and audio in the pre-fusion stage
    '''

    def __init__(self, in_size, hidden_size, n_class, dropout, modal_name='text',output_size=None):
        self.n_class=n_class
        self.output_size=output_size
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            dropout: dropout probability
        Output:
            (return value in forward) a tensor of shape (batch_size, hidden_size)
        '''
        super(SubNet, self).__init__()
        # self.norm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, 2*hidden_size)
        self.linear_3 = nn.Linear(2*hidden_size, hidden_size)
        if output_size is not None:
            self.output_linear=nn.Linear(hidden_size,output_size)
        if n_class is not None:
            self.linear_4= nn.Linear(hidden_size, n_class)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, in_size)
        '''
        # normed = self.norm(x)
        dropped = self.drop(x)
        y_1 = torch.tanh(self.linear_1(dropped))
        y_2 = torch.tanh(self.linear_2(y_1))
        y_2 = self.drop(y_2)
        y_3 = torch.tanh(self.linear_3(y_2))
        if self.n_class is not None:
            y_4 = self.linear_4(y_3)
            return y_3, y_4
        elif self.output_size is not None:
            y_4 = self.output_linear(y_3)
            return y_4

class SumFusion(nn.Module):
    def __init__(self, input_dim=512, output_dim=100,dropout=0.1,n_class=1):
        super(SumFusion, self).__init__()
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(input_dim, output_dim)
        self.linear_2 = nn.Linear(output_dim, 2 * output_dim)
        self.linear_3 = nn.Linear(2 * output_dim, output_dim)
        self.linear_4=nn.Linear(output_dim,n_class)

    def forward(self, x, y):
        sum_result=x+y
        dropped = self.drop(sum_result)
        y_1 = torch.tanh(self.linear_1(dropped))
        y_2 = torch.tanh(self.linear_2(y_1))
        y_2 = self.drop(y_2)
        y_3 = torch.tanh(self.linear_3(y_2))
        preds = self.linear_4(y_3)
        return y_3,preds

class ConcatFusion(nn.Module):
    def __init__(self, input_dim=1024, output_dim=100,dropout=0.1,n_class=1):
        super(ConcatFusion, self).__init__()
        self.drop = nn.Dropout(p=dropout)
        self.linear_1=nn.Linear(input_dim*2,output_dim)
        self.linear_2 = nn.Linear(output_dim, 2 * output_dim)
        self.linear_3 = nn.Linear(2 * output_dim, output_dim)
        self.fc_out = nn.Linear(input_dim, n_class)

    def forward(self, x, y):
        modal_cat = torch.cat((x, y), dim=1)
        dropped = self.drop(modal_cat)
        y_1 = torch.tanh(self.linear_1(dropped))
        y_2 = torch.tanh(self.linear_2(y_1))
        y_2 = self.drop(y_2)
        y_3 = torch.tanh(self.linear_3(y_2))
        preds = self.linear_4(y_3)
        return y_3,preds


class FusionTrans(nn.Module):
    def __init__(self, hp, n_class):
        super(FusionTrans, self).__init__()
        self.hp = hp
        self.d_l, self.d_a, self.d_v = 30, 30, 30
        self.vonly = hp.vonly
        self.aonly = hp.aonly
        self.lonly = hp.lonly
        self.num_heads = hp.num_heads
        self.layers = hp.layers
        self.attn_dropout = hp.attn_dropout
        self.attn_dropout_a = hp.attn_dropout_a
        self.attn_dropout_v = hp.attn_dropout_v
        self.relu_dropout = hp.relu_dropout
        self.res_dropout = hp.res_dropout
        self.out_dropout = hp.out_dropout
        self.d_prjh = hp.d_prjh
        self.embed_dropout = hp.embed_dropout
        self.attn_mask = hp.attn_mask
        self.n_lv = hp.n_tv
        self.n_la = hp.n_ta

        combined_dim = 2 * self.d_l  # assuming d_l == d_a == d_v == 30

        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(hp.d_tin, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(hp.d_ain, self.d_a, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(hp.d_vin, self.d_v, kernel_size=1, padding=0, bias=False)

        #  Crossmodal Transformer（Attentions在里面）  CM
        self.trans_l_with_a = self.get_network(self_type='la')
        self.trans_l_with_v = self.get_network(self_type='lv')
        self.trans_l_with_al = self.get_network(self_type='lla')
        self.trans_l_with_vl = self.get_network(self_type='llv')

        # Projection layers
        self.proj1 = nn.Linear(self.d_l, self.d_l)
        self.proj2 = nn.Linear(self.d_l, self.d_l)
        #  再加一层映射！！！  180到128的
        self.proj3 = nn.Linear(self.d_l, self.d_prjh)
        self.out_layer = nn.Linear(self.d_l, n_class)

    def get_network(self, self_type='l', layers=-1):
        # self.d_l, self.d_a, self.d_v = hyp_params.embed_dim
        if self_type in ['l', 'al', 'vl', 'lla', 'llv']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 2 * self.d_l, self.attn_dropout
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,  # 30
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    def forward(self, t, a, v,t_mask=None,a_mask=None,v_mask=None):
        """
        传入格式：(seq_len, batch_size,emb_size)
        t: torch.Size([50, 32, 768])
        a: torch.Size([134, 32, 5])
        v: torch.Size([161, 32, 20])
        """
        text = self.proj_l(t.permute(1, 2, 0))  # torch.Size([32, 30, 50])   传入、得到 batch_size, n_feature,seq_len
        acoustic = self.proj_a(a.permute(1, 2, 0))  # torch.Size([32, 30, 134])
        visual = self.proj_v(v.permute(1, 2, 0))  # torch.Size([32, 30, 161])

        text = text.permute(2, 0, 1)  # seq_len,batch-size,n_feature
        acoustic = acoustic.permute(2, 0, 1)
        visual = visual.permute(2, 0, 1)

        #  Crossmodal Transformer
        # (V,A) --> L
        l_with_a = self.trans_l_with_a(text, acoustic, acoustic)  # Dimension (L, N, d_l)
        l_with_v = self.trans_l_with_v(text, visual, visual)  # Dimension (L, N, d_l)

        l2a = None
        l2v = None
        for i in range(max(self.n_la, self.n_lv)):
            if i < self.n_la:
                l_with_aa = self.trans_l_with_a(text, l_with_a, l_with_a)  # Dimension (L, N, d_l)
                l_with_av = self.trans_l_with_a(text, l_with_v, l_with_v)  # Dimension (L, N, d_l)
                l2a = torch.mean((l_with_aa + l_with_av),dim=0) #取序列中的最后一个元素，可以改pooling
            if i < self.n_lv:
                l_with_vv = self.trans_l_with_v(text, l_with_v, l_with_v)
                l_with_va = self.trans_l_with_v(text, l_with_a, l_with_a)
                l2v = torch.mean((l_with_vv + l_with_va),dim=0) #取序列中的最后一个元素，可以改pooling
            l_with_a = l2a
            l_with_v = l2v

        if min(self.n_la, self.n_lv) > 0:
            last_hs = l2a + l2v
        elif self.n_la == 0 and self.n_lv != 0:
            last_hs = l2v
        elif self.n_la != 0 and self.n_lv == 0:
            last_hs = l2a
        else:
            last_hs = torch.mean((l_with_a + l_with_v),dim=0) #取序列中的最后一个元素，可以改pooling

        # A residual block
        last_hs_proj = self.proj2(
            F.dropout(
                F.relu(self.proj1(last_hs)),
                p=self.out_dropout,
                training=self.training
            )
        )  # torch.Size([32, 180])
        last_hs_proj += last_hs  # last_hs，last_hs_proj = torch.Size([32, 180])

        last_hs = self.proj3(last_hs)  # torch.Size([32, 128])

        output = self.out_layer(last_hs_proj)  # torch.Size([32, 1])
        return last_hs, output #torch.Size([32, 180]),torch.Size([32, 128])


class CrossAttention(nn.Module):
    def __init__(self, hp, d_modal1,d_modal2, d_model, nhead, dim_feedforward, dropout, num_layers=6):
        super(CrossAttention, self).__init__()
        self.hp = hp
        self.d_modal1 = d_modal1
        self.d_modal2=d_modal2
        self.num_heads = nhead
        self.d_model = d_model
        self.proj_modal1 = nn.Conv1d(self.d_modal1, self.d_model, kernel_size=1, padding=0, bias=False)
        self.proj_modal2 = nn.Conv1d(self.d_modal2, self.d_model, kernel_size=1, padding=0, bias=False)
        self.layers = num_layers
        self.linear = nn.Linear(d_model, dim_feedforward)
        self.output_linear = nn.Linear(dim_feedforward, self.d_model)

        self.attn_dropout = dropout
        self.relu_dropout = self.hp.relu_dropout
        self.res_dropout = self.hp.res_dropout
        self.embed_dropout = self.hp.embed_dropout
        self.attn_mask = self.hp.attn_mask

        self.net = self.get_network()

    def get_network(self, layers=-1):
        return TransformerEncoder(embed_dim=self.d_model,  # 30
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=self.attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    def forward(self, input_modal1,input_modal2,Tmask=None,Amask=None,Vmask=None,whether_add_position=True,counterfactual_attention_type=None): #这里下面都不用给counterfactual attention type
        """
        传入格式：(seq_len, batch_size,emb_size)
        t: torch.Size([50, 32, 768])
        a: torch.Size([134, 32, 5])
        """
        modal1 = self.proj_modal1(input_modal1.permute(1, 2, 0))
        modal2 = self.proj_modal2(input_modal2.permute(1, 2, 0))
        modal1 = modal1.permute(2, 0, 1)
        modal2 = modal2.permute(2,0,1)
        if self.hp.d_tin==self.d_modal1 and self.hp.d_ain==self.d_modal2:
            encoded= self.net(modal1, modal2, modal2,Tmask,Amask,whether_add_position=whether_add_position,counterfactual_attention_type=counterfactual_attention_type)
        elif self.hp.d_tin==self.d_modal1 and self.hp.d_vin==self.d_modal2:
            encoded = self.net(modal1, modal2, modal2, Tmask, Vmask,whether_add_position=whether_add_position,counterfactual_attention_type=counterfactual_attention_type)
        elif self.hp.d_ain==self.d_modal1 and self.hp.d_tin==self.d_modal2:
            encoded = self.net(modal1, modal2, modal2, Amask, Tmask, whether_add_position=whether_add_position,counterfactual_attention_type=counterfactual_attention_type)
        elif self.hp.d_ain==self.d_modal1 and self.hp.d_vin==self.d_modal2:
            encoded = self.net(modal1, modal2, modal2, Amask, Vmask, whether_add_position=whether_add_position,counterfactual_attention_type=counterfactual_attention_type)
        elif self.hp.d_vin==self.d_modal1 and self.hp.d_tin==self.d_modal2:
            encoded = self.net(modal1, modal2, modal2, Vmask, Tmask, whether_add_position=whether_add_position,counterfactual_attention_type=counterfactual_attention_type)
        elif self.hp.d_vin == self.d_modal1 and self.hp.d_ain == self.d_modal2:
            encoded = self.net(modal1, modal2, modal2, Vmask, Amask, whether_add_position=whether_add_position,counterfactual_attention_type=counterfactual_attention_type)

        #专门给因果注意力用的
        elif self.d_modal1 == self.d_modal2 ==self.hp.d_tin:
            encoded = self.net(modal1, modal2, modal2, Tmask, None, whether_add_position=whether_add_position)
        elif self.d_modal1 == self.d_modal2 ==self.hp.d_ain:
            encoded = self.net(modal1, modal2, modal2, Amask, None, whether_add_position=whether_add_position)
        elif self.d_modal1 == self.d_modal2 ==self.hp.d_vin:
            encoded = self.net(modal1, modal2, modal2, Vmask,None, whether_add_position=whether_add_position)
        output = self.output_linear(F.relu(self.linear(encoded)))
        return output


###已经完成咯
class SelfAttention(nn.Module):
    def __init__(self,hp, d_in,d_model, nhead, dim_feedforward, dropout,num_layers=6):
        super(SelfAttention, self).__init__()
        self.hp=hp
        self.d_in=d_in
        self.num_heads = nhead
        self.d_model=d_model
        self.proj = nn.Conv1d(self.d_in, self.d_model, kernel_size=1, padding=0, bias=False)
        self.layers = num_layers
        self.linear = nn.Linear(d_model, dim_feedforward)
        self.output_linear=nn.Linear(dim_feedforward,d_in)
        self.attn_dropout=dropout
        self.relu_dropout=self.hp.relu_dropout
        self.res_dropout = self.hp.res_dropout
        self.embed_dropout = self.hp.embed_dropout
        self.attn_mask=self.hp.attn_mask

        self.net=self.get_network()


    def get_network(self, layers=-1):
        return TransformerEncoder(embed_dim=self.d_model,  # 30
                           num_heads=self.num_heads,
                           layers=max(self.layers, layers),
                           attn_dropout=self.attn_dropout,
                           relu_dropout=self.relu_dropout,
                           res_dropout=self.res_dropout,
                           embed_dropout=self.embed_dropout,
                           attn_mask=self.attn_mask)
    def forward(self,input,maskT=None,maskA=None,maskV=None,whether_add_position=True):
        """
        传入格式：(seq_len, batch_size,emb_size)
        t: torch.Size([50, 32, 768])
        a: torch.Size([134, 32, 5])
        v: torch.Size([161, 32, 20])
        """

        input = self.proj(input.permute(1, 2, 0))

        x=input.permute(2, 0, 1)
        if self.d_in == self.hp.d_ain:
            encoded=self.net(x,maskA,whether_add_position=whether_add_position)
        elif self.d_in==self.hp.d_vin:
            encoded = self.net(x, maskV,whether_add_position=whether_add_position)
        elif self.d_in==self.hp.d_tin:
            encoded=self.net(x,maskT,whether_add_position=whether_add_position)
        output=self.output_linear(self.linear(encoded))#放到原来的
        return output


class FinalFusionSelfAttention(nn.Module):
    def __init__(self,hp, text_in_dim,audio_in_dim,vision_in_dim,cross_ta_dim,cross_tv_dim,d_model, nhead, dim_feedforward, dropout,num_layers=6,n_class=1):
        super(FinalFusionSelfAttention, self).__init__()
        self.hp=hp
        self.text_in_dim=text_in_dim
        self.audio_in_dim=audio_in_dim
        self.vision_in_dim=vision_in_dim
        self.cross_ta_dim=cross_ta_dim
        self.cross_tv_dim=cross_tv_dim


        self.num_heads = nhead

        self.d_model=d_model

        self.proj_t_to_model = nn.Conv1d(self.text_in_dim, self.d_model, kernel_size=1, padding=0, bias=False)
        self.proj_a_to_model = nn.Conv1d(self.audio_in_dim, self.d_model, kernel_size=1, padding=0, bias=False)
        self.proj_v_to_model = nn.Conv1d(self.vision_in_dim, self.d_model, kernel_size=1, padding=0, bias=False)
        self.proj_ta_to_model = nn.Conv1d(self.cross_tv_dim, self.d_model, kernel_size=1, padding=0, bias=False)
        self.proj_tv_to_model = nn.Conv1d(self.cross_ta_dim, self.d_model, kernel_size=1, padding=0, bias=False)

        self.layers = num_layers

        self.attn_dropout=dropout
        self.relu_dropout=self.hp.relu_dropout
        self.res_dropout = self.hp.res_dropout
        self.embed_dropout = self.hp.embed_dropout
        self.attn_mask=self.hp.attn_mask
        self.n_class=n_class

        self.linear = nn.Linear(d_model, dim_feedforward)
        self.proj_linear=nn.Linear(dim_feedforward,hp.d_prjh)
        self.output_linear=nn.Linear(hp.d_prjh,self.n_class)
        # # Projection layers
        # self.proj1 = nn.Linear(self.d_l, self.d_l)
        # self.proj2 = nn.Linear(self.d_l, self.d_l)
        # #  再加一层映射！！！  180到128的
        # self.proj3 = nn.Linear(self.d_l, self.d_prjh)

        self.net=self.get_network()




    def get_network(self, layers=-1):
        return TransformerEncoder(embed_dim=self.d_model,  # 30
                           num_heads=self.num_heads,
                           layers=max(self.layers, layers),
                           attn_dropout=self.attn_dropout,
                           relu_dropout=self.relu_dropout,
                           res_dropout=self.res_dropout,
                           embed_dropout=self.embed_dropout,
                           attn_mask=self.attn_mask)
    def forward(self,text,audio,vision,ta,tv):
        """
        传入格式：(seq_len, batch_size,emb_size)
        t: torch.Size([50, 32, 768])
        a: torch.Size([134, 32, 5])
        v: torch.Size([161, 32, 20])
        """
        text = self.proj_t_to_model(text.permute(1, 2, 0))
        text=text.permute(2, 0, 1)##文本

        audio= self.proj_a_to_model(audio.permute(1, 2, 0))
        audio=audio.permute(2, 0, 1)##音频

        vision = self.proj_v_to_model(vision.permute(1, 2, 0))
        vision=vision.permute(2, 0, 1)##视频

        ta = self.proj_ta_to_model(ta.permute(1, 2, 0))
        ta=ta.permute(2, 0, 1)##跨模态ta

        tv = self.proj_tv_to_model(tv.permute(1, 2, 0))
        tv=tv.permute(2, 0, 1)##跨模态tv

        final_input=torch.cat([text,audio,vision,ta,tv],dim=0)

        encoded=self.net(final_input,final_input,final_input)
        pooled_encoded=torch.mean(encoded,dim=0)#[bs,dim]
        fusion=self.proj_linear(F.relu(self.linear(pooled_encoded)))
        pred=self.output_linear(fusion)

        return fusion,pred


class CLUB(nn.Module):
    """
        Compute the Contrastive Log-ratio Upper Bound (CLUB) given a pair of inputs.
        Refer to https://arxiv.org/pdf/2006.12013.pdf and https://github.com/Linear95/CLUB/blob/f3457fc250a5773a6c476d79cda8cb07e1621313/MI_DA/MNISTModel_DANN.py#L233-254

        Args:
            hidden_size(int): embedding size
            activation(int): the activation function in the middle layer of MLP
    """

    def __init__(self, hidden_size, activation='Tanh'):
        super(CLUB, self).__init__()
        try:
            self.activation = getattr(nn, activation)
        except:
            raise ValueError("Error: CLUB activation function not found in torch library")
        self.mlp_mu = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            self.activation(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.mlp_logvar = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            self.activation(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )

    def forward(self, modal_a, modal_b, sample=False):
        """
            CLUB with random shuffle, the Q function in original paper:
                CLUB = E_p(x,y)[log q(y|x)]-E_p(x)p(y)[log q(y|x)]
            
            Args:
                modal_a (Tensor): x in above equation
                model_b (Tensor): y in above equation
        """
        mu, logvar = self.mlp_mu(modal_a), self.mlp_logvar(modal_a)  # (bs, hidden_size)
        batch_size = mu.size(0)
        pred = mu

        # pred b using a
        pred_tile = mu.unsqueeze(1).repeat(1, batch_size, 1)  # (bs, bs, emb_size)
        true_tile = pred.unsqueeze(0).repeat(batch_size, 1, 1)  # (bs, bs, emb_size)

        positive = - (mu - modal_b) ** 2 / 2. / torch.exp(logvar)
        negative = - torch.mean((true_tile - pred_tile) ** 2, dim=1) / 2. / torch.exp(logvar)

        lld = torch.mean(torch.sum(positive, -1))
        bound = torch.mean(torch.sum(positive, -1) - torch.sum(negative, -1))
        return lld, bound


class MMILB(nn.Module):  # 双模态表示 的 互信息下界
    """Compute the Modality Mutual Information Lower Bound (MMILB) given bimodal representations.
    Args:
        x_size (int): embedding size of input modality representation x
        y_size (int): embedding size of input modality representation y
        mid_activation(int): the activation function in the middle layer of MLP
        last_activation(int): the activation function in the last layer of MLP that outputs logvar
    """

    def __init__(self, x_size, y_size, mid_activation='ReLU', last_activation='Tanh'):
        super(MMILB, self).__init__()
        try:
            self.mid_activation = getattr(nn, mid_activation)
            self.last_activation = getattr(nn, last_activation)
        except:
            raise ValueError("Error: CLUB activation function not found in torch library")
        self.mlp_mu = nn.Sequential(
            nn.Linear(x_size, y_size),
            self.mid_activation(),
            nn.Linear(y_size, y_size)
        )
        self.mlp_logvar = nn.Sequential(
            nn.Linear(x_size, y_size),
            self.mid_activation(),
            nn.Linear(y_size, y_size),
        )
        self.entropy_prj = nn.Sequential(
            nn.Linear(y_size, y_size // 4),
            nn.Tanh()
        )

    def forward(self, x, y, labels=None, mem=None):
        """ Forward lld (gaussian prior) and entropy estimation, partially refers the implementation
        of https://github.com/Linear95/CLUB/blob/master/MI_DA/MNISTModel_DANN.py
            Args:
                x (Tensor): x in above equation, shape (bs, x_size)
                y (Tensor): y in above equation, shape (bs, y_size)
        """
        mu, logvar = self.mlp_mu(x), self.mlp_logvar(x)  # (bs, hidden_size)
        batch_size = mu.size(0)  # 32

        positive = -(mu - y) ** 2 / 2. / torch.exp(logvar)  # 32*16
        lld = torch.mean(torch.sum(positive, -1))  # tensor(-2.1866, grad_fn=<MeanBackward0>)

        # For Gaussian Distribution Estimation 高斯分布估计
        pos_y = neg_y = None
        H = 0.0
        sample_dict = {'pos': None, 'neg': None}

        if labels is not None:
            # store pos and neg samples
            y = self.entropy_prj(y)
            pos_y = y[labels.squeeze() > 0]
            neg_y = y[labels.squeeze() < 0]

            sample_dict['pos'] = pos_y
            sample_dict['neg'] = neg_y

            # estimate entropy
            if mem is not None and mem.get('pos', None) is not None:
                pos_history = mem['pos']
                neg_history = mem['neg']

                # Diagonal setting            
                # pos_all = torch.cat(pos_history + [pos_y], dim=0) # n_pos, emb
                # neg_all = torch.cat(neg_history + [neg_y], dim=0)
                # mu_pos = pos_all.mean(dim=0)
                # mu_neg = neg_all.mean(dim=0)

                # sigma_pos = torch.mean(pos_all ** 2, dim = 0) - mu_pos ** 2 # (embed)
                # sigma_neg = torch.mean(neg_all ** 2, dim = 0) - mu_neg ** 2 # (embed)
                # H = 0.25 * (torch.sum(torch.log(sigma_pos)) + torch.sum(torch.log(sigma_neg)))

                # compute the entire co-variance matrix
                pos_all = torch.cat(pos_history + [pos_y], dim=0)  # n_pos, emb
                neg_all = torch.cat(neg_history + [neg_y], dim=0)
                mu_pos = pos_all.mean(dim=0)
                mu_neg = neg_all.mean(dim=0)
                sigma_pos = torch.mean(torch.bmm((pos_all - mu_pos).unsqueeze(-1), (pos_all - mu_pos).unsqueeze(1)),
                                       dim=0)
                sigma_neg = torch.mean(torch.bmm((neg_all - mu_neg).unsqueeze(-1), (neg_all - mu_neg).unsqueeze(1)),
                                       dim=0)
                a = 17.0795
                H = 0.25 * (torch.logdet(sigma_pos) + torch.logdet(sigma_neg))  # 公式(8)

        return lld, sample_dict, H


class CPC(nn.Module):  # 对比预测编码（可以更换！！！）
    """
        Contrastive Predictive Coding: score computation. See https://arxiv.org/pdf/1807.03748.pdf.

        Args:
            x_size (int): embedding size of input modality representation x
            y_size (int): embedding size of input modality representation y
    """

    def __init__(self, x_size, y_size, n_layers=1, activation='Tanh'):
        # x是：t a v     y是融合后的
        super().__init__()
        self.x_size = x_size
        self.y_size = y_size
        self.layers = n_layers
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.activation = getattr(nn, activation)  # 激活层的激活函数：tanh
        if n_layers == 1:  # 进
            self.net = nn.Linear(
                in_features=y_size,
                out_features=x_size
            )
        else:
            net = []
            for i in range(n_layers):
                if i == 0:
                    net.append(nn.Linear(self.y_size, self.x_size))
                    net.append(self.activation())
                else:
                    net.append(nn.Linear(self.x_size, self.x_size))
            self.net = nn.Sequential(*net)

    def forward(self, x, y):
        """Calulate the score
            公式11
            eg： nce_t = self.cpc_zt(text, fusion)  # 3.4660
            x: torch.Size([32, 768])
            y: torch.Size([32, 128])    torch.Size([32, 180])(transformer)
        """
        # import ipdb;ipdb.set_trace()
        x_pred = self.net(y)  # bs, emb_size torch.Size([32, 768])
        # 从融合结果y生成的G(Z) 实际是hm的反向预测值  这个net是G（反向传播网络）

        # normalize to unit sphere  归一化
        x_pred = x_pred / x_pred.norm(dim=1, keepdim=True)  # G 归一化  公式10 torch.Size([32, 768])
        x = x / x.norm(dim=1, keepdim=True)  # hm 归一化  公式10  torch.Size([32, 768])

        pos = torch.sum(x * x_pred, dim=-1)  # bs shape:32   公式10: 得到 s(hm,Z)
        neg = torch.logsumexp(torch.matmul(x, x_pred.t()), dim=-1)  # bs
        nce = -(pos - neg).mean()
        return nce


class RNNEncoder(nn.Module):  # 视频和音频的特征提取   也可以换！！！
    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super().__init__()
        self.bidirectional = bidirectional

        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional,
                           batch_first=False)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear((2 if bidirectional else 1) * hidden_size, out_size)

    def forward(self, x, lengths):
        '''
        x: (batch_size, sequence_len, in_size)
        eg: self.visual_enc(visual, v_len) # torch.Size([134, 32, 5]) ,tensor:32
        '''
        lengths = lengths.to(torch.int64)  # tensor:32
        bs = x.size(0)   # bs：batch_size  134   将x的第一个size赋值给bs，但是这个不是batch_sizehhh

        packed_sequence = pack_padded_sequence(x, lengths, enforce_sorted=False)
        # 将序列送给 RNN 进行处理之前，需要采用 pack_padded_sequence 进行压缩，压缩掉无效的填充值
        _, final_states = self.rnn(packed_sequence)

        if self.bidirectional:  # 是否使用双向RNN
            h = self.dropout(torch.cat((final_states[0][0], final_states[0][1]), dim=-1))
        else:  # 这里
            h = self.dropout(final_states[0].squeeze())  # torch.Size([32, 8])
        y_1 = self.linear_1(h)
        return y_1  # 32*16

###暂时不用了，
class Encoder(nn.Module):
    def __init__(self, d_in,d_model, nhead, dim_feedforward, dropout, activation="relu", num_layers=6):
        super().__init__()
        self.d_in=d_in
        self.num_heads = nhead
        self.pe = PositionalEncoding(d_model, dropout)
        self.proj = nn.Conv1d(self.d_in, d_model, kernel_size=1, padding=0, bias=False)
        self.layer = TransEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.encoder = TransEncoder(self.layer, num_layers)
        self.linear = nn.Linear(d_model, dim_feedforward)
        self.output_linear=nn.Linear(dim_feedforward,d_in)

    def forward(self, inputs, attn_mask):
        """
        传入格式：(seq_len, batch_size,emb_size)
        t: torch.Size([50, 32, 768])
        a: torch.Size([134, 32, 5])
        v: torch.Size([161, 32, 20])
        """
        inputs_proj = self.proj(inputs.permute(1, 2, 0)) #[bs,seq_len,emb]
        inputs_proj =    inputs_proj.permute(2,0,1)
        inputs = self.pe(inputs_proj)
        #attn_mask=buffered_future_mask(inputs) if attn_mask else None
        attn_mask=attn_mask.transpose(1,0)
        attn_mask = compute_mask(attn_mask, attn_mask, self.num_heads)
        encoded = self.encoder(inputs, mask=attn_mask)
        encoded = self.linear(encoded)
        encoded=self.output_linear(F.relu(encoded))##恒等映射
        return encoded


def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float('-inf')).type_as(t)


def buffered_future_mask(tensor, tensor2=None):
    dim1 = dim2 = tensor.size(0)
    if tensor2 is not None:
        dim2 = tensor2.size(0)
    future_mask = torch.triu(fill_with_neg_inf(torch.ones(dim1, dim2)), 1+abs(dim2-dim1))
    if tensor.is_cuda:
        future_mask = future_mask.cuda()
    return future_mask[:dim1, :dim2]


class AddNorm(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, prior, after):
        return self.norm(prior + self.dropout(after))


class FeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout, activation="relu"):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = _get_activation_fn(activation)

    def forward(self, inputs):
        return self.linear2(self.dropout(self.activation(self.linear1(inputs))))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def drop_path(paths, drop_rates):
    lens = len(paths)
    drop_rates = torch.tensor(drop_rates)
    drop = torch.bernoulli(drop_rates)
    if torch.all(drop == 0):
        idx = randint(0, lens-1)
        output = paths[idx]
    else:
        output = sum([paths[i] * drop[i] for i in range(lens)]) / torch.sum(drop)
    return output


def compute_mask(mask_1, mask_2, num_heads):
    mask_1 = torch.unsqueeze(mask_1, 2)
    mask_2 = torch.unsqueeze(mask_2, 1)
    attn_mask = torch.bmm(mask_1, mask_2)
    attn_mask = attn_mask.repeat(num_heads, 1, 1)
    return attn_mask.bool()


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
