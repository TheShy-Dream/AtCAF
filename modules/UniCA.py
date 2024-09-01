import os

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from modules.encoders import LanguageEmbeddingLayer, CPC, MMILB, RNNEncoder, SubNet, FusionTrans,Encoder,SelfAttention,CrossAttention,FinalFusionSelfAttention,SumFusion,ConcatFusion
from modules.position_embedding import SinusoidalPositionalEmbedding
from modules.transformer import TransformerEncoder

#单模态去偏器
class UnimodalDebiasModule(nn.Module):
    def init_confounder_dictionary(self):
        root_dir = os.path.join(r".", self.hp.npy_path)
        if self.hp.use_kmean == True:
            if self.modal_type=="audio":
                self.acoustic_center = np.load(os.path.join(root_dir, f"kmeans_{self.hp.dataset}-{self.hp.audio_kmean_size}_audio.npy"))
                self.acoustic_center= np.expand_dims(self.acoustic_center, axis=0)
                self.acoustic_center = nn.Parameter(torch.from_numpy(self.acoustic_center).cuda().requires_grad_())  # [kmean_size, d_ain]
                print(f"Initialize the {self.modal_type} confounder dictionary with the Kmean weights!")
            elif self.modal_type=="vision":
                self.visual_center = np.load(os.path.join(root_dir, f"kmeans_{self.hp.dataset}-{self.hp.vision_kmean_size}_visual.npy"))
                self.visual_center= np.expand_dims(self.visual_center, axis=0)
                self.visual_center = nn.Parameter(torch.from_numpy(self.visual_center).cuda().requires_grad_())  # [kmean_size,d_vin]
                print(f"Initialize the {self.modal_type} confounder dictionary with the Kmean weights!")
            elif self.modal_type=="text":
                self.text_center = np.load(os.path.join(root_dir, f"kmeans_{self.hp.dataset}-{self.hp.text_kmean_size}_text_{self.hp.npy_selection}.npy"))
                self.text_center= np.expand_dims(self.text_center, axis=0)
                self.text_center = nn.Parameter(torch.from_numpy(self.text_center).cuda().requires_grad_())  # [kmean_size,d_tin]
                print(f"Initialize the {self.modal_type} confounder dictionary with the Kmean weights!")
        else:
            if self.modal_type=="audio":
                self.acoustic_center = torch.rand([1, self.hp.kmean_size, self.hp.d_ain], dtype=torch.float32) / 100
                self.acoustic_center = nn.Parameter(self.acoustic_center.requires_grad_())
            elif self.modal_type=="vision":
                self.visual_center = torch.rand([1, self.hp.kmean_size, self.hp.d_vin], dtype=torch.float32) / 100
                self.visual_center = nn.Parameter(self.visual_center.requires_grad_())
            elif self.modal_type=="text":
                self.text_center = torch.rand([1, self.hp.kmean_size, self.hp.d_tin], dtype=torch.float32) / 100
                self.text_center = nn.Parameter(self.text_center.requires_grad_())


    def gen_mask(self, a, length=None):
        if length is None:
            msk_tmp = torch.sum(a, dim=-1)
            # 特征全为0的时刻加mask
            mask = (msk_tmp == 0)
            return mask
        else:
            b = a.shape[0]
            l = a.shape[1]
            msk = torch.ones((b, l))
            x = []
            y = []
            for i in range(b):
                for j in range(length[i], l):
                    x.append(i)
                    y.append(j)
            msk[x, y] = 0
            return (msk == 0)

    def __init__(self,hp,modal_type):
        super(UnimodalDebiasModule, self).__init__()
        self.modal_type=modal_type
        self.hp=hp


        if self.modal_type=="vision":
            self.proj = nn.Conv1d(hp.d_vin, hp.model_dim_self, kernel_size=1, padding=0, bias=False)
            self.proj_dict=nn.Conv1d(hp.d_vin, hp.model_dim_self, kernel_size=1, padding=0, bias=False)
            self.IS_vision_encoder = SelfAttention(hp, d_in=hp.d_vin, d_model=hp.model_dim_self,
                                                    nhead=hp.num_heads_self,
                                                    dim_feedforward=4 * hp.model_dim_self, dropout=hp.attn_dropout_debias,
                                                    num_layers=hp.vision_debias_layers)
            self.CS_vision_encoder= CrossAttention(hp,d_modal1=hp.d_vin,d_modal2=hp.d_vin,d_model=hp.model_dim_cross,nhead=hp.num_heads_self,
                                          dim_feedforward=4*hp.model_dim_cross,dropout=hp.attn_dropout_debias,num_layers=hp.vision_debias_layers)
        if self.modal_type=="audio":
            self.proj = nn.Conv1d(hp.d_ain, hp.model_dim_self,kernel_size=1, padding=0, bias=False)
            self.proj_dict = nn.Conv1d(hp.d_ain, hp.model_dim_self, kernel_size=1, padding=0, bias=False)
            self.IS_audio_encoder = SelfAttention(hp,d_in=hp.d_ain, d_model=hp.model_dim_self, nhead=hp.num_heads_self,
                                     dim_feedforward=4 * hp.model_dim_self,dropout=hp.attn_dropout_debias,
                                     num_layers=hp.audio_debias_layers)

            self.CS_audio_encoder=CrossAttention(hp, d_modal1=hp.d_ain, d_modal2=hp.d_ain, d_model=hp.model_dim_cross,
                           nhead=hp.num_heads_self,
                           dim_feedforward=4 * hp.model_dim_cross, dropout=hp.attn_dropout_debias,
                           num_layers=hp.audio_debias_layers)

        if self.modal_type=="text":
            self.proj = nn.Conv1d(hp.d_tin, hp.model_dim_self, kernel_size=1, padding=0, bias=False)
            self.proj_dict = nn.Conv1d(hp.d_tin, hp.model_dim_self, kernel_size=1, padding=0, bias=False)
            self.IS_text_encoder=SelfAttention(hp,d_in=hp.d_tin, d_model=hp.model_dim_self, nhead=hp.num_heads_self,
                                     dim_feedforward=4 * hp.model_dim_self,dropout=hp.attn_dropout_debias ,
                                     num_layers=hp.text_debias_layers)

            self.CS_text_encoder=CrossAttention(hp, d_modal1=hp.d_tin, d_modal2=hp.d_tin, d_model=hp.model_dim_cross,
                           nhead=hp.num_heads_self,
                           dim_feedforward=4 * hp.model_dim_cross, dropout=hp.attn_dropout_debias,
                           num_layers=hp.text_debias_layers)

        self.init_confounder_dictionary()

    def forward(self,input_modal,modal_mask): #进行L层自注意力和跨注意力的堆叠,
        bs=input_modal.size(1)
        if self.modal_type=="audio":
            audio_local=self.IS_audio_encoder(input_modal,maskA=modal_mask)#[seq_len,bs,a_dim]
            audio_confounder_dict=self.acoustic_center.expand(bs, -1, -1).permute(1,0,2)#[seq_len,bs,dim]
            audio_global=self.CS_audio_encoder(input_modal,audio_confounder_dict,Amask=modal_mask,whether_add_position=False)
            return torch.cat([audio_local,audio_global],dim=2) #[seq_len,bs,dim]
        elif self.modal_type=="vision":
            vision_local = self.IS_vision_encoder(input_modal, maskV=modal_mask)#[seq_len,bs,d_tin]
            vision_confounder_dict = self.visual_center.expand(bs, -1, -1).permute(1, 0, 2)  # [seq_len,bs,dim]
            vision_global = self.CS_vision_encoder(input_modal, vision_confounder_dict, Vmask=modal_mask,whether_add_position=False)
            return torch.cat([vision_local, vision_global], dim=2)  # [seq_len,bs,dim]
        elif self.modal_type=="text":
            text_local = self.IS_text_encoder(input_modal, maskT=modal_mask)#[seq_len,bs,d_tin]
            text_confounder_dict = self.text_center.expand(bs, -1, -1).permute(1, 0, 2)  # [seq_len,bs,dim]
            text_global = self.CS_text_encoder(input_modal, text_confounder_dict, Tmask=modal_mask,whether_add_position=False)
            return torch.cat([text_local, text_global], dim=2)  # [seq_len,bs,dim]
        else:
            print("only three modal_types: audio,vision,text")


#跨模态反事实注意力
class CounterFactualAttention(nn.Module):
    def __init__(self, hp, d_modal1,d_modal2, d_model, nhead, dim_feedforward, dropout, num_layers=6):
        super(CounterFactualAttention, self).__init__()
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

    def forward(self, input_modal1, input_modal2, Tmask=None, Amask=None, Vmask=None, whether_add_position=True,counterfactual_attention_type=None): #是否加上位置编码,是否使用跨模态注意力
            """
            传入格式：(seq_len, batch_size,emb_size)
            t: torch.Size([50, 32, 768])
            a: torch.Size([134, 32, 5])
            """
            modal1 = self.proj_modal1(input_modal1.permute(1, 2, 0))
            modal2 = self.proj_modal2(input_modal2.permute(1, 2, 0))
            modal1 = modal1.permute(2, 0, 1)
            modal2 = modal2.permute(2, 0, 1)
            if self.hp.d_tin == self.d_modal1 and self.hp.d_ain == self.d_modal2:
                encoded = self.net(modal1, modal2, modal2, Tmask, Amask, whether_add_position=whether_add_position,counterfactual_attention_type=counterfactual_attention_type)
            elif self.hp.d_tin == self.d_modal1 and self.hp.d_vin == self.d_modal2:
                encoded = self.net(modal1, modal2, modal2, Tmask, Vmask, whether_add_position=whether_add_position,counterfactual_attention_type=counterfactual_attention_type)
            elif self.hp.d_ain == self.d_modal1 and self.hp.d_tin == self.d_modal2:
                encoded = self.net(modal1, modal2, modal2, Amask, Tmask, whether_add_position=whether_add_position,counterfactual_attention_type=counterfactual_attention_type)
            elif self.hp.d_ain == self.d_modal1 and self.hp.d_vin == self.d_modal2:
                encoded = self.net(modal1, modal2, modal2, Amask, Vmask, whether_add_position=whether_add_position,counterfactual_attention_type=counterfactual_attention_type)
            elif self.hp.d_vin == self.d_modal1 and self.hp.d_tin == self.d_modal2:
                encoded = self.net(modal1, modal2, modal2, Vmask, Tmask, whether_add_position=whether_add_position,counterfactual_attention_type=counterfactual_attention_type)
            elif self.hp.d_vin == self.d_modal1 and self.hp.d_ain == self.d_modal2:
                encoded = self.net(modal1, modal2, modal2, Vmask, Amask, whether_add_position=whether_add_position,counterfactual_attention_type=counterfactual_attention_type)
            output = self.output_linear(F.relu(self.linear(encoded)))
            return output



