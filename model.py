import os

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from modules.encoders import LanguageEmbeddingLayer, CPC, MMILB, RNNEncoder, SubNet, FusionTrans,Encoder,SelfAttention,CrossAttention,FinalFusionSelfAttention,SumFusion,ConcatFusion
from modules.InfoNCE import InfoNCE
from modules.UniCA import UnimodalDebiasModule,CounterFactualAttention
from utils.gen_Kmeans_center import gen_npy
from transformers import BertModel, BertConfig


class AtCAF(nn.Module):
    def __init__(self, hp):
        """Construct MultiMoldal InfoMax model.
        Args: 
            hp (dict): a dict stores training and model configurations
        """
        # Base Encoders
        super().__init__()
        self.hp = hp

        self.add_va = hp.add_va
        hp.d_tout = hp.d_tin

        self.uni_text_enc = LanguageEmbeddingLayer(hp)  # BERT Encoder
        self.uni_visual_enc = RNNEncoder(  # 视频特征提取
            in_size=hp.d_vin,
            hidden_size=hp.d_vh,
            out_size=hp.d_vout,
            num_layers=hp.n_layer,
            dropout=hp.dropout_v if hp.n_layer > 1 else 0.0,
            bidirectional=hp.bidirectional
        )
        self.uni_acoustic_enc = RNNEncoder(  # 音频特征提取
            in_size=hp.d_ain,
            hidden_size=hp.d_ah,
            out_size=hp.d_aout,
            num_layers=hp.n_layer,
            dropout=hp.dropout_a if hp.n_layer > 1 else 0.0,
            bidirectional=hp.bidirectional
        )

        # For MI maximization   互信息最大化
        # Modality Mutual Information Lower Bound（MMILB）
        self.mi_tv = MMILB(
            x_size=hp.d_tout,
            y_size=hp.d_vout,
            mid_activation=hp.mmilb_mid_activation,
            last_activation=hp.mmilb_last_activation
        )

        self.mi_ta = MMILB(
            x_size=hp.d_tout,
            y_size=hp.d_aout,
            mid_activation=hp.mmilb_mid_activation,
            last_activation=hp.mmilb_last_activation
        )

        if hp.add_va:  # 一般是tv和ta   若va也要MMILB
            self.mi_va = MMILB(
                x_size=hp.d_vout,
                y_size=hp.d_aout,
                mid_activation=hp.mmilb_mid_activation,
                last_activation=hp.mmilb_last_activation
            )



        # CPC MI bound   d_prjh是什么？？？
        self.cpc_zt = CPC(
            x_size=hp.d_tout,  # to be predicted  各个模态特征提取后得到的维度
            y_size=hp.d_prjh,
            n_layers=hp.cpc_layers,
            activation=hp.cpc_activation
        )
        self.cpc_zv = CPC(
            x_size=hp.d_vout,
            y_size=hp.d_prjh,
            n_layers=hp.cpc_layers,
            activation=hp.cpc_activation
        )
        self.cpc_za = CPC(
            x_size=hp.d_aout,
            y_size=hp.d_prjh,
            n_layers=hp.cpc_layers,
            activation=hp.cpc_activation
        )

        if hp.whether_debias_unimodal:
            #前门调整器 modal_type in ['audio','text','vision']
            if hp.whether_debias_audio:
                self.audio_mediator=UnimodalDebiasModule(hp,modal_type="audio")
                self.audio_mlp = SubNet(in_size=hp.d_ain + hp.model_dim_cross, hidden_size=hp.audio_mlp_hidden_size,
                                    n_class=None, dropout=hp.dropout_prj, output_size=hp.d_ain)  # [bs,seq_len,d_ain]
            else:
                self.uni_audio_encoder = SelfAttention(hp, d_in=hp.d_ain, d_model=hp.model_dim_self,
                                                       nhead=hp.num_heads_self,
                                                       dim_feedforward=4 * hp.model_dim_self,
                                                       dropout=hp.attn_dropout_self,
                                                       num_layers=hp.num_layers_self)
            if hp.whether_debias_text:
                self.text_mediator=UnimodalDebiasModule(hp,modal_type="text")
                self.text_mlp = SubNet(in_size=hp.d_tin + hp.model_dim_cross, hidden_size=hp.text_mlp_hidden_size,
                                       n_class=None, dropout=hp.dropout_prj, output_size=hp.d_tin)  # [bs,seq_len,d_tin]
            else:
                pass

            if hp.whether_debias_vision:
                self.vision_mediator=UnimodalDebiasModule(hp,modal_type="vision")
                self.vision_mlp = SubNet(in_size=hp.d_vin + hp.model_dim_cross, hidden_size=hp.vision_mlp_hidden_size,
                                     n_class=None, dropout=hp.dropout_prj, output_size=hp.d_vin)  # [bs,seq_len,d_vin]
            else:
                self.uni_vision_encoder = SelfAttention(hp, d_in=hp.d_vin, d_model=hp.model_dim_self,
                                                        nhead=hp.num_heads_self,
                                                        dim_feedforward=4 * hp.model_dim_self,
                                                        dropout=hp.attn_dropout_self,
                                                        num_layers=hp.num_layers_self)

        else:
            self.uni_audio_encoder = SelfAttention(hp, d_in=hp.d_ain, d_model=hp.model_dim_self,
                                                   nhead=hp.num_heads_self,
                                                   dim_feedforward=4 * hp.model_dim_self, dropout=hp.attn_dropout_self,
                                                   num_layers=hp.num_layers_self)
            self.uni_vision_encoder = SelfAttention(hp, d_in=hp.d_vin, d_model=hp.model_dim_self,
                                                    nhead=hp.num_heads_self,
                                                    dim_feedforward=4 * hp.model_dim_self, dropout=hp.attn_dropout_self,
                                                    num_layers=hp.num_layers_self)



        # 用MULT融合 每个模块的输出都是[bs,query_length,model_dim_cross]
        self.ta_cross_attn=CrossAttention(hp,d_modal1=hp.d_tin,d_modal2=hp.d_ain,d_model=hp.model_dim_cross,nhead=hp.num_heads_cross,
                                          dim_feedforward=4*hp.model_dim_cross,dropout=hp.attn_dropout_cross,num_layers=hp.num_layers_cross)
        self.tv_cross_attn=CrossAttention(hp,d_modal1=hp.d_tin,d_modal2=hp.d_vin,d_model=hp.model_dim_cross,nhead=hp.num_heads_cross,
                                          dim_feedforward=4*hp.model_dim_cross,dropout=hp.attn_dropout_cross,num_layers=hp.num_layers_cross)

        #反事实 每个模块的输出都是[bs,query_length,model_dim_cross]
        if hp.whether_use_counterfactual:
            self.ta_counterfactual_attn=CounterFactualAttention(hp,d_modal1=hp.d_tin,d_modal2=hp.d_ain,d_model=hp.model_dim_cross,nhead=hp.num_heads_cross,
                                              dim_feedforward=4*hp.model_dim_cross,dropout=hp.attn_dropout_cross,num_layers=hp.num_layers_counterfactual_attention)
            self.tv_counterfactual_attn=CounterFactualAttention(hp,d_modal1=hp.d_tin,d_modal2=hp.d_vin,d_model=hp.model_dim_cross,nhead=hp.num_heads_cross,
                                              dim_feedforward=4*hp.model_dim_cross,dropout=hp.attn_dropout_cross,num_layers=hp.num_layers_counterfactual_attention)
            self.fusion_mlp_for_counterfactual_regression = SubNet(in_size=hp.model_dim_cross*2,hidden_size=hp.d_prjh,dropout=hp.dropout_prj,n_class=hp.n_class)

        self.fusion_mlp_for_regression = SubNet(in_size=hp.model_dim_cross*2,hidden_size=hp.d_prjh,dropout=hp.dropout_prj,n_class=hp.n_class)

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

    def forward(self, sentences, visual, acoustic, v_len, a_len, bert_sent, bert_sent_type, bert_sent_mask, y=None,
                mem=None,v_mask=None,a_mask=None):
        enc_word= self.uni_text_enc(sentences, bert_sent, bert_sent_type,bert_sent_mask)  # 32*50*768 (batch_size, seq_len, emb_size)

        with torch.no_grad():
            maskT = (bert_sent_mask == 0)
            maskV = self.gen_mask(visual.transpose(0,1),v_len)
            maskA = self.gen_mask(acoustic.transpose(0,1),a_len)
        # gen_npy(enc_word.mean(dim=1).cpu(), self.hp.dataset, n_clusters=25)
        # gen_npy(enc_word.mean(dim=1).cpu(), self.hp.dataset, n_clusters=50) 
        # gen_npy(enc_word.mean(dim=1).cpu(), self.hp.dataset, n_clusters=100)  
        # gen_npy(enc_word.mean(dim=1).cpu(), self.hp.dataset, n_clusters=200)  
        text_trans = enc_word.transpose(0, 1)  # torch.Size([50, 32, 768]) (seq_len, batch_size,emb_size)

        # 1. 三个模态 分别去偏
        if self.hp.whether_debias_unimodal:
            if self.hp.whether_debias_audio:
                acoustic=self.audio_mediator(acoustic,maskA)
                acoustic = self.audio_mlp(acoustic)  # [seq_len,bs,dim]
            else:
                acoustic = self.uni_audio_encoder(acoustic)  # [seq_len,bs,dim] 自注意力
            if self.hp.whether_debias_text:
                text_trans=self.text_mediator(text_trans,maskT)
                text_trans = self.text_mlp(text_trans)  # [seq_len,bs,dim]
            else:
                pass
            if self.hp.whether_debias_vision:
                visual=self.vision_mediator(visual,maskV)
                visual=self.vision_mlp(visual)#[seq_len,bs,dim]
            else:
                visual = self.uni_vision_encoder(visual)  # [seq_len,bs,dim] 自注意力
        else:
        # 1. 只有自注意力
            acoustic = self.uni_audio_encoder(acoustic)  # [seq_len,bs,dim] 自注意力
            visual=self.uni_vision_encoder(visual) #[seq_len,bs,dim] 自注意力
        vision_trans = visual
        audio_trans = acoustic

        # 2. 跨模态注意力部分
        cross_tv = self.tv_cross_attn(text_trans, vision_trans,Tmask=maskT,Vmask=maskV)
        cross_ta = self.ta_cross_attn(text_trans, audio_trans,Tmask=maskT,Amask=maskA)


        # 反事实模块构建
        if self.training and self.hp.whether_use_counterfactual:
            cross_counterfactual_tv=self.tv_counterfactual_attn(text_trans,vision_trans,Tmask=maskT,Vmask=maskV,counterfactual_attention_type=self.hp.counterfactual_attention_type)
            cross_counterfactual_ta=self.ta_counterfactual_attn(text_trans,audio_trans,Tmask=maskT,Amask=maskA,counterfactual_attention_type=self.hp.counterfactual_attention_type)
            counterfactual_fusion,counterfactual_preds=self.fusion_mlp_for_counterfactual_regression(torch.cat([cross_counterfactual_ta.mean(dim=0),cross_counterfactual_tv.mean(dim=0)],dim=1))
            fusion, preds = self.fusion_mlp_for_regression(torch.cat([cross_ta.mean(dim=0),cross_tv.mean(dim=0)],dim=1))  # 32*128,32*1
        else:
            fusion, preds = self.fusion_mlp_for_regression(torch.cat([cross_ta.mean(dim=0),cross_tv.mean(dim=0)], dim=1))  # 32*128,32*1

        if self.training:
            fusion=fusion-counterfactual_fusion
            text = text_trans[0,:,:]  # 32*768 (batch_size, emb_size)
            acoustic = self.uni_acoustic_enc(acoustic, a_len)  # 32*16
            visual = self.uni_visual_enc(visual, v_len)  # 32*16

            if y is not None:
                lld_tv, tv_pn, H_tv = self.mi_tv(x=text, y=visual, labels=y, mem=mem['tv'])
                lld_ta, ta_pn, H_ta = self.mi_ta(x=text, y=acoustic, labels=y, mem=mem['ta'])
                # for ablation use
                if self.add_va:
                    lld_va, va_pn, H_va = self.mi_va(x=visual, y=acoustic, labels=y, mem=mem['va'])
            else:  # 默认进这
                lld_tv, tv_pn, H_tv = self.mi_tv(x=text, y=visual)  # mi_tv 模态互信息
                # lld_tv:-2.1866  tv_pn:{'pos': None, 'neg': None}  H_tv:0.0
                lld_ta, ta_pn, H_ta = self.mi_ta(x=text, y=acoustic)
                if self.add_va:
                    lld_va, va_pn, H_va = self.mi_va(x=visual, y=acoustic)

            nce_t = self.cpc_zt(text, fusion)  # 3.4660
            nce_v = self.cpc_zv(visual, fusion)  # 3.4625
            nce_a = self.cpc_za(acoustic, fusion)  # 3.4933

            nce = nce_t + nce_v + nce_a  # 10.4218  CPC loss

            pn_dic = {'tv': tv_pn, 'ta': ta_pn, 'va': va_pn if self.add_va else None}
            lld = lld_tv + lld_ta + (lld_va if self.add_va else 0.0)  # -5.8927
            H = H_tv + H_ta + (H_va if self.add_va else 0.0)
        if self.training:
            return lld, nce, preds, pn_dic, H,counterfactual_preds
        else:
            return None,None, preds, None, None,None



if __name__=="__main__":
    net=Encoder(4, 8, 2,32,0.1,'relu',2)
    data=torch.randn(30,32,4)
    data_mask=pad_sequence([torch.zeros(torch.FloatTensor(sample).size(0)) for sample in data])
    data_mask[:,4:].fill_(float(1.0))
    output=net(data,data_mask.transpose(1,0))
    print(data_mask,data)