import os
import argparse
from datetime import datetime
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import pprint
from torch import optim
import torch.nn as nn

# path to a pretrained word embedding file
word_emb_path = '/home/henry/glove/glove.840B.300d.txt'
assert (word_emb_path is not None)

username = Path.home().name
project_dir = Path(__file__).resolve().parent.parent
sdk_dir = project_dir.joinpath('CMU-MultimodalSDK')
data_dir = project_dir.joinpath('datasets')
data_dict = {'mosi': data_dir.joinpath('MOSI'), 'mosei': data_dir.joinpath(
    'MOSEI'), 'ur_funny': data_dir.joinpath('ur_funny_v2')}
optimizer_dict = {'RMSprop': optim.RMSprop, 'Adam': optim.Adam}
activation_dict = {'elu': nn.ELU, "hardshrink": nn.Hardshrink, "hardtanh": nn.Hardtanh,
                   "leakyrelu": nn.LeakyReLU, "prelu": nn.PReLU, "relu": nn.ReLU, "rrelu": nn.RReLU,
                   "tanh": nn.Tanh}

output_dim_dict = {
    'mosi': 1,
    'mosei_senti': 1,
    'sims':1,
    'ur_funny':2
}

criterion_dict = {
    'mosi': 'L1Loss',
    'iemocap': 'CrossEntropyLoss',
    'ur_funny': 'CrossEntropyLoss',
    'mosei':'L1Loss',
    'sims':'L1Loss'
}


def get_args():
    parser = argparse.ArgumentParser(description='MOSI-and-MOSEI Sentiment Analysis')
    parser.add_argument('-f', default='', type=str)

    # Tasks
    parser.add_argument('--dataset', type=str, default='mosi', choices=['mosi', 'mosei','sims','ur_funny'],
                        help='dataset to use (default: mosi)')
    parser.add_argument('--data_path', type=str, default='datasets',
                        help='path for storing the dataset')
    parser.add_argument('iid_setting',action='store_false',help="test using iid dataset")
    parser.add_argument('ood_setting',action='store_false',help="test using ood dataset")
    parser.add_argument('seven_class',action='store_false',help="seven classification")
    parser.add_argument('--npy_path',type=str,default='npy_folder',help='path for storing the Kmeans center')
    parser.add_argument('--npy_selection',choices=['bert_0','xlnet_0','bert_mean','xlnet_mean'],default="bert_mean",help="text npy file")

    # Dropouts
    parser.add_argument('--dropout_a', type=float, default=0.1,
                        help='dropout of acoustic LSTM out layer')
    parser.add_argument('--dropout_v', type=float, default=0.1,
                        help='dropout of visual LSTM out layer')
    parser.add_argument('--dropout_prj', type=float, default=0.1,
                        help='dropout of projection layer')
    parser.add_argument('--text_encoder',choices=['bert','xlnet'],default="bert",help="text_encoder")

    # Kmeans size
    parser.add_argument('--use_kmean',action='store_false',help='use Kmean initialization')
    parser.add_argument('--kmean_size',type=int,default=50,help='size of confounder dictionary')
    parser.add_argument('--audio_kmean_size',type=int,default=50,help='size of confounder dictionary')
    parser.add_argument('--text_kmean_size',type=int,default=50,help='size of confounder dictionary')
    parser.add_argument('--vision_kmean_size',type=int,default=50,help='size of confounder dictionary')

    # Debias模块
    parser.add_argument('--whether_debias_unimodal',action='store_false',help='whether to use debias module in unimodal')
    parser.add_argument('--whether_debias_audio',action='store_false',help='whether to debias audio')
    parser.add_argument('--whether_debias_text',action='store_false',help='whether to debias text')
    parser.add_argument('--whether_debias_vision',action='store_false',help='whether to debias vision')
    parser.add_argument('--audio_debias_layers',type=int,default=3,help="layers in debias self attention")
    parser.add_argument('--vision_debias_layers',type=int,default=3,help="layers in debias self attention")
    parser.add_argument('--text_debias_layers',type=int,default=3,help="layers in debias self attention")
    parser.add_argument('--attn_dropout_debias', type=float, default=0.1, help="dropout in single modal self attention")

    #单模态Mlp模块
    parser.add_argument('--audio_mlp_hidden_size',type=int,default=16,help="the hidden size of mlp in audio projection")
    parser.add_argument('--vision_mlp_hidden_size',type=int,default=16,help="the hidden size of mlp in vision projection")
    parser.add_argument('--text_mlp_hidden_size',type=int,default=64,help="the hidden size of mlp in text projection")

    # CounterFactual模块
    parser.add_argument('--whether_use_counterfactual',action='store_false',help="whether to use counterfactual")
    parser.add_argument('--counterfactual_attention_type',choices=['random', 'uniform', 'reversed','shuffle'],default="shuffle",help="the type of counterfactual attention")
    parser.add_argument('--num_layers_counterfactual_attention',type=int,default=2,help="layers in counterfactual attention")

    #自注意力部分
    parser.add_argument('--model_dim_self',type=int,default=30,help="dim in single modal self attention")
    parser.add_argument('--num_heads_self',type=int,default=5,help="heads in single modal self attention")
    parser.add_argument('--num_layers_self', type=int, default=3, help="layers in self attention")
    parser.add_argument('--attn_dropout_self', type=float, default=0.1, help="dropout in single modal self attention")

    #跨模态注意力部分
    parser.add_argument('--model_dim_cross',type=int,default=30,help="dim in single modal cross attention")
    parser.add_argument('--num_heads_cross',type=int,default=5,help="heads in single modal cross attention")
    parser.add_argument('--num_layers_cross', type=int, default=6, help="layers in cross attention")
    parser.add_argument('--attn_dropout_cross', type=float, default=0.1, help="dropout in single modal cross attention")


    # infonce
    # parser.add_argument('--embed_dropout_infonce', type=float, default=0.1, help="infonce emb dropout")
    # parser.add_argument('--embed_dropout_infonce_cross', type=float, default=0.1, help="infonce emb dropout after cross attention")
    #
    # fusion method
    # parser.add_argument('--fusion',type=str,default='sum',help='the fusion method used in final fusion')#[sum,fusion]

    # 最后的全模态自注意力
    # parser.add_argument('--model_dim_final',type=int,default=30,help="model dim in final self attention")
    # parser.add_argument('--attn_dropout_final',type=float,default=0.1,help="dropout in final self attention")
    # parser.add_argument('--num_layers_final', type=int, default=6, help="layers in final self attention")
    # parser.add_argument('--num_heads_final',type=int,default=6,help="heads in final self attention")


    # 我加的transformer dropout
    # parser.add_argument('--attn_dropout', type=float, default=0.1,
    #                     help='attention dropout')
    # parser.add_argument('--attn_dropout_a', type=float, default=0.0,
    #                     help='attention dropout (for audio)')
    # parser.add_argument('--attn_dropout_v', type=float, default=0.0,
    #                     help='attention dropout (for visual)')
    parser.add_argument('--relu_dropout', type=float, default=0.1,
                        help='relu dropout')
    parser.add_argument('--res_dropout', type=float, default=0.1,
                        help='residual block dropout')
    parser.add_argument('--attn_mask', action='store_false', help='use attention mask for Transformer (default: true)')
    # parser.add_argument('--out_dropout', type=float, default=0.0,
    #                     help='output layer dropout')
    parser.add_argument('--embed_dropout', type=float, default=0.25,
                        help='embedding dropout')


    parser.add_argument('--vonly', action='store_true',
                        help='use the crossmodal fusion into v (default: False)')
    parser.add_argument('--aonly', action='store_true',
                        help='use the crossmodal fusion into a (default: False)')
    parser.add_argument('--lonly', action='store_true',
                        help='use the crossmodal fusion into l (default: False)')

    # parser.add_argument('--layers', type=int, default=5,
    #                     help='number of layers in the network (default: 5)')
    # parser.add_argument('--num_heads', type=int, default=5,
    #                     help='number of heads for the transformer network (default: 5)')

    # Architecture
    # parser.add_argument('--n_tv', type=int, default=0,
    #                     help='number of V-T transformer  (default: 0)')
    # parser.add_argument('--n_ta', type=int, default=1,
    #                     help='number of A-T transformer (default: 1)')

    parser.add_argument('--multiseed', action='store_true', help='training using multiple seed')
    parser.add_argument('--contrast', action='store_false', help='using contrast learning')
    parser.add_argument('--add_va', action='store_false', help='if add va MMILB module')  # 是否采用VA互信息最大化
    parser.add_argument('--n_layer', type=int, default=1,
                        help='number of layers in LSTM encoders (default: 1)')
    parser.add_argument('--cpc_layers', type=int, default=1,
                        help='number of layers in CPC NCE estimator (default: 1)')
    parser.add_argument('--d_vh', type=int, default=16,
                        help='hidden size in visual rnn')
    parser.add_argument('--d_ah', type=int, default=16,
                        help='hidden size in acoustic rnn')
    parser.add_argument('--d_vout', type=int, default=16,
                        help='output size in visual rnn')
    parser.add_argument('--d_aout', type=int, default=16,
                        help='output size in acoustic rnn')
    parser.add_argument('--bidirectional', action='store_true', help='Whether to use bidirectional rnn')
    parser.add_argument('--d_prjh', type=int, default=64,
                        help='hidden size in projection network')
    parser.add_argument('--pretrain_emb', type=int, default=768,
                        help='dimension of pretrained model output')
    parser.add_argument('--mem_size',type=int,default=1,
                        help='Memory size')

    # Activations
    parser.add_argument('--mmilb_mid_activation', type=str, default='ReLU',
                        help='Activation layer type in the middle of all MMILB modules')
    parser.add_argument('--mmilb_last_activation', type=str, default='Tanh',
                        help='Activation layer type at the end of all MMILB modules')
    parser.add_argument('--cpc_activation', type=str, default='Tanh',
                        help='Activation layer type in all CPC modules')

    # Training Setting
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='batch size (default: 32)')
    parser.add_argument('--clip', type=float, default=1.0,
                        help='gradient clip value (default: 0.8)')
    parser.add_argument('--lr_main', type=float, default=1e-3,
                        help='initial learning rate for main model parameters (default: 1e-3)')
    parser.add_argument('--lr_bert', type=float, default=5e-5,
                        help='initial learning rate for bert parameters (default: 5e-5)')
    parser.add_argument('--lr_mmilb', type=float, default=1e-3,  # Modality Mutual Information Lower Bound（MMILB）
                        help='initial learning rate for mmilb parameters (default: 1e-3)')


    parser.add_argument('--alpha', type=float, default=0.1, help='weight for CPC NCE estimation item (default: 0.1)')
    parser.add_argument('--beta', type=float, default=0.1, help='weight for lld item (default: 0.1)')
    parser.add_argument('--eta',type=float,default=0.1,help='weight for counterfactual')
    
    parser.add_argument('--weight_decay_main', type=float, default=1e-4,
                        help='L2 penalty factor of the main Adam optimizer')  # ！！！！ 越大训练越不好  正则  小了
    parser.add_argument('--weight_decay_bert', type=float, default=1e-4,
                        help='L2 penalty factor of the main Adam optimizer')
    parser.add_argument('--weight_decay_club', type=float, default=1e-4,
                        help='L2 penalty factor of the main Adam optimizer')

    parser.add_argument('--optim', type=str, default='Adam',
                        help='optimizer to use (default: Adam)')
    parser.add_argument('--num_epochs', type=int, default=40,
                        help='number of epochs (default: 40)')
    parser.add_argument('--when', type=int, default=20,
                        help='when to decay learning rate (default: 20)')
    parser.add_argument('--patience', type=int, default=4,
                        help='when to stop training if best never change')
    parser.add_argument('--update_batch', type=int, default=1,
                        help='update batch interval')

    # Logistics
    parser.add_argument('--log_interval', type=int, default=100,
                        help='frequency of result logging (default: 100)')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    args = parser.parse_args()

    valid_partial_mode = args.lonly + args.vonly + args.aonly
    # 默认全为False的话 valid_partial_mode==0

    if valid_partial_mode == 0:
        args.lonly = args.vonly = args.aonly = True
    elif valid_partial_mode != 1:
        raise ValueError("You can only choose one of {l/v/a}only.")

    return args


def str2bool(v):
    """string to boolean"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Config(object):
    def __init__(self, data, mode='train',args=None):
        """Configuration Class: set kwargs as class attributes with setattr"""
        self.dataset_dir = data_dict[data.lower()]
        self.sdk_dir = sdk_dir
        self.mode = mode
        self.args=args
        # Glove path
        self.word_emb_path = word_emb_path

        # Data Split ex) 'train', 'valid', 'test'
        self.data_dir = self.dataset_dir

    def __str__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str


def get_config(dataset='mosi', mode='train', batch_size=32,args=None):
    config = Config(data=dataset, mode=mode,args=args)

    config.dataset = dataset
    config.batch_size = batch_size

    return config
