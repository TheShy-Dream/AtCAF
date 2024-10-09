import torch
import argparse
import numpy as np
import pandas as pd

from utils import *
from torch.utils.data import DataLoader
from solver import Solver
from config import get_args, get_config, output_dim_dict, criterion_dict
from data_loader import get_loader

def set_seed(seed):
    torch.set_default_tensor_type('torch.FloatTensor')
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # torch.set_default_tensor_type('torch.cuda.FloatTensor')

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        use_cuda = True


if __name__ == '__main__':
    args = get_args()
    dataset = str.lower(args.dataset.strip())
    set_seed(args.seed)
    print("Start loading the data....")
    train_config = get_config(dataset, mode='train', batch_size=args.batch_size,args=args)
    valid_config = get_config(dataset, mode='valid', batch_size=args.batch_size,args=args)
    test_config = get_config(dataset, mode='test', batch_size=args.batch_size,args=args)

    # pretrained_emb saved in train_config here
    train_loader = get_loader(args, train_config, shuffle=True)
    print('Training data loaded!')
    valid_loader = get_loader(args, valid_config, shuffle=False)
    print('Validation data loaded!')
    test_loader = get_loader(args, test_config, shuffle=False)
    print('Test data loaded!')
    print('Finish loading the data....')

    torch.autograd.set_detect_anomaly(True)

    # addintional appending
    args.word2id = train_config.word2id


    # architecture parameters
    args.d_tin, args.d_vin, args.d_ain = train_config.tva_dim
    args.dataset = args.data = dataset
    args.when = args.when
    args.n_class = output_dim_dict.get(dataset, 1)
    args.criterion = criterion_dict.get(dataset, 'MSELoss')

    import time
    result = {
        'mae': [],
        'cor': [],
        'acc7': [],
        'acc2_1': [],
        'acc2_2': [],
        'f1_1': [],
        'f1_2': [],
        'weight_name': [],
        'time': [],
        'beta':[],
        'alpha': [],
        'args':[]
    }
    for i in range(1):
        beta = args.beta
        alpha = args.alpha
        print("start", i)
        print(args)
        start_time = time.time()
        solver = Solver(args, train_loader=train_loader, dev_loader=valid_loader,
                        test_loader=test_loader, is_train=True)
        to_exl, wight_name = solver.train_and_eval()

        cost_time = time.time() - start_time
        result['mae'].append(to_exl[0])
        result['cor'].append(to_exl[1])
        result['acc7'].append(to_exl[2])
        result['acc2_1'].append(to_exl[3])
        result['acc2_2'].append(to_exl[4])
        result['f1_1'].append(to_exl[5])
        result['f1_2'].append(to_exl[6])
        result['weight_name'].append(wight_name)
        result['time'].append(cost_time)
        result['beta'].append(beta)
        result['alpha'].append(alpha)
        result['args'].append(args)
        print(result)
        print('*'*30)

    data_frame = pd.DataFrame(
        data={'mae': result['mae'], 'cor': result['cor'], 'acc7': result['acc7'], 'acc2_1':result['acc2_1'],
              'acc2_2': result['acc2_2'], 'f1_1': result['f1_1'], 'f1_2': result['f1_2'],
              'time': result['time'], 'weight_name': result['weight_name'], 'beta': result['beta'],
              'alpha': result['alpha'],'args':result['args']},
        index=range(1)
    )

    now_time = time.strftime("_%m%d_%H%M", time.localtime())
    path = 'pre_trained_best_models_mosei/' + args.dataset + now_time + '_result.csv'
    print(path)
    data_frame.to_csv(path)