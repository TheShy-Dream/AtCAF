# AtCAF

The code is related to the paper: AtCAF: Attention-based causality-aware fusion network for multimodal sentiment analysis.

## Datasets

You can download the CMU-MOSI and CMU-MOSEI datasets using [CMU-MultimodalDataSDK](https://github.com/Jie-Xie/CMU-MultimodalDataSDK).

You can download the UR-FUNNY dataset using [UR-FUNNY resp](https://github.com/ROC-HCI/UR-FUNNY).

You can download the ood version of  CMU-MOSI and CMU-MOSEI datasets in https://msa-clue.wixsite.com/clue.

## Preparation

Create a folder named `npy_folder` in the root directory.

```
mkdir npy_folder
```

In order to perfectly replicate the precision, please use these functions from `model.py` to generate the global dictionary initialization as an .npy file and place it in the `npy_folder`.

```
gen_npy(enc_word.mean(dim=1).cpu(), self.hp.dataset, n_clusters=25)
gen_npy(enc_word.mean(dim=1).cpu(), self.hp.dataset, n_clusters=50) 
gen_npy(enc_word.mean(dim=1).cpu(), self.hp.dataset, n_clusters=100)  
gen_npy(enc_word.mean(dim=1).cpu(), self.hp.dataset, n_clusters=200)  
```

## Training

```
python main.py
```

## Environment Requirements

python == 3.8.8

torch == 1.8.1

numpy == 1.20.0

## Citation

If you use this code please cite it as:

```
@article{huang2025atcaf,
  title={AtCAF: Attention-based causality-aware fusion network for multimodal sentiment analysis},
  author={Huang, Changqin and Chen, Jili and Huang, Qionghao and Wang, Shijin and Tu, Yaxin and Huang, Xiaodi},
  journal={Information Fusion},
  volume={114},
  pages={102725},
  year={2025},
  publisher={Elsevier}
}
```

**Thank you for your support. If you have any questions, feel free to post them in the [issues](https://github.com/TheShy-Dream/AtCAF/issues) or contact us via [irelia@zjnu.edu.cn](mailto:irelia@zjnu.edu.cn).**