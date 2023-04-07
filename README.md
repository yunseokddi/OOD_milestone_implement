# OOD_milestone_implement

---
This is Pytorch implementation of the OOD experiments with [pytorch-template](https://github.com/victoresque/pytorch-template) in the following milestone papers:

- [MSP] A Baseline for Detecting Missclassified and Out-of-Distribution Examples in Neural Networks | **[ICLR2017]** [[paper]](https://arxiv.org/pdf/1610.02136.pdf)
- [ODIN] Enhancing The Reliability of Out-of-Distribution Image Detection in Neural Networks | **[ICLR2018]** [[paper]](https://arxiv.org/pdf/1706.02690.pdf)
- [Mahalanobis] A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks | **[NIPS2018]** [[paper]](https://arxiv.org/abs/1807.03888)

## Installation & requirement

---
The current version of the code has been tested with `python 3.6.9` on an Ubuntu 18.04 OS with the following versions of Pytorch and Torchvision:

- `pytorch 1.7.1`
- `torchvision 0.8.2`

Further Python-packages used are listed in `requirements.txt`.
Assuming Python and pip are set up, these packages can be installed using:
```bash
pip install -r requirements.txt
```

---
## Folder Structure
```angular2html
OOD_milestone_implement/
│
├── data_loader/
│   └── data_loaders.py - full training dataloader
│   └── in_data_loaders.py - in-distribution eval dataloader
│   └── out_data_loaders.py - out-of-distribution eval dataloader
│   └── svhn_data_loaders.py - svhb dataset data loader
├── datasets/ - put standard or your dataset
├── detector/ - package for evaluation
│   └── detector.py
├── model/ - models, losses, and metrics
│   ├── densenet.py
│   ├── metric.py - metrics of confidence score and evaluation
│   └── wideresnet.py
├── runs/ - tensorboard log folder, it will be updated
├── trainer/ - trainers
│   └── trainer.py/ - full training src
├── utils/ - small utility functions
│   └── select_svhn_data.py - function for select svhn data file
├── eval_ood_detection.py - OOD evaluation of trained model
└── train.py - main script to start training
```

## Running custom experiments

---
**The main options of this script are:**
- `--in_dataset`: choose in distribution dataset (`CIFAR-10`|`CIFAR-100`|`SVHN`)
- `--model_arch`: choose model architecture (`densenet`|`wideresnet`)
- `--method`: choose confidence scoring method (`msp`|`odin`|`mahalanobis`)

**How to run?**
1. If you want to full-train
```train
python3 train.py --in_dataset CIFAR-10 --model_arch densenet --epochs 100 --name experiment_1 --tensorboard
```
2. If you want to evaluate OOD score
```eval
python3 eval_ood_detection.py --in_dataset CIFAR-10 --method msp --name eval_msp
```

**Result sample**
```result
Natural OOD
nat_in vs. nat_out
in_distribution: CIFAR-10
out_distribution: All
Model Name: test_odin

 OOD detection method: odin
 FPR    DTERR  AUROC  AUIN   AUOUT
 17.20   9.00  95.51  96.12  94.30
 ```


## License

---
This project is licensed under the Apache-2.0 License. See [LICENSE](https://github.com/yunseokddi/OOD_milestone_implement/blob/main/license) for more details

## Reference

---
- **Project structure**: [https://github.com/jfc43/informative-outlier-mining](https://github.com/jfc43/informative-outlier-mining)
- **Odin**: [https://github.com/facebookresearch/odin](https://github.com/facebookresearch/odin)
- **Mahalanobis**: [https://github.com/pokaxpoka/deep_Mahalanobis_detector](https://github.com/pokaxpoka/deep_Mahalanobis_detector)
