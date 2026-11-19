# BiMC

This is the official implementation of paper **Enhancing Few-Shot Class-Incremental Learning via Training-Free Bi-Level Modality Calibration (CVPR 2025)**.

## Abstract

Few-shot Class-Incremental Learning (FSCIL) challenges models to adapt to new classes with limited samples, presenting greater difficulties than traditional class-incremental learning. While existing approaches rely heavily on visual models and require additional training during base or incremental phases, we propose a training-free framework that leverages pre-trained visual-language models like CLIP. At the core of our approach is a novel Bi-level Modality Calibration (BiMC) strategy. Our framework initially performs intra-modal calibration, combining LLM-generated fine-grained category descriptions with visual prototypes from the base session to achieve precise classifier estimation. This is further complemented by inter-modal calibration that fuses pre-trained linguistic knowledge with task-specific visual priors to mitigate modality-specific biases. To enhance prediction robustness, we introduce additional metrics and strategies that maximize the utilization of limited data. Extensive experimental results demonstrate that our approach significantly outperforms existing methods.

## Installation

### Dataset

Please follow [CEC](https://github.com/icoz69/CEC-CVPR2021) to download *mini*-ImageNet, CUB-200 and CIFAR-100.

### Requirement

- \`torch==1.13.1\`
- \`torchvision==0.14.1\`
- \`yacs==0.1.8\`
- \`tqdm==4.66.1\`
- \`ftfy==6.1.1\`
- \`regex==2023.10.3\`
- \`scikit-learn==1.3.2\`

## Experiments

First, remember to modify the data path \`ROOT\` in the dataset configuration file.

~~~BASH
# CIFAR BIMC
python main.py --data_cfg ./configs/datasets/cifar100.yaml --train_cfg ./configs/trainers/bimc.yaml

# CIFAR BIMC_Ensemble
python main.py --data_cfg ./configs/datasets/cifar100.yaml --train_cfg ./configs/trainers/bimc_ensemble.yaml

# MiniImagenet BIMC
python main.py --data_cfg ./configs/datasets/miniimagenet.yaml --train_cfg ./configs/trainers/bimc.yaml

# MiniImagenet BIMC_Ensemble
python main.py --data_cfg ./configs/datasets/miniimagenet.yaml --train_cfg ./configs/trainers/bimc_ensemble.yaml

# CUB200 BIMC
python main.py --data_cfg ./configs/datasets/cub200.yaml --train_cfg ./configs/trainers/bimc.yaml

# CUB200 BIMC_Ensemble
python main.py --data_cfg ./configs/datasets/cub200.yaml --train_cfg ./configs/trainers/bimc_ensemble.yaml
~~~

## CSPC V2.1

The dev branch also includes an initial CSPC V2.1 implementation. It keeps the CLIP image and text encoders frozen, trains one shared text prompt on the base session, and continues prompt/scorer/gate optimization with real incremental-session K-shot samples. It does not create pseudo-incremental tasks and does not replay historical images. Historical visual prototypes are kept fixed; only base prototypes are used as transfer sources, while all seen prototypes participate in the unified visual classifier.

The initial configuration values are starting points for controlled experiments, not claimed optimal hyperparameters. Review and tune them only through the planned validation and ablation protocol. All non-empty class descriptions in the JSON file are encoded and averaged into one normalized class text feature.

Example:

~~~BASH
# MiniImageNet CSPC V2.1
python main.py --data_cfg ./configs/datasets/miniimagenet.yaml --train_cfg ./configs/trainers/cspc.yaml

# CUB200 CSPC V2.1
python main.py --data_cfg ./configs/datasets/cub200.yaml --train_cfg ./configs/trainers/cspc.yaml

# CIFAR100 CSPC V2.1
python main.py --data_cfg ./configs/datasets/cifar100.yaml --train_cfg ./configs/trainers/cspc.yaml
~~~

CSPC checkpoints and session metrics are written under TRAINER.CSPC.OUTPUT_DIR/<dataset>/seed_<seed>, preventing different dataset/seed runs from overwriting each other. AA is the arithmetic mean of session-level overall sample accuracies; AA_class_macro is reported separately as the mean of session-level class-macro accuracies. PD is the drop in base-class macro accuracy from session 0 to the final session. Checkpoints contain prompts, transfer parameters, and prototype memory; they do not store historical images or per-image features.

## Acknowledgment

In this repository, we build our code based on the following excellent open-source projects. We sincerely thank all the authors for sharing their great work:

- [LP-DiF](https://github.com/1170300714/LP-DiF)
- [TEEN](https://github.com/wangkiw/TEEN)
- [FeCAM](https://github.com/dipamgoswami/FeCAM)
- [CuPL](https://github.com/sarahpratt/CuPL)
- [AdaptCLIPZS](https://github.com/cvl-umass/AdaptCLIPZS)
- [LibContinual](https://github.com/RL-VIG/LibContinual)
- [LibFewShot](https://github.com/RL-VIG/LibFewShot)
