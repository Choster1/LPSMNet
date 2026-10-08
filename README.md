# LPSMNet
# LPSMNet: Robust Day–Night Vehicle Re-Identification for Intelligent Transportation Systems
by Mingfu Xiong, Zhang Jiang, Tengfei Tu, Javier Del Ser, Khan Muhammad
<img width="1095" height="604" alt="image" src="https://github.com/user-attachments/assets/38ba938c-de36-4c6d-93d7-3cca41f9b4ee" />
## Introduction
LPSMNet is a framework for day-night cross-domain vehicle re-identification(DN-VReID). It consists of two key modules: 
- **Learnable Mask Module (LMM)**: adaptively suppresses glare and noise in nighttime images via a learnable mask, improving feature robustness under low-light conditions.
- **Prototype Structure Semantic Module (PSSM)**: learns a set of shared structural prototypes to align day and night features into a unified semantic space, enhancing cross-domain structural consistency and identity discriminability.
## Environment
- Python 3.8.20
- PyTorch 1.10.1
- CUDA 11.1
- NVIDIA A30 GPU
## Datasets
After downloading all datasets, please create a `data_path/` folder in the root directory, and organize it as follows:
```text
data_path/
├── dn348/
    ├──day
    ├──night
    ├──train_test_split
├── dnwild/
    ├──day
    ├──night
    ├──train_test_split
text```
The DN348 dataset and DNwild dataset can be downloaded from [here](https://github.com/chenjingong/DN-ReID/tree/main/data_path).
## Training
Train a model by: 
