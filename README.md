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
```
The DN348 dataset and DNwild dataset can be downloaded from [here](https://github.com/chenjingong/DN-ReID/tree/main/data_path).
## Training
Train a model by: 
```text
python train.py --dataset dn348 --gpu 0
```
-- datasets: which dataset to use, including dn348, dnwild.
-- gpu: which GPU to use.
## Test
Test a model on dn348 or dnwild dataset by
```text
python train.py --test-only --resume 'model_path' --dataset dn348 --gpu 0
```
-- test-only: Used for model testing.
-- resume: the saved model path.
-- datasets: which dataset to use, including dn348, dnwild.
-- gpu: which GPU to use.
## Traffic simulation
The `simulation/` folder reproduces the system-level results of the paper (Sections V and VI): a mesoscopic simulation of a 14-node, 18-link network with a motorway, a signalized arterial and a camera at every node. Vehicles are re-identified between consecutive cameras, link travel times are estimated from the matches, and a route guidance controller advises drivers every 60 s. Matching errors are generated from the feature distances measured for DNDM, SFM and LPSMNet.
## Contents
```text
simulation/
├── data/                  measured feature distances (DN-348, 50 test identities, 5 day + 5 night images each)
│   ├── DNDM_distances.csv
│   ├── SFM_distances.csv
│   └── LPSMNet_distances.csv
├── error_model.py         matching-error model built from the measured distances
├── simulator.py           network, evening and morning scenarios, closed-loop simulator
├── run_experiments.py     runs all simulations (resumable, optional parallel workers)
├── analyze.py             tables and figures of the paper from the simulation results
├── requirements.txt
└── results/               results reported in the paper
    ├── evening.csv, morning.csv          one row per simulation run
    ├── summary.md                        all numbers reported in Section VI
    ├── error_model_consistency.csv       check of the error model against the benchmarks (Section V)
    ├── hourly.csv, sensitivity.csv       data of Figs. 4 and 5
    └── figures/fig_moments.pdf, fig_sensitivity.pdf
```
## Requirements
The simulation does not need PyTorch or a GPU. It runs with Python 3.8 or later:
```text
cd simulation
pip install -r requirements.txt
```
## Distance files
Each file lists one image pair per row:
```text
model,id_i,id_j,modality_i,modality_j,distance,label
LPSMNet,0,0,day,day,1.335566,same
```
`distance` is the Euclidean distance between the L2-normalized embeddings of the two images, `modality_*` is `day` or `night`, and `label` is `same` or `different`. To evaluate another ReID model, add a file `simulation/data/<MODEL>_distances.csv` in the same format and add `<MODEL>` to `MODELS` in `error_model.py`.
## Running the simulations
Quick test (a few runs of each condition, about 2 minutes):
```text
python run_experiments.py --quick
```
Full experiments of the paper (evening period with sensitivity study, and morning period):
```
python run_experiments.py --workers 8
```
This performs 1,960 evening runs (400 main runs and 1,560 sensitivity runs) and 400 morning runs, about 1 h on a single core. Results are appended to results/evening.csv and results/morning.csv; an interrupted run continues where it stopped. Each scenario can also be run separately with --scenario evening or --scenario morning. Random seeds are fixed, so the runs reproduce the results in results/ exactly; delete or move these files before running to generate them again.
| Experiment | Conditions | Replications |
| :--- | :--- | :---: |
| Main, evening (18:00–02:00) and morning (00:00–08:00) | no guidance, perfect sensing, 3 models × 2 matching strategies (naive, traffic-aware) | 50 |
| Sensitivity, evening | night demand ×0.6 and ×1.4, compliance 0.4 and 1.0, logit sensitivity 1/60 and 1/240 s⁻¹, control interval 30 and 180 s | 20 per level |

## Reproducing the tables and figures
```
python analyze.py
```
