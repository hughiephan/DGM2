# Dynamic Gaussian Mixture based Deep Generative Model ForRobust Forecasting on Sparse Multivariate Time Series with MIMIC3

- DGM2_L uses LSTM for transition 
- DGM2_O uses ODE for transition

## Prerequisites:
- python version 3.11.9
- pytorch 
- matplotlib
- pandas
- scikit-learn 
- tensorboardX
- torchdiffeq

## Run

Go to folder `data` then unzip mimic3.zip and run:

```
python generate_time_series.py --dataset MIMIC3
```

Then go back to the main folder and run:

```
python train.py --dataset MIMIC3 --model DGM2_L -b 100 --epochs 50 --GPU --GPUID 0 --max_kl 5 --use_gate --wait_epoch 0
```

### The arguments for running this program are:

--dataset: the name of the dataset (MIMIC3)

--model: the model name (DGM2_L or DGM2_O)

-b: mini-batch size

--epochs: epoch count for training

--GPU: flag of using GPU or not

--GPUID: ID of the GPU for running train.py

--max_kl: the maximal coefficient for the KL divergence term in the loss function. We use annealing technique to tune the coefficient during the training process.

--use_gate: flag of using the gate function or not

--gaussian: the parameter gamma to balance the dynamic component and the basis mixture component in the dynamic gaussian mixture distribution, which will take effect when --use_gate is not used, e.g. "--gaussian 0.001"

--wait_epoch: number of epochs for the warm-up phase with annealing technique during which the coefficient for the KL divergence term in the loss function is zero. The default value is 0

--cluster_num: number of clusters for DGM2_L and DGM2_O. The default value is 20.