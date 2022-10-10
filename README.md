# MSRL
This repository contains the reproducible codes for the numerical results in the Paper "Deep Sufficient Representation Learning via Mutual Information"

## Requirements

To install requirements:

    pip install -r requirements.txt

## Numerical Experiments

### Simulation

To reproduce the results for the simulation study in the paper, the files in should be implemented. For example, to obtain the MSRL results for model 1) and distributional setting 1), run the following command:

    python3 midr_demo.py --model=1 --scenario=1 --latent_dim=1
    
After running midr_demo.py, to obtain the SDR results for model 1) and distributional setting 1), run the following command:
    
    python3 sdr_demo.py --model=1 --scenario=1 --latent_dim=1
    
The default dimension $p$ of the covariate vector is 10. If you want to obtain the results for $p=30$, assign the value 30 to the variable $p$ in midr_demo.py and sdr_demo.py.

### The superconductivity dataset

The dataset is available at [https://archive.ics.uci.edu/ml/datasets/Superconductivty+Data](https://archive.ics.uci.edu/ml/datasets/Superconductivty+Data).

### The Pole-Telecommunication dataset

The dataset is available at [https://www.dcc.fc.up.pt/~ltorgo/Regression/DataSets.html](https://www.dcc.fc.up.pt/~ltorgo/Regression/DataSets.html).

### The MNIST dataset

The dataset is available at [http://yann.lecun.com/exdb/mnist](http://yann.lecun.com/exdb/mnist)
