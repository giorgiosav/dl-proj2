# EE-559 Deep Learning - Project 2

## Introduction
In this project we created a small Deep Learning (DL) framework, using Pytorch’s tensor operations but no autograd machinery, that enables easy differentiation. 
The implementation contains all the basic modules to build, train and test a simple DL network. 
Our framework’s performances are later evaluated by implementing a basic neural network with 3 hidden layers and using it for a geometric classification task, with a randomly generated dataset. 
Finally, we compared our implementation to an equivalent one in Pytorch and found similar performance under the same conditions.

## Prerequites
### Basic prerequisites
The minimum requirements for the `test.py` are:
- `Python` (tested on version **_3.8_**)
- [pip](https://pip.pypa.io/en/stable/) (tested on version *20.2.3*) (For package installation, if needed)
- `Pytorch` (tested on version *1.6.0*)
- `Matplotlib` (tested on version *3.3.2*)

**NOTE**: If you have LaTeX installed on your machine, you can uncomment lines `plot.py:12-20` to produce plots with a "LaTeX style". 

## Usage instruction
1. Open CMD/Bash
2. Activate the environment with needed packages installed (if you have Anaconda or any virtual environment on your machine)
3. Move to the src folder, where the `test.py` is located
4. Execute the command ```python test.py``` with zero or more of the following arguments:
```
Optional arguments:
  -validation     Run validation on the model. If not set, already selected
                  best eta SGD param will be used. (default false)
  -pytorch        Create a copy of the model in pytorch to estimate
                  performances differencesbetween our implementation and one
                  using NN package (default false)
  -plots          Create the errors/loss plot over epochs for the selected
                  model as shown in the report. Create prediction
                  visualization on a xy grid every 50 epochs (default false)
  -n_runs N_RUNS  Define number of runs of the train/test process (default 10)

  Possible activations:
    Select one of the two possible activations
    -tanh           Use Tanh as activation function (default)
    -relu           Use ReLU as activation function

```

## Results
Detailed explanation of the results we achieved can be found in our [Report](https://github.com/giorgiosav/dl-proj2/blob/master/Project2_Report.pdf) and inside the
[plot folder](https://github.com/giorgiosav/dl-proj2/tree/master/src/plot), where you can find different classification results with different activation functions,
along with error with respect to the Pytorch reference implementation. 

## Folder structure
```
.
├── extra_stuff
│   ├── ee559-miniprojects.pdf
│   └── torch-test.py
├── Project2_Report.pdf
├── README.md
└── src
    ├── data.py
    ├── myNN.py
    ├── myOptim.py
    ├── plot
    │   ├── circles
    │   │   ├── test_classes_relu0.pdf
    │   │   ├── test_classes_relu25.pdf
    │   │   ├── test_classes_relu50.pdf
    │   │   ├── test_classes_relu74.pdf
    │   │   ├── test_classes_tanh0.pdf
    │   │   ├── test_classes_tanh25.pdf
    │   │   ├── test_classes_tanh50.pdf
    │   │   └── test_classes_tanh74.pdf
    │   ├── err_pytorch_relu_15runs.pdf
    │   ├── err_pytorch_tanh_15runs.pdf
    │   ├── err_relu_15runs.pdf
    │   ├── err_tanh_15runs.pdf
    │   ├── losstot_pytorch_relu_15runs.pdf
    │   ├── losstot_pytorch_tanh_15runs.pdf
    │   ├── losstot_relu_15runs.pdf
    │   └── losstot_tanh_15runs.pdf
    ├── plot.py
    ├── test.py
    ├── train.py
    └── validation.py

```

## Code reproducibility
Even if all the random seeds are set in the `test.py`, `Pytorch` always has a bit of randomness.
For this reason, the reader is advised that **different runs with the same parameters, and also different runs of CV can produce slightly different results**.  

## Authors
- [Manuel Leone](https://github.com/manuleo)
- [Giorgio Savini](https://github.com/giorgiosav)
