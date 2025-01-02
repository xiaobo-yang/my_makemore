# Introduction

This repo is a learning note of Andrej Karpathy's [Make More](https://github.com/karpathy/makemore) and Youtube series [nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero/).

# Content
- Infrastructure
  - `micrograd.py`: a minimalistic autograd engine for scalar
  - `tiny_torch.py`: a minimalistic autograd engine for tensor
  - `tiny_torch_nn.py`: a minimalistic implementation of different models using `tiny_torch.py`
  - `micro_torch/`: try to implement a minimalistic autograd data structure like torch tensor.
- Play with models
  - `bigram.py` and `mlp_basic.ipynb`: a simple language model first glance
  - `mlp_diagnosis.ipynb` and `mlp_diagnosis_clean.ipynb`: some diagnosis of the MLP model, the latter one uses `tiny_torch.py`
  - `mlp_backprop.ipynb`: a backpropagation implementation of the MLP model
  - `mlp_demo.ipynb`: a demo of the MLP model use manual backpropagation, and a demo of wavenet
  - `model_playground.ipynb`: a playground for different models, all of them are implemented in tiny_torch_nn.py.
