# Adversarial Robustness of Temporal GNNs on Node-Level Tasks

Final project for a course at the University of Cambridge.

This repository contains the attack implementations and experiment scripts used to evaluate 
adversarial robustness of Temporal Graph Neural Networks on node-level classification tasks, 
using the tgbn-trade benchmark from TGB.

## Based On

This project builds directly on the [TGM library](https://github.com/tgm-team/tgm):

> Chmura et al. "TGM: A Modular and Efficient Library for Machine Learning on Temporal Graphs." arXiv:2510.07586, 2025.

The model files `gcn.py`, `tgat.py`, `tgcn.py`, and `tgn.py` are modified from TGM's 
`examples/nodeproppred/` directory. All modifications add attack hooks or poisoned data 
loading to the original training loops.
