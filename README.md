# Hypergraph Transformer with Efficient Kernel-Sparse Attention

This repository implements an experimental Transformer layer extended to hypergraphs, inspired by the paper "Transformers are Graph Neural Networks" (arXiv:2506.22084). It treats Transformers as GNNs on fully connected graphs and extends to hypergraphs by embedding nodes and hyperedges as tokens, with bipartite self-attention.

## Features
- **Hypergraph Handling**: Models higher-order relations (hyperedges connecting >2 nodes) via joint attention on nodes and hyperedges.
- **Efficient Attention**: Uses kernel approximation (Positive Random Features from Performer) for sub-quadratic time, with sparse correction on bipartite mask for accuracy (inspired by Scatterbrain).
- **Comparison**: Includes a basic Transformer for benchmarking.
- **Application**: Tested on a toy text sentiment dataset, where hyperedges represent local word groups (sliding window).
- **PyTorch Implementation**: Modular classes for easy integration.

## Installation
Requires PyTorch (tested on 2.0+ for efficiency, but compatible with 1.8+).
