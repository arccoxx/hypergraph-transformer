# Hypergraph Transformer with Efficient Sparse Attention

This repository implements an experimental Transformer layer extended to hypergraphs, inspired by the paper "Transformers are Graph Neural Networks" (arXiv:2506.22084). It views Transformers as GNNs on fully connected graphs and extends to hypergraphs by embedding nodes and hyperedges as tokens, with bipartite self-attention for higher-order relations.

## Features
- **Modular Layer**: `HypergraphTransformerLayer` is a reusable class for building hypergraph-capable Transformers. Supports optional bipartite masking for efficiency (using PyTorch's scaled_dot_product_attention).
- **Stackable Models**: Full `HypergraphTransformer` and `BasicTransformer` models with configurable number of layers (default: 3 for experiments).
- **Efficiency**: Bipartite mask restricts attention to node-hyperedge connections, reducing effective computation while maintaining O((n+m)^2 d) worst-case (sub-quadratic with sparse kernels in extensions).
- **Experiment Setup**: Includes training script for a toy text sentiment dataset, modeling sequences as hypergraphs with sliding-window hyperedges (size >2, e.g., window=3).
- **Comparison**: Benchmarks against a basic Transformer on the same task.
- **PyTorch Implementation**: Clean, modular code for easy extension (e.g., add kernel approx or deeper stacks).

## Installation
Requires PyTorch (tested on 2.0+ for flash attention, but compatible with 1.8+).

Clone the repo: 
git clone https://github.com/yourusername/hypergraph-transformer.git
cd hypergraph-transformer

## Usage
Run the training script (e.g., `train.py` containing the code above):

- Configurable: Set `num_layers=3` (or other) for experiments.
- Outputs: Loss curves and comparison table for Hypergraph vs. Basic models.

### Custom Usage: Building with the Layer
Example to construct a custom model:
```python
# Assume node features X_V, hyperedge features X_E, incidence H
layer = HypergraphTransformerLayer(d=32)
# Compute positional encodings externally if needed
Z = torch.cat((X_V + P_V, X_E + P_E), dim=0)
Z_out = layer(Z, H=H)  # Applies masked attention
