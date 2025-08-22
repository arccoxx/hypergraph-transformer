import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Hypergraph-Capable Transformer Layer (Modular, Reusable)
class HypergraphTransformerLayer(nn.Module):
    """
    A single Transformer layer extended for hypergraphs.
    - Embeds nodes and hyperedges as tokens.
    - Applies self-attention with optional bipartite mask for efficiency.
    - Can handle graphs (hyperedges of size 2) or hypergraphs (>2).
    """
    def __init__(self, d, use_mask=True):
        super().__init__()
        self.d = d
        self.use_mask = use_mask
        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)
        self.W_o = nn.Linear(d, d)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.ReLU(), nn.Linear(4 * d, d))
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)

    def forward(self, Z, H=None, mask=None):
        """
        Z: Combined [node_emb + P_V; hyperedge_emb + P_E] ( (n+m) x d )
        H: Incidence matrix (n x m); used to compute mask if not provided.
        mask: Optional precomputed attention mask ((n+m) x (n+m)).
        """
        if self.use_mask and mask is None and H is not None:
            n, m = H.shape
            mask = torch.full((n + m, n + m), float('-inf'))
            mask[:n, n:] = torch.where(H == 1, 0.0, float('-inf'))
            mask[n:, :n] = torch.where(H.t() == 1, 0.0, float('-inf'))
            mask.diagonal().fill_(0.0)

        Q = self.W_q(Z)
        K = self.W_k(Z)
        V = self.W_v(Z)
        attn_out = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask, dropout_p=0.0)
        attn_out = self.W_o(attn_out)
        Z = self.norm1(Z + attn_out)
        ffn_out = self.ffn(Z)
        Z = self.norm2(Z + ffn_out)
        return Z

# Full Hypergraph Transformer Model (Stackable Layers)
class HypergraphTransformer(nn.Module):
    def __init__(self, d, max_n, max_m, embedding, num_layers=3, use_mask=True):
        super().__init__()
        self.d = d
        self.embedding = embedding
        self.W_V = nn.Parameter(torch.randn(max_m, d))
        self.W_E = nn.Parameter(torch.randn(max_n, d))
        self.layers = nn.ModuleList([HypergraphTransformerLayer(d, use_mask) for _ in range(num_layers)])
        self.mlp = nn.Linear(d, 1)

    def forward(self, token_ids, H, X_E):
        X_V = self.embedding(token_ids)
        n = X_V.shape[0]
        m = H.shape[1]
        P_V = torch.matmul(H, self.W_V[:m])
        P_E = torch.matmul(H.t(), self.W_E[:n])
        Z = torch.cat((X_V + P_V, X_E + P_E), dim=0)
        
        # Precompute mask if using
        mask = None
        if self.layers[0].use_mask:
            mask = torch.full((n + m, n + m), float('-inf'))
            mask[:n, n:] = torch.where(H == 1, 0.0, float('-inf'))
            mask[n:, :n] = torch.where(H.t() == 1, 0.0, float('-inf'))
            mask.diagonal().fill_(0.0)
        
        for layer in self.layers:
            Z = layer(Z, H, mask)
        
        node_out = Z[:n]
        pooled = node_out.mean(dim=0)
        pred = self.mlp(pooled)
        return pred

# Basic Transformer Model (For Comparison, Stackable Layers)
class BasicTransformer(nn.Module):
    def __init__(self, d, max_n, embedding, num_layers=3):
        super().__init__()
        self.d = d
        self.embedding = embedding
        self.pos = nn.Parameter(torch.randn(max_n, d))
        self.layers = nn.ModuleList([HypergraphTransformerLayer(d, use_mask=False) for _ in range(num_layers)])
        self.mlp = nn.Linear(d, 1)

    def forward(self, token_ids):
        X_V = self.embedding(token_ids)
        n = X_V.shape[0]
        Z = X_V + self.pos[:n]
        for layer in self.layers:
            Z = layer(Z)  # No mask for basic (full attention)
        pooled = Z.mean(dim=0)
        pred = self.mlp(pooled)
        return pred

# Function to create hypergraph incidence and X_E for a sequence
def create_hypergraph(l, d, window=3):
    if l < window:
        H = torch.zeros((l, 0))
        X_E = torch.zeros((0, d))
        return H, X_E
    m = l - window + 1
    H = torch.zeros(l, m)
    for i in range(m):
        H[i:i+window, i] = 1
    X_E = torch.zeros(m, d)
    return H, X_E

# Dataset: Simple text sentiment dataset
texts = [
    "great movie love it",
    "bad film hate it",
    "awesome story good",
    "terrible plot bad",
    "love the actors great",
    "hate the ending terrible",
    "fantastic visuals awesome",
    "boring script bad",
    "exciting adventure love",
    "disappointing sequel hate",
    "brilliant direction great",
    "awful acting terrible",
    "inspiring tale awesome",
    "predictable story bad",
    "heartwarming romance love"
]
labels = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]

# Build vocab
all_words = set(word for text in texts for word in text.split())
word_to_ix = {word: i for i, word in enumerate(all_words)}
vocab_size = len(word_to_ix)

# Tokenize texts
tokenized_texts = [[word_to_ix[word] for word in text.split()] for text in texts]

# Hyperparameters
d = 32
max_n = max(len(t) for t in tokenized_texts) + 5  # Buffer
max_m = max_n  # Approx
lr = 0.001
epochs = 200
loss_fn = nn.BCEWithLogitsLoss()
num_layers = 3  # Configurable for experiments

# Shared embedding layer
embedding = nn.Embedding(vocab_size, d)

# Initialize models
hg_model = HypergraphTransformer(d, max_n, max_m, embedding, num_layers=num_layers)
basic_model = BasicTransformer(d, max_n, embedding, num_layers=num_layers)

# Optimizers
hg_optimizer = optim.Adam(hg_model.parameters(), lr=lr)
basic_optimizer = optim.Adam(basic_model.parameters(), lr=lr)

# Training function
def train_model(model, optimizer, is_hypergraph=False):
    losses = []
    for epoch in range(epochs):
        total_loss = 0.0
        for i, token_ids in enumerate(tokenized_texts):
            token_ids = torch.tensor(token_ids)
            label = torch.tensor([labels[i]])
            optimizer.zero_grad()
            if is_hypergraph:
                H, X_E = create_hypergraph(len(token_ids), d)
                pred = model(token_ids, H, X_E)
            else:
                pred = model(token_ids)
            loss = loss_fn(pred, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(texts)
        losses.append(avg_loss)
        if epoch % 50 == 0:
            print(f'Epoch {epoch}, Avg Loss: {avg_loss:.4f}')
    return losses

# Train Hypergraph Transformer
print("Training Hypergraph Transformer:")
hg_losses = train_model(hg_model, hg_optimizer, is_hypergraph=True)
print(f'Final Hypergraph Loss: {hg_losses[-1]:.4f}')

# Train Basic Transformer
print("\nTraining Basic Transformer:")
basic_losses = train_model(basic_model, basic_optimizer, is_hypergraph=False)
print(f'Final Basic Loss: {basic_losses[-1]:.4f}')

# Compare performance
print("\nPerformance Comparison:")
print("Epoch\tHypergraph Loss\tBasic Loss")
for epoch in range(0, epochs, 50):
    print(f"{epoch}\t{hg_losses[epoch]:.4f}\t\t{basic_losses[epoch]:.4f}")
print(f"{epochs-1}\t{hg_losses[-1]:.4f}\t\t{basic_losses[-1]:.4f}")
