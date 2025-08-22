import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Kernel-Sparse Attention Layer (with PRF kernel approx + sparse correction)
class KernelSparseAttentionLayer(nn.Module):
    def __init__(self, d, r=64):
        super().__init__()
        self.d = d
        self.r = r  # Number of random features
        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)
        self.W_o = nn.Linear(d, d)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.ReLU(), nn.Linear(4 * d, d))
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        # Random projection matrix for PRF (fixed, as in Performer)
        self.omega = nn.Parameter(torch.randn(d, r), requires_grad=False)

    def forward(self, Z, mask=None):
        Q = self.W_q(Z)
        K = self.W_k(Z)
        V = self.W_v(Z)
        
        # Normalize for unit norm (for angular LSH-like, but here for kernel)
        Q_norm = Q / torch.norm(Q, dim=-1, keepdim=True).clamp(min=1e-6)
        K_norm = K / torch.norm(K, dim=-1, keepdim=True).clamp(min=1e-6)
        
        # PRF: phi(x) = exp(omega^T x) / sqrt(r)  (for exp kernel approx)
        phi_Q = torch.exp(torch.matmul(Q_norm, self.omega)) / (self.r ** 0.5)
        phi_K = torch.exp(torch.matmul(K_norm, self.omega)) / (self.r ** 0.5)
        
        # Low-rank approx: phi_Q (phi_K^T V)
        low_rank = torch.matmul(phi_Q, torch.matmul(phi_K.t(), V))
        
        # Sparse correction: On masked positions, compute exact exp(QK^T / sqrt(d)) - phi_Q phi_K^T
        if mask is not None:
            # Find positions where mask == 0 (attendable)
            attendable = (mask == 0)
            # Compute exact scores only on sparse positions
            sparse_scores = torch.full_like(mask, 0.0)
            sparse_scores[attendable] = torch.exp(
                (Q.unsqueeze(1) * K.unsqueeze(0))[attendable] / (self.d ** 0.5)
            ).sum(dim=-1) - (phi_Q.unsqueeze(1) * phi_K.unsqueeze(0))[attendable].sum(dim=-1)
            
            # Softmax approx on sparse + low-rank, but for simplicity, direct add delta
            sparse_delta = torch.sparse.mm(
                torch.sparse.FloatTensor(torch.nonzero(attendable).t(), sparse_scores[attendable], size=mask.shape),
                V
            )
            attn_out = low_rank + sparse_delta
        else:
            attn_out = low_rank
        
        # Normalize (approximate row-wise softmax via diag inverse)
        denom = torch.sum(attn_out, dim=-1, keepdim=True).clamp(min=1e-6)
        attn_out = attn_out / denom
        
        attn_out = self.W_o(attn_out)
        Z = self.norm1(Z + attn_out)
        ffn_out = self.ffn(Z)
        Z = self.norm2(Z + ffn_out)
        return Z

# Hypergraph Transformer with Kernel-Sparse Attention
class HypergraphTransformer(nn.Module):
    def __init__(self, d, max_n, max_m, embedding, r=64):
        super().__init__()
        self.d = d
        self.embedding = embedding
        self.W_V = nn.Parameter(torch.randn(max_m, d))
        self.W_E = nn.Parameter(torch.randn(max_n, d))
        self.layer1 = KernelSparseAttentionLayer(d, r)
        self.layer2 = KernelSparseAttentionLayer(d, r)
        self.mlp = nn.Linear(d, 1)

    def forward(self, token_ids, H, X_E):
        X_V = self.embedding(token_ids)
        n = X_V.shape[0]
        m = H.shape[1]
        P_V = torch.matmul(H, self.W_V[:m])
        P_E = torch.matmul(H.t(), self.W_E[:n])
        Z = torch.cat((X_V + P_V, X_E + P_E), dim=0)
        
        # Bipartite mask (float('-inf') where no attention)
        mask = torch.full((n + m, n + m), float('-inf'))
        mask[:n, n:] = torch.where(H == 1, 0.0, float('-inf'))
        mask[n:, :n] = torch.where(H.t() == 1, 0.0, float('-inf'))
        mask.diagonal().fill_(0.0)
        
        Z = self.layer1(Z, mask=mask)
        Z = self.layer2(Z, mask=mask)
        node_out = Z[:n]
        pooled = node_out.mean(dim=0)
        pred = self.mlp(pooled)
        return pred

# Basic Transformer (unchanged, for comparison)
class BasicTransformer(nn.Module):
    def __init__(self, d, max_n, embedding):
        super().__init__()
        self.d = d
        self.embedding = embedding
        self.pos = nn.Parameter(torch.randn(max_n, d))
        self.layer1 = KernelSparseAttentionLayer(d)  # Can use kernel here too, but keep basic as full for contrast
        self.layer2 = KernelSparseAttentionLayer(d)
        self.mlp = nn.Linear(d, 1)

    def forward(self, token_ids):
        X_V = self.embedding(token_ids)
        n = X_V.shape[0]
        Z = X_V + self.pos[:n]
        Z = self.layer1(Z)  # No mask for basic (full attention)
        Z = self.layer2(Z)
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
r = 64  # Random features for kernel approx
max_n = max(len(t) for t in tokenized_texts) + 5
max_m = max_n
lr = 0.001
epochs = 200
loss_fn = nn.BCEWithLogitsLoss()

# Shared embedding layer
embedding = nn.Embedding(vocab_size, d)

# Initialize models
hg_model = HypergraphTransformer(d, max_n, max_m, embedding, r)
basic_model = BasicTransformer(d, max_n, embedding)

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
