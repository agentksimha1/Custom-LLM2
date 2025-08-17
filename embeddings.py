import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import tiktoken
import streamlit as st

# --- Tokenizer ---
enc = tiktoken.get_encoding("gpt2")  # ~50k vocab

# --- Load text (placeholder for training) ---
file_path = "the-verdict.txt"
if not os.path.exists(file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("This is a placeholder text file for the model training. In a real scenario, you would have your training data here.")
text_data = ""
with open(file_path, 'r', encoding='utf-8') as f:
    text_data = f.read()

# --- Config ---
GPT_2_config = {
    'vocab_size': enc.n_vocab,       # ensure matches tokenizer
    'n_head': 12,
    'drop': 0.1,
    'n_layers': 12,
    'context_length': 256,
    'qkv_bias': False,
    'emb_dim': 768
}

# --- Utilities for encode/decode ---
def text_to_token_ids(text: str) -> torch.LongTensor:
    ids = enc.encode(text, allowed_special={'<|endoftext|>'})
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)

def token_to_text(token_ids: torch.LongTensor) -> str:
    if token_ids.dim() == 2:
        token_ids = token_ids.squeeze(0)
    return enc.decode(token_ids.tolist())

# --- Model Layers ---
class Layernorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-8
        self.gamma = nn.Parameter(torch.ones(emb_dim))
        self.beta = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return (x - mean) / (std + self.eps) * self.gamma + self.beta

class Geluactivation(nn.Module):
    def forward(self, x):
        return F.gelu(x)

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg['emb_dim']
        self.layers = nn.Sequential(
            nn.Linear(d, 4 * d),
            Geluactivation(),
            nn.Linear(4 * d, d)
        )

    def forward(self, x):
        return self.layers(x)

class Causalattention(nn.Module):
    def __init__(self, d_in, d_out, context_length, drop, qkv_bias=False):
        super().__init__()
        self.context_length = context_length
        self.drop = nn.Dropout(drop)
        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        B, T, _ = x.shape
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        scores = q @ k.transpose(1, 2)  # (B, T, T)
        scores = scores / (k.shape[-1] ** 0.5)
        mask = self.mask[:T, :T].bool()
        scores = scores.masked_fill(mask.unsqueeze(0).to(scores.device), float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = self.drop(attn)
        return attn @ v  # (B, T, d_out)

class Multiheadattention(nn.Module):
    def __init__(self, d_model, d_head, context_length, drop, num_heads, qkv_bias=False):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.heads = nn.ModuleList([
            Causalattention(d_model, d_head, context_length, drop, qkv_bias)
            for _ in range(num_heads)
        ])
        self.out_proj = nn.Linear(d_head * num_heads, d_model)

    def forward(self, x):
        contexts = [h(x) for h in self.heads]
        cat = torch.cat(contexts, dim=-1)
        return self.out_proj(cat)

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_model = cfg['emb_dim']
        self.ln1 = Layernorm(d_model)
        self.ln2 = Layernorm(d_model)
        self.drop = nn.Dropout(cfg['drop'])
        head_dim = d_model // cfg['n_head']
        self.att = Multiheadattention(
            d_model, head_dim, cfg['context_length'], cfg['drop'], cfg['n_head'], cfg['qkv_bias']
        )
        self.ffn = FeedForward(cfg)

    def forward(self, x):
        x = x + self.drop(self.att(self.ln1(x)))
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x

class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg['emb_dim']
        self.tok_emb = nn.Embedding(cfg['vocab_size'], d)
        self.pos_emb = nn.Embedding(cfg['context_length'], d)
        self.drop = nn.Dropout(cfg['drop'])
        self.trfs_blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg['n_layers'])])
        self.final_norm = Layernorm(d)
        self.lm_head = nn.Linear(d, cfg['vocab_size'], bias=False)

    def forward(self, in_idx: torch.LongTensor):
        if in_idx.dim() == 1:
            in_idx = in_idx.unsqueeze(0)
        if in_idx.dim() == 3 and in_idx.size(-1) == 1:
            in_idx = in_idx.squeeze(-1)
        assert in_idx.dim() == 2, f"Expected (B,T) token ids, got shape {in_idx.shape}"

        B, T = in_idx.shape
        T = min(T, self.cfg['context_length'])
        in_idx = in_idx[:, -T:]

        tok = self.tok_emb(in_idx)
        pos = self.pos_emb(torch.arange(T, device=in_idx.device)).unsqueeze(0).expand(B, -1, -1)
        x = self.drop(tok + pos)
        for block in self.trfs_blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.lm_head(x)

# --- Generation function ---
@torch.no_grad()
def generate_simple_text(model: GPT, idx: torch.LongTensor, max_new_tokens: int, context_size: int, device: torch.device):
    if idx.dim() == 1:
        idx = idx.unsqueeze(0)
    if idx.dim() == 3 and idx.size(-1) == 1:
        idx = idx.squeeze(-1)
    idx = idx.to(device).long()

    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        logits = model(idx_cond)
        next_token_logits = logits[:, -1, :]
        probs = torch.softmax(next_token_logits, dim=-1)
        idx_next = torch.argmax(probs, dim=-1, keepdim=True)  # greedy decoding
        idx = torch.cat([idx, idx_next], dim=-1)
    return idx

# --- Streamlit Application ---
st.set_page_config(page_title="Simple LLM App", layout="wide")
st.title("GPT-2-like Model Playground")
st.write("Enter a prompt and let the model generate text!")

@st.cache_resource
def load_model():
    """Initializes and loads the model weights directly from a .pth file."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GPT(GPT_2_config).to(device)

    weights_path = "model_weights1.pth"
    if not os.path.exists(weights_path):
        print(f"{weights_path} not found. Creating dummy weights.")
        torch.save(model.state_dict(), weights_path)

    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)

    return model, device

# Load the model and device
model, device = load_model()

# User input
prompt = st.text_area("Enter your starting text:", "Every effort moves you closer to your goal.", height=150)
max_tokens = st.slider("Max new tokens to generate", min_value=1, max_value=500, value=150)

if st.button("Generate Text"):
    if prompt:
        with st.spinner("Generating text..."):
            try:
                encoded_prompt = text_to_token_ids(prompt)
                context_size = model.pos_emb.weight.shape[0]
                generated_tokens = generate_simple_text(
                    model,
                    encoded_prompt,
                    max_new_tokens=max_tokens,
                    context_size=context_size,
                    device=device
                )
                generated_text = token_to_text(generated_tokens)
                st.subheader("Generated Text:")
                st.write(generated_text)
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter some text to generate.")
