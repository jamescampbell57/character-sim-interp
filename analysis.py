# %%
import torch
import os
import einops
import math



#d_vocab = 32000
#d_model = 5120
n_heads = 40
d_head = 128
n_layers = 40
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IGNORE = torch.tensor(-9999999, device=device)

activations_dir = f"{os.getcwd()}/data/prompt-1"

#for layer in range(activations_dir):
layer = 0
query = torch.load(f"{activations_dir}/model.layers.{layer}.self_attn.q_proj", map_location=device)
key = torch.load(f"{activations_dir}/model.layers.{layer}.self_attn.k_proj", map_location=device)
query = einops.rearrange(query, "seq (n_heads d_head) -> n_heads seq d_head", n_heads=n_heads)
key = einops.rearrange(key, "seq (n_heads d_head) -> n_heads d_head seq", n_heads=n_heads)

attn_scores = query @ key / math.sqrt(d_head)

all_ones = torch.ones(attn_scores.size(-2), attn_scores.size(-1), device=attn_scores.device)
mask = torch.triu(all_ones, diagonal=1,).bool().
attn_scores = attn_scores.masked_fill_(mask, IGNORE)
attn_probs = torch.nn.functional.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query.dtype)

print(attn_probs.shape)


