# %%
import os
import sys
tl_path = f"{os.getcwd()}/TransformerLens"
sys.path.insert(0, tl_path)

import torch
from jaxtyping import Float
from tqdm import tqdm
from IPython.display import display
from transformer_lens.loading_from_pretrained import get_pretrained_model_config
from transformer_lens.hook_points import HookPoint
from transformer_lens import HookedTransformer
from transformer_lens import utils, HookedTransformer, HookedTransformerConfig, ActivationCache
import circuitsvis as cv
from huggingface_hub import snapshot_download
from transformers import LlamaTokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
checkpoint_location = snapshot_download("decapoda-research/llama-7b-hf")
tok = LlamaTokenizer.from_pretrained(checkpoint_location)
tok.add_special_tokens({'pad_token': '[PAD]'})
cfg = get_pretrained_model_config(model_name="llama-7b-hf")
cfg.dtype = torch.float16
model = HookedTransformer(cfg, tok)

#model = HookedTransformer.from_pretrained("roneneldan/TinyStories-2Layers-33M")

# %%

text = '''Mary: "Have you seen the latest Spiderman movie?"
John: "No, how is it?"
Mary: The acting was great, but the plot was cliche. Are you going to see it?
John:'''
#how many times have I told you not to leave your rock-climbing gear in the study? This is a space of learning and solitude, not an extreme sports store."'''
text2 = '''Alex: "Hi, Bob!"

Bob: "Hey, Alex. How's your day going?"

Alex: "Not bad, thanks. Yours?"

Bob:'''



# %%
print(len(model.to_str_tokens(text2)))
# %%
completion = model.generate(text2, max_new_tokens=100)
print(completion)
# %%
def activation_to_disk(
    attn_pattern: Float[torch.Tensor, "batch heads seqQ seqK"],
    hook: HookPoint
) -> Float[torch.Tensor, "batch heads seqQ seqK"]:

    torch.save(attn_pattern, f"{hook.name}.pt")
    return attn_pattern

hooks = []
for layer in range(model.cfg.n_layers):
    hooks.append((f"blocks.{layer}.attn.hook_pattern", activation_to_disk))

logits = model.run_with_hooks(text, fwd_hooks=hooks)

# %%
print(logits.shape)