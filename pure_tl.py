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
from transformer_lens import utils, HookedTransformer, HookedTransformerConfig, ActivationCache
import circuitsvis as cv
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from transformers import LlamaForCausalLM, LlamaTokenizer
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from huggingface_hub import snapshot_download

# %% hf model loading
checkpoint_location = snapshot_download("decapoda-research/llama-7b-hf")
with init_empty_weights():  # Takes up near zero memory
    model = LlamaForCausalLM.from_pretrained(checkpoint_location)
model = load_checkpoint_and_dispatch(
    model,
    checkpoint_location,
    device_map="auto", #single precision
    no_split_module_classes=["LlamaDecoderLayer"],
)
# %% putting hf model into transformer lens
model = HookedTransformer.from_pretrained("llama-7b-hf", hf_model=model)
# %%
