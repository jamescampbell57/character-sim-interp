
# pip install sentencepiece

#%%
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from huggingface_hub import snapshot_download

import os
import sys
tl_path = f"{os.getcwd()}/TransformerLens"
sys.path.insert(0, tl_path)
#from transformer_lens import HookedTransformer

checkpoint_location = snapshot_download("decapoda-research/llama-13b-hf")

with init_empty_weights():  # Takes up near zero memory
    model = LlamaForCausalLM.from_pretrained(checkpoint_location)

# %%

model = load_checkpoint_and_dispatch(
    model,
    checkpoint_location,
    device_map="auto",
    dtype=torch.float16,
    no_split_module_classes=["LlamaDecoderLayer"],
)

#%%
tok = LlamaTokenizer.from_pretrained(checkpoint_location)


#%%
#text = '''Mary is a bookish and introverted historian with a sharp wit. John is an adventurous, outgoing photographer with a passion for extreme sports.
#
#Mary: "John, how many times have I told you not to leave your rock-climbing gear in the study? This is a space of learning and solitude, not an extreme sports store."
#
#Jasper: "Ah, come on, Mary. Can't you see how this gear tells its own history? Now, that's a story!"'''

text = '''Mary: "Have you seen the latest Spiderman movie?"
John: "No, how is it?"
Mary:'''

token_ids = tok(text, return_tensors="pt").to(model.device)
output = model.generate(
    **token_ids,
    max_new_tokens=30,
)

print(tok.batch_decode(output)[0])


# %%
