# pip install sentencepiece

#%%
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from huggingface_hub import snapshot_download

checkpoint_location = snapshot_download("decapoda-research/llama-13b-hf")

with init_empty_weights():  # Takes up near zero memory
    model = LlamaForCausalLM.from_pretrained(checkpoint_location)

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
text = """Mary: I went to the doctor's the other day.
John: What did he say?"""

token_ids = tok(text, return_tensors="pt").to(model.device)
output = model.generate(
    **token_ids,
    max_new_tokens=60,
)

#%%
print(tok.batch_decode(output)[0])
