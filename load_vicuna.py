import os
import torch
import gc
from transformers import LlamaForCausalLM, LlamaTokenizer
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from huggingface_hub import snapshot_download



def efficient_model_loading(): #can change to cache model weights on disk
    checkpoint_location = snapshot_download("decapoda-research/llama-13b-hf")
    with init_empty_weights():  # Takes up near zero memory
        model = LlamaForCausalLM.from_pretrained(checkpoint_location)
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint_location,
        device_map="auto",
        no_split_module_classes=["LlamaDecoderLayer"],
    )
    tok = LlamaTokenizer.from_pretrained(checkpoint_location)
    return model, tok #returns by-reference

model, tok = efficient_model_loading()
os.system(f"mkdir {os.getcwd()}/llama-weights")
model.save_pretrained(f"{os.getcwd()}/llama-weights")

torch.cuda.empty_cache()
gc.collect()


os.system(f"mkdir {os.getcwd()}/vicuna-weights")

os.system("pip install fschat")
os.system(f"python -m fastchat.model.apply_delta \
    --base-model-path {os.getcwd()}/llama-weights \
    --target-model-path {os.getcwd()}/vicuna-weights \
    --delta-path lmsys/vicuna-13b-delta-v1.1")