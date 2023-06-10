#%%
from typing import List, Tuple, Dict, Any, Optional
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Tuple, Callable
from functools import partial


@dataclass
class HookInfo:
    handle: torch.utils.hooks.RemovableHandle
    level: Optional[int] = None


class HookedModule(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self._hooks: List[HookInfo] = []
        self.context_level: int = 0

    @contextmanager
    def hooks(self, fwd: List[Tuple[str, Callable]] = [], bwd: List[Tuple[str, Callable]] = []):
        self.context_level += 1
        try:
            # Add hooks
            for hook_position, hook_fn in fwd:
                module = self._get_module_by_path(hook_position)
                handle = module.register_forward_hook(hook_fn)
                info = HookInfo(handle=handle, level=self.context_level)
                self._hooks.append(info)

            for hook_position, hook_fn in bwd:
                module = self._get_module_by_path(hook_position)
                handle = module.register_full_backward_hook(hook_fn)
                info = HookInfo(handle=handle, level=self.context_level)
                self._hooks.append(info)

            yield self
        finally:
            # Remove hooks
            for info in self._hooks:
                if info.level == self.context_level:
                    info.handle.remove()
            self._hooks = [h for h in self._hooks if h.level != self.context_level]
            self.context_level -= 1

    def _get_module_by_path(self, path: str) -> nn.Module:
        module = self.model
        for attr in path.split('.'):
            module = getattr(module, attr)
        return module

    def print_model_structure(self):
        print("Model structure:")
        for name, module in self.model.named_modules():
            print(f"{name}: {module.__class__.__name__}")

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


# if __name__ == "__main__":
#     bert_model = AutoModel.from_pretrained("bert-base-uncased")
#     hooked_model = HookedModule(bert_model)
#     hooked_model.print_model_structure()

#     def example_hook(module, input, output):
#         print(f"Hook called with tensor of shape: {output.shape}")

#     def example_hook2(module, input, output):
#         print(f"Hook2 called with tensor output shape {output[0].shape}")

#     tok = AutoTokenizer.from_pretrained("bert-base-uncased")
#     input_ids = torch.tensor([[101, 2054, 2003, 2026, 2171, 102]])

#     with hooked_model.hooks(fwd=[('encoder.layer.0.attention.self.query', example_hook)]):
#         output = hooked_model(input_ids)
#         with hooked_model.hooks(fwd=[('encoder.layer.9.output.dense', example_hook2)]):
#             output = hooked_model(input_ids)
#         output = hooked_model(input_ids)

#%%
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from huggingface_hub import snapshot_download

checkpoint_location = snapshot_download("decapoda-research/llama-7b-hf")

with init_empty_weights():  # Takes up near zero memory
    model = LlamaForCausalLM.from_pretrained(checkpoint_location)

model = load_checkpoint_and_dispatch(
    model,
    checkpoint_location,
    device_map="auto",
    dtype=torch.float16,
    no_split_module_classes=["LlamaDecoderLayer"],
)

tok = LlamaTokenizer.from_pretrained(checkpoint_location)


#%%
text = """Mary: I went to the doctor's the other day.
John: What did he say?"""

token_ids = tok(text, return_tensors="pt").to(model.device)
output = model(**token_ids)

#%%
hmodel = HookedModule(model)
# Print out the names modules you can add hooks to
# hmodel.print_model_structure()

#%%
cache = {}

def caching_hook_fnc(module, input, output, name=""):
    print("Hooking:", name)
    cache[name] = output[0].detach()


hook_pairs = [
    ("model.layers.31.self_attn", partial(caching_hook_fnc, name="model.layers.31.self_attn")),
]

with hmodel.hooks(fwd=hook_pairs):
    output = hmodel(**token_ids)

#%%
cache['model.layers.31.self_attn'].shape